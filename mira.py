import os
import re
import sys
import cdec
import gzip
import random
import argparse
import cdec.score
import itertools
import multiprocessing as mp
from collections import namedtuple, defaultdict

Hypothesis = namedtuple('Hypothesis', 'string, features, metric_score')
# string: target language output
# features: SparseVector of (feature, value) pairs
# metric_score: BLEU score of string with respect to reference(s)
## oracle_feat_diff: oracle.features - features
## oracle_loss = oracle.metric_score - metric_score

decoder = None
def make_decoder(config):
	global decoder
	decoder = cdec.Decoder(config)

def parse_sgml(sgml):
	match = re.match(r'\s*<seg([^>]*)>(.*)</seg>\s*', sgml)
	if match:
		properties = match.group(1)
		source = match.group(2)

		seg_id = re.search(r' id="?([0-9]+)', properties).group(1)
		seg_id = int(seg_id)	

		grammar = re.search(r' grammar="([^"]*)"', properties).group(1)
		return (source, seg_id, grammar)
	else:
		return (sgml, 0, None)

def make_kbest_list(hypergraph, k, unique):
	kbest_list = []
	kbest_translations = hypergraph.unique_kbest(k) if unique else hypergraph.kbest(k)
	kbest_features = hypergraph.unique_kbest_features(k) if unique else hypergraph.kbest_features(k)
	for translation, features in zip(kbest_translations, kbest_features):
		#features = {key: value for (key, value) in features}
		new_hyp = Hypothesis(translation, features, None)
		kbest_list.append(new_hyp)
	return kbest_list

def score_kbest_list(kbest_list, refs, scorer):
	scored_kbest_list = []
	for i in range(len(kbest_list)):
		translation = kbest_list[i].string
		features = kbest_list[i].features
		metric_score = scorer(refs).evaluate(translation).score * 100.0
		new_hyp = Hypothesis(translation, features, metric_score)
		scored_kbest_list.append(new_hyp)
	return scored_kbest_list

# hyp1 = hope, hope2 = fear
# Note, this returns the opposite of the C++ ComputeDelta function.
# In C++, delta is in [-1, 0], but here it's in [0, 1]
def compute_delta(hyp1, hyp2, oracle_best, max_step_size, weights):
	loss = hyp1.metric_score - hyp2.metric_score
	margin = hyp1.features.dot(weights) - hyp2.features.dot(weights)
	num = margin + loss

	diff = hyp1.features - hyp2.features
	diffsqnorm = sum(v ** 2 for (k, v) in diff)
	delta = num / (diffsqnorm * max_step_size) if diffsqnorm > 0 else 0.0
	delta = max(0, min(delta, 1))

	return delta

def compute_hope(hyp, oracle_best, weights):
	if args.hope_select == 1:
		return hyp.features.dot(weights) + args.metric_scale * hyp.metric_score
	elif args.hope_select == 2:
		return hyp.metric_score
	else:
		raise Exception("Invalid hope_select!")

def compute_fear(hyp, oracle_best, weights):
	if args.fear_select == 1:
		return hyp.features.dot(weights) - args.metric_scale * hyp.metric_score
	elif args.fear_select == 2:
		return -hyp.metric_score
	elif args.fear_select == 3:
		return hyp.features.dot(weights)
	else:
		raise Exception("Invalid fear_select!")

def dump_weights(weight_vector, file_name):
	weights_file = open(file_name, 'w')
	for key, value in weight_vector:
		if value != 0.0:
			weights_file.write('%s %f\n' % (key, value))
	weights_file.close()

# Cython objects (e.g. SufficientStats or SparseVector) cannot be passed between python processes.
# As such, we convert them to native python objects (dictionaries/lists) before and after passing
# them to or returning them from process_line
def process_line(params):
	global decoder
	pass_num, line_num, line, current_weights_ = params

	# Parse the current weight vector and feed it to the decoder
	current_weights = cdec._cdec.SparseVector()
	for k, v in current_weights_.iteritems():
		current_weights[k] += v
	decoder.weights = current_weights

	# Parse the line in the devset
	source_sgml, refs = [part.strip() for part in line.split('|||', 1)]
	refs = [part.strip() for part in refs.split('|||')]		
	source, seg_id, grammar_file = parse_sgml(source_sgml)

	# Read in the grammar whose filename is listed in the SGML wrapping the source segment
	grammar = None
	if grammar_file != None:
		with gzip.open(grammar_file) as f:
			grammar = f.read()

	# Decode the source and make a kbest list
	forest = decoder.translate(source, grammar=grammar)
	scorer = cdec.score.Scorer('IBM_BLEU')
	kbest_list = make_kbest_list(forest, args.k, not args.no_unique)	
	kbest_list = score_kbest_list(kbest_list, refs, scorer)

	# Find the top best (max model score), oracle best (max BLEU score), hope, and fear among the kbest list
	top_best = kbest_list[0]
	oracle_best = max(kbest_list, key=lambda hyp: hyp.metric_score)
	hope = max(kbest_list, key=lambda hyp: compute_hope(hyp, oracle_best, decoder.weights))
	fear = max(kbest_list, key=lambda hyp: compute_fear(hyp, hope, decoder.weights))
	hyps = (top_best, oracle_best, hope, fear)

	# Calculate BLEU of the top best
	evaluator = cdec.score.BLEU(refs)
	bleu_stats = evaluator.evaluate(top_best.string)

	# Compute update size
	delta = compute_delta(hope, fear, oracle_best, args.max_step_size, decoder.weights)
	step_size = delta * args.max_step_size	

	# Update the weights vector, moving step_size in the direction away from fear and towards hope
	update = (hope.features - fear.features) * step_size

	# Convert objects to native python objects before returning
	bleu_stats = [bleu_stats[i] for i in range(len(bleu_stats))]
	update = {key: value for key, value in update}	
	return line_num, hyps, bleu_stats, update

if __name__ == "__main__":
	parser = argparse.ArgumentParser("A python implementation of MIRA", add_help=False)
	parser.add_argument('-w', '--weights', help='Initial weights')
	parser.add_argument('-d', '--devset', required=True, help='Dev set')
	parser.add_argument('-c', '--config', required=True, help='cdec config file')
	parser.add_argument('-k', type=int, default=250, help='k-best list size')
	parser.add_argument('-j', '--jobs', type=int, default=1, help='Number of cores to use during decoding')
	parser.add_argument('--no-unique', action='store_true', help='Do not unique k-best lists')
	parser.add_argument('-s', '--max-step-size', type=float, default=0.01, help='max step size during update')
	parser.add_argument('-p', '--passes', type=int, default=20, help='number of passes to make over the dev set')
	parser.add_argument('-h', '--hope-select', type=int, default=1, choices=[1, 2], help='1 = Model Score + BLEU\n2 = BLEU')
	parser.add_argument('-f', '--fear-select', type=int, default=1, choices=[1, 2, 3], help='1 = Model Score - BLEU\n2 = -BLEU\n3 = Model Score ("Prediction-based")')
	parser.add_argument('-o', '--output-dir', type=str, default='./mira-work', help='output directory')
	parser.add_argument('--shuffle', action='store_true', help='Shuffle the devset each iteration to remove ordering-based effects')
	parser.add_argument('--metric_scale', type=float, default=1.0, help='When computing hope/fear, scale the metric score by this factor')
	args = parser.parse_args()

	if os.path.exists(args.output_dir):
		print >>sys.stderr, 'ERROR: working directory %s already exists!' % args.output_dir
		sys.exit(1)
	os.mkdir(args.output_dir)

	with open(args.config) as f:
		cdec_ini = f.read()

	# Initialize weight vector
	if args.weights:
		decoder2 = cdec.Decoder(cdec_ini)
		decoder2.read_weights(args.weights)
		initial_weights = decoder2.weights.tosparse()
	else:
		initial_weights = cdec._cdec.SparseVector()
	dump_weights(initial_weights, os.path.join(args.output_dir, 'weights.0'))

	# Read in the devset
	devset = [line.decode('utf-8').strip() for line in open(args.devset)]

	# Make a thread pool with one decoder for each core
	pool = mp.Pool(args.jobs, make_decoder, (cdec_ini,))

	for p in range(args.passes):	
		update_sum = cdec._cdec.SparseVector()
		updates_done = 0

		# Note: Creating a SufficientStats object normally gives a NullPointerException
		# later, because the SufficientStats.metric property is never initialized.
		# Initializing like this creates an empty BLEU SufficientStats object correctly.
		total_bleu_stats = cdec.score.BLEU('').evaluate('')

		# This little generator function produces parameter sets, each of which gets passed
		# to process_line in praallel. Note that we convert the weights to a dict before passing.
		def gen_params(devset):
			# Shuffle the devset if requested
			reordered_devset = list(enumerate(devset))
			if args.shuffle:
				random.shuffle(reordered_devset)

			for line_num, line in reordered_devset:
				current_weights = (initial_weights + update_sum / updates_done) if updates_done > 0 else initial_weights
				yield p, line_num, line, {key: value for key, value in current_weights}

		# Open up a file pointer, to which we'll dump hypotheses of interest for each sentence
		run_raw_output_file = gzip.open(os.path.join(args.output_dir, 'run.raw.%d.gz' % p), 'w')

		# Decode the devset in parallel, and keep track of the sum of both BLEU
		# sufficient stats as well as weight vector updates
		for line_num, hyps, bleu_stats_, update_ in pool.imap(process_line, gen_params(devset)):
			run_raw_output_file.write(str(line_num) + ' ||| ')
			run_raw_output_file.write(' ||| '.join(hyp.string.encode('utf-8') for hyp in hyps) + '\n')
			run_raw_output_file.flush()

			bleu_stats = cdec.score.BLEU('').evaluate('')
			for i, v in enumerate(bleu_stats_):
				bleu_stats[i] += v
			total_bleu_stats += bleu_stats

			update = cdec._cdec.SparseVector()
			for k, v in update_.iteritems():
				update[k] += v
			update_sum += update
			updates_done += 1

		# At the end of each iteration, output the BLEU score and dump the final weights
		run_raw_output_file.close()
		print 'BLEU score after iteration %d: %f' % (p + 1, 100.0 * total_bleu_stats.score)
		sys.stdout.flush()
		final_weights = (initial_weights + update_sum / updates_done) if updates_done > 0 else initial_weights	
		dump_weights(final_weights, os.path.join(args.output_dir, 'weights.%d' % (p + 1)))
		initial_weights = final_weights
