import tensorflow as tf
from include.Model import build_SE, build_AE, training
from include.Test import get_hits, get_combine_hits , get_hits_mrr, get_combine_hits_mrr
import time
from include.Load import *
import argparse

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)


if __name__ == '__main__':
	t = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument("--lan", type=str, default="zh_en", required=False,
						help="input language dir, ('en_fr_15k_V1', 'zh_en')")
	parser.add_argument("--ite", type=int, default=1, help="the number of iterations")


	args = parser.parse_args()
	iteround = str(args.ite)

	class Config:
		language = args.lan  # zh_en | ja_en | fr_en en_de_15k_V1 SRPRS/dbp_yg_15k_V1
		e1 = 'data/' + language + '/ent_ids_1'
		e2 = 'data/' + language + '/ent_ids_2'
		r1 = 'data/' + language + '/rel_ids_1'
		r2 = 'data/' + language + '/rel_ids_2'
		ill = 'data/' + language + '/ill_ent_ids'
		ref = 'data/' + language + '/ref_ent_ids'
		sup = 'data/' + language + '/sup_ent_ids' + iteround
		kg1 = 'data/' + language + '/triples_1' + iteround
		kg2 = 'data/' + language + '/triples_2' + iteround
		epochs_se = 300
		se_dim = 300
		act_func = tf.nn.relu
		gamma = 3.0  # margin based loss
		k = 25  # number of negative samples for each positive one
		seed = 3  # 30% of seeds

	e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
	print(e)
	# np.random.shuffle(ILL)
	# ILL = loadfile(Config.ill, 2)
	# illL = len(ILL)
	# train = np.array(ILL[:illL // 10 * Config.seed])
	# test = ILL[illL // 10 * Config.seed:]
	train = np.array(loadfile(Config.sup, 2))
	test = loadfile(Config.ref, 2)
	KG1 = loadfile(Config.kg1, 3)
	KG2 = loadfile(Config.kg2, 3)
	# build SE
	output_layer, loss = build_SE(Config.se_dim, Config.act_func, Config.gamma, Config.k, e, train, KG1 + KG2)
	se_vec, J = training(output_layer, loss, 25, Config.epochs_se, train, e, Config.k)

	np.save('data/' + Config.language + "/ents_vec" + iteround + '.npy', se_vec)

	print('loss:', J)
	print('Result of SE:')
	#get_hits(se_vec, test)
	get_hits_mrr(se_vec, test)
	print("SE total time = {:.3f} s".format(time.time() - t))
