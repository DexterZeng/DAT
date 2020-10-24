from utils import *
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lan", type=str, default="zh_en", required=False,
                        help="input language dir, ('en_fr_15k_V1', 'zh_en')")
    parser.add_argument("--ite", type=int, default=1, help="the number of iterations")
    args = parser.parse_args()


    iteround = str(args.ite)
    iteroundnext = str(args.ite+1)

    path = 'data/'+args.lan+'/' #en_de_15k_V1
    pathab = path

    gamma = 0.8
    hidden = 300
    batchsize = 32
    learning_rate = 0.1

    # textual embedding
    if '15k' in args.lan:
        name_embed = loadNe(path + 'name_vec_cpm_6.txt')
    else:
        name_embed = loadNe(path + args.lan.split('_')[0] + '_vectorList.json')
        # print(name_embed.shape)
    name_shape = 300

    # structural embeddding
    a = np.load(pathab + 'ents_vec'+iteround+'.npy')
    # previously aligned entities, their corresponding ranks
    recordedranks = []; pairs = []
    inf = open(pathab + 'condi_ranks'+iteround+'.txt')
    for line in inf:
        strs = line.strip().split('\t')
        recordedranks.append(int(strs[1]))
        pairs.append([strs[0], int(strs[1])])
    filled = copy.deepcopy(recordedranks)
    pre = np.array(filled)

    # get entities
    ents1 = getent(path + 'ent_ids_1'); ents2 = getent(path + 'ent_ids_2')

    ## get seeds
    # seeds here refer to two things:
    # 1) the seed entities from previous round, the test entities in seeds will not participate the testing in this round
    # 2) the seed entities for training the co-attention network
    pathseed = pathab + 'sup_ent_ids'+iteround
    seedleft_vec, seedright_vec, seedleft_namevec, seedright_namevec,seedleft, \
    seedright, seedDict,seedDictrev,seedleft_retr, seedright_retr = obtainseed(path, a, name_embed,pathseed)

    # pad embeddings to the largest dimension, construct feature matrices for entities (training data for attention)
    left_vec = np.concatenate([np.expand_dims(np.pad(seedleft_vec, ((0,0), (0, seedleft_namevec.shape[-1] - seedleft_vec.shape[-1])), 'constant'),axis=1), np.expand_dims(seedleft_namevec,axis=1)],axis=1)
    right_vec = np.concatenate([np.expand_dims(np.pad(seedright_vec, ((0,0), (0, seedleft_namevec.shape[-1] - seedleft_vec.shape[-1])), 'constant'),axis=1), np.expand_dims(seedright_namevec,axis=1)],axis=1)

    ### get test set
    seed = open(path + 'ref_ent_ids')
    evaleft = []; evaright = []
    evaleft_retr = []; evaright_retr = [] # for RSNs, the ids of entities are reordered

    for coun, line in enumerate(seed):
        if coun>= 1050:
            strs = line.strip().split('\t')
            if strs[0] not in seedDict.keys():
                evaleft.append(int(strs[0]))
                evaleft_retr.append(int(strs[0]))
            if strs[1] not in seedDict.values(): # note that the error of selected entity pairs could propagate
                evaright.append(int(strs[1]))
                evaright_retr.append(int(strs[1]))
    print('New length:' + str(len(evaleft))) # the entities that were added into the seeds will not be evaluated again
    evaleft_vec = a[evaleft_retr]; evaright_vec = a[evaright_retr]
    evaleft_namevec = name_embed[evaleft]; evaright_namevec = name_embed[evaright]

    # concatenate Feature matrix construction for test
    eva_left_vec = np.concatenate([np.expand_dims(np.pad(evaleft_vec, ((0,0), (0, seedleft_namevec.shape[-1] - seedleft_vec.shape[-1])), 'constant'),axis=1), np.expand_dims(evaleft_namevec,axis=1)],axis=1)
    eva_right_vec = np.concatenate([np.expand_dims(np.pad(evaright_vec, ((0,0), (0, seedleft_namevec.shape[-1] - seedleft_vec.shape[-1])), 'constant'),axis=1), np.expand_dims(evaright_namevec,axis=1)],axis=1)
    oneHot, id2fre = get_freoneHot(path)  # degree vec

    # construct loss
    newloss, train_step, aep_left,aep_right, att_left = constructloss(oneHot, name_shape, gamma, batchsize,learning_rate)

    # main operation
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        # print('initializing...')
        sess.run(init)
        num_ite = len(seedleft)//batchsize
        max_hits1, times, max_times = 0, 0, 3 # early stopping

        for epoch in range(500):
            for ite in range(num_ite):
                left_vec_train = left_vec[ite*batchsize:(ite+1)*batchsize] # feature matrices
                right_vec_train = right_vec[ite*batchsize:(ite+1)*batchsize]
                attleft = np.array(seedleft[ite*batchsize:(ite+1)*batchsize]) # ids
                attright = np.array(seedright[ite*batchsize:(ite+1)*batchsize])
                feed_dic_train = {"K1:0": left_vec_train, "K2:0": right_vec_train, "seedleft:0": attleft,
                                "seedright:0": attright, "datasize:0": len(attleft)}
                sess.run(train_step, feed_dict = feed_dic_train)
                lossvalue = sess.run(newloss, feed_dict = feed_dic_train)
                # if ite == num_ite-1:
                #     print(lossvalue)

            # training： use batches just for training (pairwise)/ use all to check the results
            aep_fuse = sess.run(aep_left, feed_dict = {"K1:0": left_vec, "K2:0": right_vec, "seedleft:0": np.array(seedleft),
                                                    "seedright:0": np.array(seedright), "datasize:0": len(left_vec)})
            probs_eva = aep_fuse - aep_fuse[range(len(seedleft)), range(len(seedleft))].reshape(len(aep_fuse), 1)
            ranks = (probs_eva >= 0).sum(axis=1)
            MR, H10, MRR = cal_performance(ranks, top=10)
            _, H1, _ = cal_performance(ranks, top=1)
            # msg = 'Train: Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
            # print(msg + '\n')

            # early stopping
            hits1 = H1
            if hits1 > max_hits1:
                max_hits1 = hits1
                times = 0
            else:
                times += 1
            if times >= max_times:
                break

        print('Seed Dic: ' + str(len(seedDict)))
        # add to seeds, augment training set first
        aep_fuse = sess.run(aep_right, feed_dict={"K1:0": left_vec, "K2:0": right_vec, "seedleft:0": np.array(seedleft),
                                                  "seedright:0": np.array(seedright), "datasize:0": len(left_vec)})
        seedDict, seedDictrev = enrichseed(probs_eva, aep_fuse, seedleft, seedright, seedDict, seedDictrev)


        # then test set
        aep_fuse_eva = sess.run(aep_left, feed_dict = {"K1:0": eva_left_vec, "K2:0": eva_right_vec, "seedleft:0": np.array(evaleft),
                                                "seedright:0": np.array(evaright), "datasize:0": len(eva_left_vec)})


        aep_fuse_evar = sess.run(aep_right, feed_dict = {"K1:0": eva_left_vec, "K2:0": eva_right_vec, "seedleft:0": np.array(evaleft),
                                                "see"
                                                "dright:0": np.array(evaright), "datasize:0": len(eva_left_vec)})

        ind_left, ind_right, gap_left, gap_right, ranks_left,truths = enrichtestv1(aep_fuse_eva, aep_fuse_evar, evaleft, evaright,pre)

        counter = 0
        truecounter = 0
        dicrank = {} # records the results of those which have been added to the training set!!!
        for i in range(len(ind_left)):
            if ind_right[ind_left[i]] == i:
                if gap_left[i] >= 0.05 and gap_right[i] >= 0.05:
                    counter += 1
                    dicrank[str(evaleft[i])] = ranks_left[i] # records the ranks of confident results, should be 0
                    seedDict[str(evaleft[i])] = str(evaright[ind_left[i]])
                    seedDictrev[str(evaright[ind_left[i]])] = str(evaleft[i])
                    if evaleft[i]+10500 == evaright[ind_left[i]]:
                        truecounter += 1
        print('Seed detected： ' +str(counter) +'\tcorrect： ' +str(truecounter))

        # write dickrank and seed
        ouf = open(pathab + 'condi_ranks'+iteroundnext+'.txt', 'w')
        for item in pairs:
            ouf.write(item[0] + '\t' + str(item[1]) + '\n')
        for item in dicrank:
            ouf.write(str(item) + '\t' + str(dicrank[item]) + '\n')
        print('Seed Dic after test: ' + str(len(seedDict)))

        # write new seed
        ouf_seed = open(pathab + 'sup_ent_ids' + iteroundnext, 'w')
        for item in seedDict.keys():
            ouf_seed.write(item + '\t' + seedDict[item] + '\n')

        # KG completion
        completion(pathab, seedDict, seedDictrev, iteroundnext)







