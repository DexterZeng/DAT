import tensorflow as tf
import numpy as np
import json
import csv
import time
import copy

def cal_performance(ranks, top=10):
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_10 = sum(ranks <= top) * 1.0 / len(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    return m_r, h_10, mrr

def loadNe(path):
    if '15k' in path:
        f1 = open(path)
        vectors = []
        for i, line in enumerate(f1):
            id, word, vect = line.rstrip().split('\t', 2)
            vect = np.fromstring(vect, sep=' ')
            vectors.append(vect)
        embeddings = np.vstack(vectors)

    else:
        with open(file=path, mode='r', encoding='utf-8') as f:
            embedding_list = json.load(f)
            print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
        embeddings = np.array(embedding_list)

    return embeddings

def get_freoneHot(path):
    inf2 = open(path + 'triples_11')
    id2fre = dict()
    for line in inf2:
        strs = line.strip().split('\t')
        if strs[0] not in id2fre:
            fre = 0
        else:
            fre = id2fre[strs[0]]
        fre += 1
        id2fre[strs[0]] = fre

        if strs[2] not in id2fre:
            fre1 = 0
        else:
            fre1 = id2fre[strs[2]]
        fre1 += 1
        id2fre[strs[2]] = fre1

    inf2 = open(path + 'triples_21')
    for line in inf2:
        strs = line.strip().split('\t')
        if strs[0] not in id2fre:
            fre = 0
        else:
            fre = id2fre[strs[0]]
        fre += 1
        id2fre[strs[0]] = fre

        if strs[2] not in id2fre:
            fre1 = 0
        else:
            fre1 = id2fre[strs[2]]
        fre1 += 1
        id2fre[strs[2]] = fre1
    print('total ent fre dic length: ' + str(len(id2fre)))

    oneHotLen = len(set(id2fre.values()))
    print('one hot length \t' + str(oneHotLen))
    oneHot = tf.one_hot(list(id2fre.values()), oneHotLen)
    return oneHot, id2fre

def getent(path):
    inf = open(path)
    ents = []
    for line in inf:
        strs = line.strip().split('\t')
        ents.append(int(strs[0]))
    return ents

def ent_ori2new(arr, dic):
    # int 2 int
    newarr = []
    for item in arr:
        newarr.append(int(dic[str(item)]))
    return newarr

def obtainseed(path, a, name_embed,pathseed):
    seed = open(pathseed)
    seedDict = dict()
    seedDictrev = dict()
    seed1 = []
    for line in seed:
        strs = line.strip().split('\t')
        seedDict[strs[0]] = strs[1]
        seedDictrev[strs[1]] = strs[0]
        seed1.append(strs[0])

    seed = open(path + 'ref_ent_ids')
    seedleft = []
    seedright = []
    seedleft_retr = []
    seedright_retr = []

    recordEva = dict()
    recordEva_right = dict()

    for coun, line in enumerate(seed):
        if coun == 1050: break
        strs = line.strip().split('\t')

        seedleft.append(int(strs[0]))
        seedright.append(int(strs[1]))
        recordEva[int(strs[0])] = int(strs[1])
        recordEva_right[int(strs[1])] = int(strs[0])

        seedleft_retr.append(int(strs[0]))
        seedright_retr.append(int(strs[1]))

    seedleft_vec = a[seedleft_retr]  # train stuctural embedding set...
    seedright_vec = a[seedright_retr]
    print(seedleft_vec.shape)

    seedleft_namevec = name_embed[seedleft]
    seedright_namevec = name_embed[seedright]
    print(seedleft_namevec.shape)

    return seedleft_vec, seedright_vec, seedleft_namevec, seedright_namevec, seedleft, seedright, seedDict,seedDictrev, seedleft_retr, seedright_retr

def constructloss(oneHot, name_shape, gamma, batchsize, learning_rate):
    data_size = tf.placeholder(tf.int32, shape=[], name='datasize')
    # number sequence
    ind = tf.concat([tf.expand_dims(tf.range(data_size), axis=1), tf.expand_dims(tf.range(data_size), axis=1)], axis=1)

    holder_left = tf.placeholder(tf.int32, shape=[None], name='seedleft') #ids
    Q1 = tf.gather(oneHot, holder_left) # degree feature
    K1 = tf.placeholder(tf.float32, [None, 2, name_shape], 'K1')  # the other two features

    holder_right = tf.placeholder(tf.int32, [None], 'seedright')
    Q2 = tf.gather(oneHot, holder_right)
    K2 = tf.placeholder(tf.float32, [None, 2, name_shape], 'K2')

    attention_W = tf.get_variable(name="attention_W", shape=[Q1.shape[-1], name_shape],
                                  initializer=tf.contrib.layers.xavier_initializer())

    Q1_cont = tf.expand_dims(tf.tensordot(Q1, attention_W, axes=1), -2) # convert one-hot to continuous vector!
    Q2_cont = tf.expand_dims(tf.tensordot(Q2, attention_W, axes=1), -2)

    K1 = tf.concat([K1, Q1_cont], axis=-2) # the final feature matrix nx3x900
    K2 = tf.concat([K2, Q2_cont], axis=-2)

    K1_expand = tf.expand_dims(K1, -3) #nx1x3x900
    K1_expand = tf.tile(K1_expand, (1, 3, 1, 1))#nx3x3x900

    K2_expand = tf.expand_dims(K2, -2) #nx3x1x900
    K2_expand = tf.tile(K2_expand, (1, 1, 3, 1)) #nx3x3x900

    sim_mat = K1_expand * K2_expand #nx3x3x900

    con = tf.concat((K1_expand, K2_expand, sim_mat), -1) #nx3x3x2700

    sim_mat = tf.reduce_mean(
        tf.layers.dense(con, units=1, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer()), -1) #nx3x3

    sim_left = tf.nn.softmax(sim_mat[:, :2, :], -2) # softmax layer
    att_left = tf.reduce_mean(sim_left, axis=-1) # average layer

    sim_right = tf.nn.softmax(sim_mat[:, :, :2], -1)
    att_right = tf.reduce_mean(sim_right, axis=-2)

    K1_stru = tf.squeeze(K1[:, 0, :])
    K1_text = tf.squeeze(K1[:, 1, :])  # batch*300 #get the weight
    K2_stru = tf.squeeze(K2[:, 0, :])
    K2_text = tf.squeeze(K2[:, 1, :])

    he_stru = tf.nn.l2_normalize(K1_stru, dim=-1)
    norm_e_em_stru = tf.nn.l2_normalize(K2_stru, dim=-1)
    aep_stru = tf.matmul(he_stru, tf.transpose(norm_e_em_stru)) # similarity matrix for stru

    he_text = tf.nn.l2_normalize(K1_text, dim=-1)
    norm_e_em_text = tf.nn.l2_normalize(K2_text, dim=-1)
    aep_text = tf.matmul(he_text, tf.transpose(norm_e_em_text)) # similarity matrix for text

    aep_left = tf.stack([aep_stru, aep_text], axis=1)  # batch*2*batch
    aep_left = tf.multiply(aep_left, tf.expand_dims(att_left, -1))  # batch*2*batch
    aep_left = tf.reduce_sum(aep_left, axis=1)  # batch* batch

    aep_right = tf.stack([tf.transpose(aep_stru), tf.transpose(aep_text)], axis=1)
    aep_right = tf.multiply(aep_right, tf.expand_dims(att_right, -1))
    aep_right = tf.reduce_sum(aep_right, axis=1)

    probs_left = tf.gather_nd(aep_left, ind)
    probs_right = tf.gather_nd(aep_right, ind)

    newloss = tf.reduce_sum(tf.nn.relu(tf.add(-probs_left, gamma)))
    newloss = (newloss + tf.reduce_sum(tf.nn.relu(tf.add(-probs_right, gamma)))) / (2 * batchsize)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(newloss)  # optimizer can be changed

    return newloss, train_step, aep_left, aep_right, att_left

def enrichseed(probs_eva, aep_fuse, seedleft, seedright, seedDict, seedDictrev):
    ind_left = np.argmax(probs_eva, axis=1)
    maxes = np.max(probs_eva, axis=1)
    probs_eva[range(len(probs_eva)), np.argmax(probs_eva, axis=1)] = np.min(probs_eva)
    maxes1 = np.max(probs_eva, axis=1)
    gap_left = maxes - maxes1

    probs = aep_fuse - aep_fuse[range(len(seedleft)), range(len(seedleft))].reshape(len(aep_fuse), 1)
    ranks = (probs > 0).sum(axis=1) + 1
    MR, H10, MRR = cal_performance(ranks, top=10)
    _, H1, _ = cal_performance(ranks, top=1)
    # msg = 'Train_right: Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
    # print(msg + '\n')

    ind_right = np.argmax(probs, axis=1)
    maxes = np.max(probs, axis=1)
    probs[range(len(probs)), np.argmax(probs, axis=1)] = np.min(probs)
    maxes1 = np.max(probs, axis=1)
    gap_right = maxes - maxes1

    counter = 0
    truecounter = 0
    for i in range(len(ind_left)):
        if ind_right[ind_left[i]] == i:
            if gap_left[i] >= 0.05 and gap_right[i] >= 0.05:
                counter += 1
                seedDict[str(seedleft[i])] = str(seedright[ind_left[i]])
                seedDictrev[str(seedright[ind_left[i]])] = str(seedleft[i])
                if ind_right[i] == i and ind_left[i] == i:
                    truecounter += 1

    # print('Seed detected(train)： ' + str(counter) + '\tcorrect： ' + str(truecounter))
    return seedDict, seedDictrev

def enrichtestv1(aep_fuse_eva,  aep_fuse_evar,evaleft,evaright, pre):
    probs = aep_fuse_eva - aep_fuse_eva[range(len(evaleft)), range(len(evaleft))].reshape(len(aep_fuse_eva), 1)
    ranks_left = []
    truths = []
    mrr_sum_l = 0

    top_k = (1, 10)
    top_lr = [0] * len(top_k)
    for i in range(len(evaleft)):
        trueid = evaleft[i]
        truerightid = evaleft[i] + 10500
        if truerightid in evaright:
            p = evaright.index(truerightid)
            rank = (-aep_fuse_eva[i]).argsort()
            rank_index = np.where(rank == p)[0][0]
        else:
            rank_index = 100000
        ranks_left.append(rank_index)
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
        if rank_index == 0:
            truths.append(trueid)

    # print('\nFor each left:\t' + str(len(evaleft)))
    # for i in range(len(top_lr)):
    #     print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(evaleft) * 100))
    # print("MRR: " + str(mrr_sum_l / len(evaleft)))
    # print()

    for preitem in pre:
        mrr_sum_l = mrr_sum_l + 1.0 / (preitem + 1)
        for j in range(len(top_k)):
            if preitem < top_k[j]:
                top_lr[j] += 1
    print('For each left full:\t'+ str(len(evaleft)+len(pre)))
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / (len(evaleft)+len(pre)) * 100))
    print("MRR: " + str(mrr_sum_l / (len(evaleft)+len(pre))))
    print()

    ind_left = np.argmax(probs, axis=1)
    maxes = np.max(probs, axis=1)
    probs[range(len(probs)), np.argmax(probs, axis=1)] = np.min(probs)
    maxes1 = np.max(probs, axis=1)
    gap_left = maxes - maxes1

    probs = aep_fuse_evar - aep_fuse_evar[range(len(evaleft)), range(len(evaleft))].reshape(len(aep_fuse_evar), 1)

    ind_right = np.argmax(probs, axis=1)
    maxes = np.max(probs, axis=1)
    probs[range(len(probs)), np.argmax(probs, axis=1)] = np.min(probs)
    maxes1 = np.max(probs, axis=1)
    gap_right = maxes - maxes1

    return ind_left, ind_right, gap_left, gap_right, np.array(ranks_left), truths

def evatest(aep_fuse_eva,evaleft,evaright):
    top_k = (1, 10)
    mrr_sum_l = 0
    top_lr = [0] * len(top_k)
    for i in range(len(evaleft)):
        trueid = evaleft[i]
        truerightid = evaleft[i] + 10500
        if truerightid in evaright:
            p = evaright.index(truerightid)
            rank = (-aep_fuse_eva[i]).argsort()
            rank_index = np.where(rank == p)[0][0]
        else:
            print(truerightid)
            rank_index = 100000
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        # could be nothing
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(evaleft) * 100))
    print("MRR: " + str(mrr_sum_l / len(evaleft)))

def completion(path,seedDict,seedDictrev,iteroundnext):
    inf1 = open(path + 'triples_21')
    triples2 = []
    triplesfor1 = []
    for counter, line in enumerate(inf1):
        strs = line.strip().split('\t')
        triples2.append([strs[0], strs[1], strs[2]])
        if strs[0] in seedDictrev.keys() and strs[2] in seedDictrev.keys():
            triplesfor1.append([seedDictrev[strs[0]], strs[1], seedDictrev[strs[2]]])

    print()
    print('#Triples in KG2: ' + str(len(triples2)))
    print('#Triples for KG1: ' + str(len(triplesfor1)))

    inf2 = open(path + 'triples_11')
    triples1 = []
    triplesfor2 = []
    for line in inf2:
        strs = line.strip().split('\t')
        triples1.append([strs[0], strs[1], strs[2]])
        if strs[0] in seedDict.keys() and strs[2] in seedDict.keys():
            triplesfor2.append([seedDict[strs[0]], strs[1], seedDict[strs[2]]])

    print('#Triples in KG1: ' + str(len(triples1)))
    print('#Triples for KG2: ' + str(len(triplesfor2)))

    # #ab1
    for item in triplesfor1:
        # if item not in triples1:
            triples1.append(item)

    print('#Triples in KG1: (added)' + str(len(triples1)))

    for item in triplesfor2:
        # if item not in triples2:
            triples2.append(item)

    print('#Triples in KG2: (added)' + str(len(triples2)))
    # ab1

    ouf = open(path + 'triples_1' + iteroundnext, 'w')
    for item in triples1:
        ouf.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\n')

    ouf = open(path + 'triples_2' + iteroundnext, 'w')
    for item in triples2:
        ouf.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\n')