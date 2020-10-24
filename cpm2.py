import numpy as np
import re
import pickle

def gen_mean(vals, p):
    p = float(p)
    return np.power(
        np.mean(
            np.power(
                np.array(vals, dtype=complex),
                p),
            axis=0),
        1 / p
    )
operations = dict([
    ('mean', (lambda word_embeddings: [np.mean(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('max', (lambda word_embeddings: [np.max(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('min', (lambda word_embeddings: [np.min(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('p_mean_2', (lambda word_embeddings: [gen_mean(word_embeddings, p=2.0).real], lambda embeddings_size: embeddings_size)),
    ('p_mean_3', (lambda word_embeddings: [gen_mean(word_embeddings, p=3.0).real], lambda embeddings_size: embeddings_size)),
])
def get_sentence_embedding(sentence, embeddings, chosen_operations, con):
    word_embeddings = []
    for tok in sentence:
        if tok in embeddings:
            vec = embeddings[tok]
            word_embeddings.append(vec)

    if not word_embeddings:
        print('No word embeddings for sentence:\n{}'.format(sentence))
        con += 1
        size = 0
        for o in chosen_operations:
            size += operations[o][1](300)
        sentence_embedding = np.zeros(size)
    else:
        concat_embs = []
        for o in chosen_operations:
            concat_embs += operations[o][0](word_embeddings)
        sentence_embedding = np.concatenate(
            concat_embs,
            axis=0
        )
    return sentence_embedding, con

id2embed = dict()
id2name = dict()

path = 'data/en_fr_15k_V1/'

data_input = open(path + 'name2embed1.pkl','rb')
name2embed1 = pickle.load(data_input)
data_input = open(path + 'name2embed2.pkl','rb')
name2embed2 = pickle.load(data_input)

inf = open(path + 'ent_ids_1')
con = 0
for i1, line in enumerate(inf):
    strs = line.strip().split('\t')
    id2name[int(strs[0])] = strs[1]
    wordline = strs[1].split('/')[-1].lower().replace('_',' ')
    words = re.findall(r'\b\w+\b', wordline)
    embed, con = get_sentence_embedding(words, name2embed1, ['mean', 'min', 'max'], con)
    # id2embed[strs[0]] = embed
    embed_, con = get_sentence_embedding(words, name2embed2, ['mean', 'min', 'max'], con)
    id2embed[strs[0]] = np.concatenate((embed,embed_), axis = 0)


con1 = 0
inf = open(path + 'ent_ids_2')
for i2, line in enumerate(inf):
    strs = line.strip().split('\t')
    id2name[int(strs[0])] = strs[1]
    wordline = strs[1].split('/')[-1].lower().replace('_', ' ')
    words = re.findall(r'\b\w+\b', wordline)
    embed, con1 = get_sentence_embedding(words, name2embed2, ['mean', 'min', 'max'], con1)
    # id2embed[strs[0]] = embed
    embed_, con1 = get_sentence_embedding(words, name2embed1, ['mean', 'min', 'max'], con1)
    id2embed[strs[0]] = np.concatenate((embed,embed_), axis = 0)

print(con)
print(con1)
print(len(id2embed))
outf = open(path + 'name_vec_cpm_6.txt', 'w')
for id in range(len(id2embed)):
    ent = id2name[id]
    embed = id2embed[str(id)]
    dis_str = ''
    for i in embed:
        dis_str = dis_str+ str(i) + ' '
    dis_str = dis_str[:-1]
    outf.write(str(id) + '\t' + ent + '\t' + dis_str + '\n')


