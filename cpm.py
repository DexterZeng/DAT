import re
import numpy as np
import io
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_vec(emb_path, nmax=20000000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

# word embeddings for different languages
src_path = './data/wiki.multi.en.vec'
tgt_path = './data/wiki.multi.fr.vec'

src_embeddings, src_id2word, src_word2id = load_vec(src_path)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path)

print(len(src_embeddings))
print(len(tgt_embeddings))

path = 'data/en_fr_15k_V1/'
embed = np.zeros(300)

inf = open(path + 'ent_ids_1')
allwords1 = []
for i1, line in enumerate(inf):
    strs = line.strip().split('\t')
    wordline = strs[1].split('/')[-1].lower().replace('_',' ')
    words = re.findall(r"\w+|[^\w\s]", wordline)
    allwords1.extend(words)

print(len(allwords1))
allwords1 = list(set(allwords1))
print(len(allwords1))

name2embed1 = dict()
for word in allwords1:
    try:
        w_emb = src_embeddings[src_word2id[word.strip()]]
    except:
         try:
            w_emb = tgt_embeddings[tgt_word2id[word.strip()]]
         except:
            continue
    name2embed1[word] = w_emb

# reduce the number of word embeddings to the ones in the dataset
data_output = open(path + 'name2embed1.pkl','wb')
pickle.dump(name2embed1,data_output)


inf = open(path + 'ent_ids_2')
allwords2 = []
for i2, line in enumerate(inf):
    strs = line.strip().split('\t')
    wordline = strs[1].split('/')[-1].lower().replace('_', ' ')
    words = re.findall(r"\w+|[^\w\s]", wordline)
    allwords2.extend(words)

print(len(allwords2))
allwords2 = list(set(allwords2))
print(len(allwords2))

name2embed2 = dict()

for word in allwords2:
    try:
        w_emb = tgt_embeddings[tgt_word2id[word.strip()]]
    except:
         try:
            w_emb = src_embeddings[src_word2id[word.strip()]]
         except:
            continue
    name2embed2[word] = w_emb

# reduce the number of word embeddings to the ones in the dataset
data_output = open(path + 'name2embed2.pkl','wb')
pickle.dump(name2embed2,data_output)