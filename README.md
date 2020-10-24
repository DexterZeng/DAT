# DAT
Source code for the SIGIR paper, [Degree-Aware Alignment for Entities in Tail](https://arxiv.org/abs/2005.12132)

## Data
Place the <code>wiki.multi.de.vec</code>, <code>wiki.multi.fr.vec</code>, <code>wiki.multi.en.vec</code> (obtained from [MUSE](https://github.com/facebookresearch/MUSE)) under "./data" 

## Run

### Step 1: Generate concatenated power mean embedding
1. <code>cpm.py</code>. The inputs are multilingual/monolingual word embeddings; the outputs are the word embeddings merely containing the words in the names of ent1 and ent2 (<code>name2embed1.pkl</code> and <code>name2embed2.pkl</code>).
2. <code>cpm2.py</code>. The outputs are the embeddings of names (1-average, 3-cpm, 6-cpm(multilingual)).

### Step 2: Choose a structural model
We use [RSNs](https://github.com/nju-websoft/RSN) in our paper. As pointed out in the paper, other models are also viable, e.g., [GCN](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).

This code is based on [GCN](https://github.com/1049451037/GCN-Align) due to its simplicity. It can be easily replaced with [RSNs](https://github.com/nju-websoft/RSN).

Run <code>bash run.sh</code> to get the results.

<!--
Run <code>train.py</code>, note to alter the parameter "iteround" in <code>train.py</code> and <code>pre.py</code> if not the first round.
    ### Step 3: DAT.py
    1. Determine the weights of features using Co-attention network.
    2. Obtain the alignment results.
    3. Select confident results to complete KGs, and generate new seed entities pairs.
    4. Note the parameters: iteround, iteround1, iteroundnext.
    5. name_shape should be tuned according to name_embed
    Iterate Step 2 and 3 for a certain number of rounds, or until reaching a certain criterion. (note to change "iteround")
-->
    



## CITATION
If you find our work useful, please cite it as follows:
```
@inproceedings{DAT,
	Author = {Weixin Zeng and Xiang Zhao and Wei Wang and Jiuyang Tang and Zhen Tan},
	Booktitle = {SIGIR 2020},
    Pages = {811--820},
	Title = {Degree-Aware Alignment for Entities in Tail},
	Year = {2020}
}
```
