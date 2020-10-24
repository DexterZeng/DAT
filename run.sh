#!/usr/bin/env bash
#iter=5
#python main.py --ite $iter
#python DAT-gcn.py --ite $iter

dataset='en_fr_15k_V1'
# a refers to the number of rounds
for a in {1..5}
do
        python main.py --lan $dataset --ite $a
        python DAT.py --lan $dataset --ite $a
done