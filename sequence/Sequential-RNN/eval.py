import os
import shutil
import argparse
import random
import numpy as np
import torch
import time
import json
import math
from modeling import SeqRNN
from modeling import SeqData_minmax,SeqData_normbylen
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
data_file={'train':'seq-train.json','test':'seq-test.json','all':'seq-filled.json'}


INPUT_SIZE = 3
OUTPUT_SIZE = 1
HIDDEN_SIZE = 8
seqrnn = SeqRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
seqrnn.load_state_dict(torch.load('model/RNNmodel10.0w_lr0.001_hidden8_norm1.pkl'))

data_dir='../data'
# print(data_dir, data_file['test'])

option_file=open(data_dir+"/seq-public.json", 'r')
option_file = json.load(option_file)
seqdict={}
for i in range(len(option_file)):
    curid=str(option_file[i]["id"])
    seqdict[curid]={"stem": option_file[i]["stem"], "options": option_file[i]["options"], "category": option_file[i]["category"]}

if __name__ == "__main__":
    seq_test_data = SeqData_normbylen(data_dir, data_file['all'])
    # oneObs(seq_train_data)
    curacc=seq_test_data.cal_accuracy(seqrnn,seqdict)
    ans_dict,acc=seq_test_data.cal_accuracy(seqrnn,seqdict,True)
    cnt=0
    for key in ans_dict:
        print(key,ans_dict[key])
        cnt+=1
        if cnt==5:
            break

    resfile=open("baseline_answer.json", "w", encoding='utf-8')
    json.dump(ans_dict, resfile, indent=2, sort_keys=True, ensure_ascii=False)
