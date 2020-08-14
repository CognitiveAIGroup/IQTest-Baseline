'''
This file trains an RNN model to solve sequence problems in IQ-test,
and display losses on train set and valid set, and accuracy on valid set to evaluate performance
The visulized result and trained model are saved.
Sequences are splited into sub-sequences with length 3 to train the model.
'''
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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data_file={'train':'seq-train.json','valid':'seq-valid.json','question':'seq-public.json'}

INPUT_SIZE = 3
OUTPUT_SIZE = 1
NUM_LAYERS = 1

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(seqrnn,args,data,real_val):
    hidden = seqrnn.initHidden()
    seqrnn.zero_grad()
    for i in range(data.size()[0]):
        output, hidden = seqrnn(data[i].float(), hidden)
    # MSE loss
    loss = (output-real_val)**2
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in seqrnn.parameters():
        p.data.add_(p.grad.data, alpha=-args.lr)
    return output, loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        help="The input data dir. Should contain the input json file")
    parser.add_argument("--output_dir",
                        default='res',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--lr",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--hidden_size",
                        default=8,
                        type=int,
                        help="Hidden size for rnn")
    parser.add_argument("--niters",
                        default=int(7e4),
                        type=int,
                        help="Hidden size for rnn")
    parser.add_argument("--norm",
                        default=1, 
                        type=float,
                        help="0 for minmax, 1 for length-based normalize")
    parser.add_argument("--save",
                        default=0, 
                        type=float,
                        help="1 for save image result and model")

    args = parser.parse_args()
    question_file=os.path.join(args.data_dir, data_file['question'])
    option_file=open(question_file, 'r')
    option_file = json.load(option_file)
    seqdict={}
    for i in range(len(option_file)):
        curid=str(option_file[i]["id"])
        seqdict[curid]={"stem": option_file[i]["stem"], "options": option_file[i]["options"], "category": option_file[i]["category"]}

    if args.norm==0:
        seq_train_data = SeqData_minmax(args.data_dir, data_file['train'])
        seq_valid_data = SeqData_minmax(args.data_dir, data_file['valid'])
    else:
        seq_train_data = SeqData_normbylen(args.data_dir, data_file['train'])
        seq_valid_data = SeqData_normbylen(args.data_dir, data_file['valid'])
    seqrnn = SeqRNN(INPUT_SIZE, args.hidden_size, OUTPUT_SIZE)

    n_iters = args.niters
    print_every = n_iters/10
    plot_every = n_iters/40
    current_loss = 0
    curvad_loss=0
    train_losses = []
    vad_acc = []
    vad_loss=[]
    start = time.time()

    print('trainning...')
    for t in range(n_iters + 1):
        data=seq_train_data.__getitem__(np.random.randint(0,seq_train_data.length))
        '''
        data contains multiple sub-sequence as patterns used to predict the answer,
        a data sample(normalized):
        tensor([[[0.1600, 0.2700, 0.1600]],
        [[0.2700, 0.1600, 0.0500]],
        [[0.1600, 0.0500, 0.0100]]], dtype=torch.float64)
        '''
        real_value=data[-1][-1][2]
        output, loss = train(seqrnn,args,data,real_value)
        current_loss += loss
        # print iter number, loss, time cost
        if t % print_every == 0:
            correct = '✓' if output == real_value else '✗'
            print('iter:%d completion:%d%% time cost:(%s) loss:%.4f correct:%s' % (t, t / n_iters * 100, timeSince(start), loss, correct))
        # record losses on train set and valid set, and accuracy for plotting
        if t % plot_every == 0 and t!=0:
            curlos=current_loss / plot_every
            data=seq_valid_data.__getitem__(np.random.randint(0,seq_valid_data.length))
            real_value=data[-1][-1][2]
            output, curvad_loss = train(seqrnn,args,data,real_value)
            curacc=seq_valid_data.cal_accuracy(seqrnn,seqdict)
            train_losses.append(curlos)
            vad_loss.append(curvad_loss)
            vad_acc.append(curacc)
            current_loss = 0
            curvad_loss=0

    fig, (ax1, ax2)= plt.subplots(1,2, figsize = (10,5))
    ax1.plot(train_losses,label='train loss')
    ax1.plot(vad_loss,label='valid loss')
    ax1.legend()
    ax2.plot(vad_acc)
    xaxis_len=len(vad_acc)//2
    xList=[i for i in range(xaxis_len)]
    yList=[vad_acc[i*2] for i in range(xaxis_len)]
    for x, y in zip(xList, yList):
	    ax2.text(x*2, y, '%.1f'%(y*100), ha='center', va='bottom', fontsize=5)
    
    plt.title(str(n_iters/10000)+'w iters '+'lr'+str(args.lr)+'_hidden'+str(args.hidden_size)+'_norm'+str(int(args.norm)),pad=20)
    img_path=os.path.join(args.output_dir,'train_eval'+str(n_iters/10000)+'w_'+'lr'+str(args.lr)+'_hidden'+str(args.hidden_size)+'_norm'+str(int(args.norm))+'.png')
    if args.save==1:
        plt.savefig(img_path)
        print('img saved at:',img_path)
    plt.tight_layout()
    plt.show()

    model_path=os.path.join(args.output_dir,'RNNmodel'+str(n_iters/10000)+'w_'+'lr'+str(args.lr)+'_hidden'+str(args.hidden_size)+'_norm'+str(args.norm)+'.pkl')
    if args.save==1:
        torch.save(seqrnn.state_dict(), model_path)
        print('model saved at:',model_path)



