'''
This file contains two data classes, which uses MinMaxScaler or divide by 10**MaxNumberLength to normalize data
This file also contains an RNN model called SeqRNN to solve seqence problem in IQ-test
'''
import torch
from torch import nn
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
options_dic={0:'A',1:'B',2:'C',3:'D'}
# This function takes a sequence in, convert its element to floats,
# and split the sequence to multiple overlapping sub-sequence of length three.
def split_input(stem):
    onelist=[]
    tmplist=stem.split(",")
    if len(tmplist)==1:
        tmplist=stem.split(" ")
    # tmplist = stem.split()
    for i in range(len(tmplist)):
        if "/" in tmplist[i]:
            field=tmplist[i].split('/')
            try:
                tmplist[i]=float(field[0])/float(field[1])
            except:
                continue
        elif "√" in tmplist[i]:
            field=tmplist[i].strip(' ').split('√')
            if field[0]=='':
                tmplist[i]=math.sqrt(float(field[1]))
            else:
                tmplist[i]=float(field[0])*math.sqrt(float(field[1]))
        elif "^" in tmplist[i]:
            field=tmplist[i].strip(' ').split('^')
            tmplist[i]=float(field[0])**float(field[1])
    for i in range(len(tmplist)-2):
        try:

            tmplist=list(map(float,tmplist))
            onelist.append(tmplist[i:i+3])
        except ValueError as e:
            print(e, tmplist, i, stem)
            # NewTmpList = []
            # for EachElement in tmplist:
            #     NewTmpList.append(float(EachElement.strip()))
            # pass
    return onelist

# This function takes real answer to sequence question and the predicted value
# and judge if the prediction is correct
def accuracy(cnt,right,op,recover_output,recover_value,gen_ans=False):
    ans_idx=None
    if len(op)==0:
        if abs(int(recover_output)-recover_value)<0.01:
            right+=1
        cnt+=1
    else:
        diff=[]
        for j in range(len(op)):
            if isinstance(op[j], str) and "/" in op[j]:
                field=op[j].split('/')
                try:
                    op[j]=float(field[0])/float(field[1])
                except ValueError as e:
                    print(op[j])
                    op[j] = 10000

            elif isinstance(op[j], str) and "," in op[j]:
                field=op[j].split(',')
                try:
                    op[j]=int(field[-1])
                except ValueError as e:
                    print(op[j])
                    op[j] = 10000
            try:
                diff.append(abs(recover_output-float(op[j])))
            except:
                try:
                    op[j]=float(op[j].split(' ')[-1])
                except ValueError as e:
                    print(op[j])
                    op[j] = 10000
                diff.append(abs(recover_output-op[j]))
        # the most similar option to output
        ans_idx=diff.index(min(diff))
        ans=op[ans_idx]
        cnt+=1
        if abs(float(ans)-recover_value)<0.01:
            right+=1
    if gen_ans==False:
        return cnt,right
    else:
        return cnt,right,ans_idx

# data class for train and valid set
# To normalize input, find max length n of each observation, 
# and divide all number in this observation by 10^n
class SeqData_normbylen(Dataset):
    def __init__(self, data_dir, filename):
        files=os.path.join(data_dir, filename)
        inputdic=open(files, 'r')
        self.seqdict = json.load(inputdic)
        self.seqlist=[]
        self.keylist=[]
        for key in self.seqdict:
            stem=self.seqdict[key]['stem']
            # skip all next operations in for if sequence is illegal
            if stem[0][0].isalpha()==True or stem[0][0].isdigit()==False:
                continue 
            onelist=split_input(stem)
            if len(onelist)>1 and len(onelist[0])!=0:
                self.seqlist.append(onelist)
                self.keylist.append(key)
        self.length=len(self.seqlist)

    def __getitem__(self,index,acc=False):
        tmp=np.copy(self.seqlist[index])
        real_value=tmp[-1][-1]
        res=[]
        for i in range(len(tmp)):
            if len(tmp[i])!=1:
                tmp[i]=np.array([tmp[i]])
                lengthes=[len(str(int(arr))) for arr in tmp[i]]
                maxlen=max(lengthes)
                if maxlen!=0:
                    for j in range(len(tmp[i])):
                        tmp[i][j]=tmp[i][j]/(10**maxlen)
                res.append([tmp[i]])
            else:
                lengthes=[len(arr) for arr in tmp[i][0]]
                maxlen=max(lengthes)
                if maxlen!=0:
                    for j in len(tmp[i][0]):
                        tmp[i][0][j]=tmp[i][0][j]/(10**maxlen)
                res.append(tmp[i])
        res=torch.tensor(res,dtype=torch.double)
        if acc:
            return res,maxlen,real_value
        else:
            return res

    #norm by len
    def cal_accuracy(self,seqrnn,seqdict,gen_answer=False):
        setlen=self.length
        right=0
        cnt=0
        if gen_answer==True:
            ans_dict={}
        # for each sequence, find the most similar option to the output of network
        # if the chosen option is the answer, the output is accurate
        for i in range(setlen):
            data,maxlen,real_value=self.__getitem__(i,acc=True)
            curID=self.keylist[i]
            op=seqdict[curID]["options"]
            
            hidden = seqrnn.initHidden()
            data=data[:-1]
            for i in range(data.size()[0]):
                output, hidden = seqrnn(data[i].float(), hidden)
            output=output[0][0].item()
            recover_output=output*(10**maxlen)
            if gen_answer==False:
                cnt,right=accuracy(cnt,right,op,recover_output,real_value)
            else:
                cnt,right,ans_idx=accuracy(cnt,right,op,recover_output,real_value,gen_answer)
                if ans_idx==None:
                    recover_output=round(recover_output,2)
                else:
                    recover_output=options_dic[ans_idx]
                ans_dict[curID]={'answer': [recover_output]}
        if gen_answer==False:
            return right/cnt
        else:
            return ans_dict,right/cnt
    def gen_answer(self, seqrnn, seqdict, gen_answer=True):
        setlen = self.length
        # right = 0
        # cnt = 0
        if gen_answer == True:
            ans_dict = {}
        # for each sequence, find the most similar option to the output of network
        # if the chosen option is the answer, the output is accurate
        for i in range(setlen):
            data, maxlen, real_value = self.__getitem__(i, acc=True)
            curID = self.keylist[i]
            op = seqdict[curID]["options"]

            hidden = seqrnn.initHidden()
            data = data[:-1]
            for i in range(data.size()[0]):
                output, hidden = seqrnn(data[i].float(), hidden)
            output = output[0][0].item()
            recover_output = output * (10 ** maxlen)
            if gen_answer == False:
                cnt, right = accuracy(cnt, right, op, recover_output, real_value)
            else:
                cnt, right, ans_idx = accuracy(cnt, right, op, recover_output, real_value, gen_answer)
                if ans_idx == None:
                    recover_output = round(recover_output, 2)
                else:
                    recover_output = options_dic[ans_idx]
                ans_dict[curID] = {'answer': [recover_output]}
        if gen_answer == False:
            return right / cnt
        else:
            return ans_dict, right / cnt
# RNN module
class SeqRNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(SeqRNN, self).__init__()
        self.hidden_size = hidden_size
    
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

