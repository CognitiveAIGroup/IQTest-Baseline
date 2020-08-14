import json
import re
import torch
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.utils.data as Data
import math
import numpy as np

# inNode by inNode trainning

## check whether it's purely a num stems where "?"" locates at the end ##
def FindNumStem(stm):
    # if the stem is completely consist of numbers ---> is_num=1; otherwise is_num!=1
    # if the question mark is at the end ---> is_end=-1/-2; otherwise is_end=0
    is_end=-1
    is_num=1
    for i in range(len(stm)):
        if stm[i]=="?" :
            if i!=(len(stm)-1):
                is_end=0
                is_num=2                       # 2 meaningless
        if stm[i]==")" and stm[i-1]=="(":
            if i!=(len(stm)-1):
                is_end=0
                is_num=2 
            else:
                is_end=-2
        if (ord(stm[i])>=65) & (ord(stm[i])<=90):
            is_num=0
        elif (ord(stm[i])>=97) & (ord(stm[i])<=122):
            is_num=0
    return is_num,is_end

## convert string to number ##
#need update
def toNum(s):
    neg_flag=0
    frac_flag=-1
    sqrt_flag=-1
    dot_flag=-1
    num=0
    for i in range(len(s)):
        if s[i]=='-':               #assume - is always be index 0
            neg_flag=-1
        if s[i]=='/':
            frac_flag=i
        if s[i]=='√':
            sqrt_flag=i             #assume that / and √ cannot occur at the same s
        if s[i]=='.':
            dot_flag=i
    if frac_flag!=(-1):
        if neg_flag==0:
            num=int(s[:frac_flag])/int(s[frac_flag+1:])
        else:
            num=int(s[1:frac_flag])/int(s[frac_flag+1:])*(-1)
    #if frac_flag!=(-1):
    #    num=NegNum(neg_flag,frac_flag,s)        
    if sqrt_flag!=(-1):
        if neg_flag==0:
            if sqrt_flag>0:
                num=math.sqrt(int(s[sqrt_flag+1:]))*int(s[:sqrt_flag])
            elif sqrt_flag==0:
                num=math.sqrt(int(s[sqrt_flag+1:]))
            
        else:
            if sqrt_flag>0:
                num=math.sqrt(int(s[sqrt_flag+1:]))*int(s[1:sqrt_flag])*(-1)
            elif sqrt_flag==0:
                num=math.sqrt(int(s[sqrt_flag+1:]))*(-1)
    if dot_flag!=(-1):
        if neg_flag==0:
            num=float(s)
        else:
            num=float(s[1:])*(-1)
            
    if (frac_flag==-1) and (sqrt_flag==-1) and (dot_flag==-1):
        if neg_flag==0:
            num=int(s)
        else:
            num=int(s[1:])*(-1)
    return num

## construct the list of dicts which contain num stems and corresponding answers ##      
def NumList(path1,path2):
    #to classify the list into Num or Letter
    with open(path1,'r')as seq_json:
        seq_data=json.load(seq_json)           #  total len 861
    with open(path2,'r')as answer_json:
        answer=json.load(answer_json)

    Nlist=list()            #list of dicts which contains the pure number dict
    for i in range(len(seq_data)):
        id=seq_data[i]["id"]
        IsNum,IndexofQue=FindNumStem(seq_data[i]["stem"])
        if IndexofQue:                #only consider the question mark locates at the end
            if IsNum !=0 :
                seq_data[i]["stem"]=seq_data[i]["stem"][:IndexofQue]
                seq_data[i]["answer"]=answer[str(id)]["answer"]
                Nlist.append(seq_data[i])
    return Nlist

## normalize the items in the arr ##
def Norm(arr):
    mini=np.min(arr)
    maxi=np.max(arr)
    n1,n2=0,0
    while mini!=0:
        mini=int(mini/10)
        n1=n1+1
    while maxi!=0:
        maxi=int(maxi/10)
        n2=n2+1
    n=max(n1,n2)
    return arr/pow(10,n),n


Numlist=list()
Numlist=NumList('/home/linhan/Desktop/baseline of sequence/seq-public.json','/home/linhan/Desktop/baseline of sequence/seq-public.answer.json')

for s in range(len(Numlist)):
    stem=Numlist[s]["stem"]
    splitStem=re.split('\s|,',stem)
    opt=Numlist[s]["options"]
    print(opt)
    stemList=list()
    optList=list()
    ## convert option str list into numlist and put into each stem dict ##
    try:
        for k in opt:
            if k!='':
                optList.append(toNum(k))
    except ValueError:
        next
    else:
        Numlist[s]["options"]=optList

    ## "stem":["str1","str2","str3"] ---> "stem":[num1,num2,num3] ##
    try:
        for h in splitStem:
            if h!='':
                stemList.append(toNum(h))
            
    except ValueError:
        next
    else:
        Numlist[s]["stem"]=stemList 
    
for q in Numlist:
    print(q,"\n")  



#ANN
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
    
    def forward(self,x):
        x=torch.tanh(self.hidden(x))  #active function 
        x=self.predict(x)             #output layer predict
        return x



#real data
inNode=3               # # of input nodes
hidNode=3
done=0
right=0
for p in range(len(Numlist)):
    print("done:",done,"   id:",Numlist[p]["id"])
    stemLen=len(Numlist[p]["stem"])

    ## select the # of input nodes ##
    if stemLen>=7:
        inNode=4
    else:
        inNode=3

    ## in which case we cannot predict since no former rule ##
    if stemLen==inNode:                                      
        inNode=2  
    elif stemLen<inNode:
        continue
    net=Net(inNode,hidNode,1)
    optimizer=torch.optim.SGD(net.parameters(),lr=0.125)
    loss_func=torch.nn.MSELoss(reduction='sum')
    
    ## normalization --> get the nparray with normalized value and the norm factor ##
    try:
        normArray,norm=Norm(np.asarray(Numlist[p]["stem"]))   
    except TypeError:
           continue
    
    ## training begin! ##
    # each stem is trained 10000 times
    for t in range(10000):                      
        for i in range(stemLen-inNode):           
            try:
                x=torch.from_numpy(normArray[i:i+inNode])
                y=torch.tensor([normArray[i+inNode]])
            except IndexError:
                pass
            else:
                x=torch.tensor(x,dtype=torch.float32)
                y=torch.tensor(y,dtype=torch.float32)

                optimizer.zero_grad()         #set 0 to grad every time
                prediction=net(x)
                loss=loss_func(prediction,y)

                loss.backward()               #compute the gradient for each node
                optimizer.step()              #optimize using gradients
                
    ## predict the last unknown number ##
    testX=torch.from_numpy(normArray[stemLen-inNode:])
    testX=torch.tensor(testX,dtype=torch.float32)
    testPredict=net(testX).data.numpy()*pow(10,norm)
    done=done+1   # record the number of question done    


############################ calculate the accuracy ###############################
    
    ## nonempty options ##
    if Numlist[p]["options"]:
        try:
            answer=Numlist[p]["options"][Numlist[p]["answer"][0]-1]  #the value of answer
        except (ValueError,TypeError):
            continue
        dist=max(Numlist[p]["options"][0]-testPredict[0],testPredict[0]-Numlist[p]["options"][0])
        chosenAnswer=Numlist[p]["options"][0]
        for t in Numlist[p]["options"]:
            each_dist=max(t-testPredict[0],testPredict[0]-t)
            print("each_dist:",each_dist)
            if each_dist<dist:
                dist=each_dist
                chosenAnswer=t
        if chosenAnswer==answer: 
            right=right+1
            print("right:",right)
            print("predict:",testPredict[0],"chosenAnswer:",chosenAnswer,"dist:",dist,"answer",answer)
    ## empty option ##
    else:
        try:
            answer=toNum(Numlist[p]["answer"])
        except (ValueError,TypeError):
            continue
        if testPredict[0] <= (answer+1) and testPredict[0] >= (answer-1):
            right=right+1
            print("right:",right)
            print("predict:",testPredict[0],"answer",answer)

if done!=0:
    accuracy=right/done
    print(accuracy)
    print("total accuracy:",accuracy*done/1000)            
############################ calculate the accuracy ###############################    
print("end")



############################ accuracy rate ###############################
# learning rate= 0.125
# accuracy rate= # of correct answers/# of stems have been done
###  1) # of hidden nodes= 2;                     accuracy rate=31.54%
###  2) # of hidden nodes= # of input nodes;      accuracy rate=32.59% 
###  3) # of hidden nodes= 3;                     accuracy rate=33.37%
###  4) # of hidden nodes= 6;                     accuracy rate=31.41%