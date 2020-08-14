'''
This file take the question file that contains stem, options, category and id,
and the answer file that contains answer and hint for a question,
then combine the stem and answer together for training.
The outputed two files are train set and valid set.
'''

import json
import argparse
import os

train_ratio=0.8
data_file={'question':'seq-public.json','answer':'seq-public.answer.json',\
        'train':'seq-train.json','valid':'seq-valid.json'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        help="The input data dir. Should contain the question and answer json file")
    args = parser.parse_args()
    
    # each sequence info has an unique id, 
    # take the question file to find corresponding items to an id
    question_file=os.path.join(args.data_dir, data_file['question'])
    seqpub=open(question_file, 'r')
    seqpub = json.load(seqpub)
    seqdict={}
    for i in range(len(seqpub)):
        curid=str(seqpub[i]["id"])
        seqdict[curid]={"stem": seqpub[i]["stem"], "options": seqpub[i]["options"], "category": seqpub[i]["category"]}

    ans_file=os.path.join(args.data_dir, data_file['answer'])
    seqans=open(ans_file, 'r')
    seqans = json.load(seqans)

    # for each id in answer file, combine the stem and answer together
    resdict={}
    cnt=0
    for i in seqans:
        resdict[i]={}
        if len(seqdict[i]["options"])==0:
            ans=seqans[i]["answer"]
        else:
            idx=seqans[i]["answer"][0]-1
            ans=seqdict[i]["options"][idx]

        stem=''
        for j in seqdict[i]["stem"]:
            if j!='?':
                stem+=j
            else:
                stem+=ans
        resdict[i]["stem"]=stem
        resdict[i]["category"]=seqdict[i]["category"]
        resdict[i]["answer"]=ans
        resdict[i]["hint"]=seqans[i]["hint"]
        cnt+=1
    train_cnt=int(cnt*train_ratio)
    
    # seperate train set and valid set
    train_dict={}
    valid_dict={}
    cnt=0
    for key in resdict:
        if cnt<train_cnt:
            train_dict[key]=resdict[key]
        else:
            valid_dict[key]=resdict[key]
        cnt+=1

    print('size of train set:',len(train_dict)," size of valid set:",len(valid_dict))
    train_file=open(os.path.join(args.data_dir, data_file['train']), "w", encoding='utf-8')
    valid_file=open(os.path.join(args.data_dir, data_file['valid']), "w", encoding='utf-8')
    json.dump(train_dict, train_file, indent=2, sort_keys=True, ensure_ascii=False)
    json.dump(valid_dict, valid_file, indent=2, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    main()