'''
This file mainly used to generate answer json file from existing model.
Parameters are in line 9-12. you can change them to fit your own setting.
This py file will generate a answer json file named 'seq-private.answer.json' in data_dir. This is the ANSWER FILE you need to upload to the website.
'''

from main import *

data_dir = './data'
model_path = './model/RNNmodel10.0w_lr0.001_hidden8_norm1.pkl'
data_file={'question':'seq-public.json','answer':'seq-public.answer.json','private':'seq-private.json',\
        'train':'seq-train.json','valid':'seq-valid.json','test':'parsed-seq-private.json'}

INPUT_SIZE = 3
OUTPUT_SIZE = 1
NUM_LAYERS = 1

if __name__ == "__main__":

    hidden_size = model_path.split('/')[-1].split('hidden')[-1].split('_')[0]  # Get hidden size from saved model filename.
    # Process private sequence json file.
    private_file=os.path.join(data_dir, data_file['private'])
    seqpri=open(private_file, 'r',encoding='utf-8')
    TempFileContent = seqpri.read()
    TempFileContent = TempFileContent.replace('( )', '?').replace('()', '?')  # Replace () into ? for further processing.

    seqpri = json.loads(TempFileContent)
    test_dict={}
    for seq in seqpri:
        test_dict[seq['id']]={}
        stem=''
        for j in seq["stem"]:
            if j!='?':
                stem+=j
            else:
                stem=stem[:-1]# remove , or other separation symbol
        test_dict[seq['id']]['stem']=stem
        test_dict[seq['id']]['options']=seq['options']
        test_dict[seq['id']]['category']=seq['category']
    test_file=open(os.path.join(data_dir, data_file['test']), "w", encoding='utf-8')
    json.dump(test_dict, test_file, indent=2, sort_keys=True, ensure_ascii=False)
    # Process private sequence json file ends.

    question_file=os.path.join(data_dir, 'seq-private.json')
    option_file=open(question_file, 'r')
    option_file = json.load(option_file)
    seqdict={}
    for i in range(len(option_file)):
        curid=str(option_file[i]["id"])
        seqdict[curid]={"stem": option_file[i]["stem"], "options": option_file[i]["options"], "category": option_file[i]["category"]}

    # Process Input Train and Valid data.  length-based normalize
    seq_test_data = SeqData_normbylen(args.data_dir, 'seq-test_yym.json')
    # print(seq_test_data.length)

    seqrnn = SeqRNN(INPUT_SIZE, args.hidden_size, OUTPUT_SIZE)
    # Load state dict.
    seqrnn.load_state_dict(torch.load(model_path))

    test_answer, acc = seq_test_data.cal_accuracy(seqrnn, seqdict, gen_answer=True)
    with open(os.path.join(args.data_dir, 'seq-private.answer.json'), 'w', encoding='utf-8') as f_json_output:
        json.dump(test_answer, f_json_output)
    # print(test_answer)
    print('Answer file generating complete. ')

