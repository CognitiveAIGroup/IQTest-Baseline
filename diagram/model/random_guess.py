# --------------------------------------------------------
# Diagram part baseline
# Licensed under The MIT License 
# Written by Pengbo-Hu
# --------------------------------------------------------


import sys
sys.path.append('../')
from utils import split_task
import numpy as np 
import json


def get_splited_training_tasks(path):
    """
        output:
        {   
            "vqa":[...],
            "visual_reasoning":[...],
            "single_image_reasoning":[...],
            "image_analogy":[...],
            "difference_judgment":[...],
            "other":[...]
        }

    """
    data_list = split_task.get_all_training_questions(path)
    splited_tasks = split_task.split_task(data_list)
    return splited_tasks

def get_splited_private_tasks(path):
    """
        output:
        {   
            "vqa":[...],
            "visual_reasoning":[...],
            "single_image_reasoning":[...],
            "image_analogy":[...],
            "difference_judgment":[...],
            "other":[...]
        }

    """
    data_list = split_task.get_all_private_questions(path)
    splited_tasks = split_task.split_task(data_list)
    return splited_tasks



def guess_options(item):
    """ random choice one item from options""" 

    # get the number of options
    length = item["options"].__len__() 
    if length ==0:
        return {item["id"]:{"answer":[1]}}
    
    # random choice an answer
    ans = np.random.randint(1,length+1)
    return {item["id"]:{"answer":[ans]}}


def random_guess(task,path):
    """ random choice for all tasks"""

    # get answers of vqa task
    vqa_ans_list =[guess_options(item) for item in task['vqa']]
    
    # get answers of visual reasoning task
    visual_reasoning_ans_list = [guess_options(item) for item in task['visual_reasoning']]

    # get asnwers of single image reasoning task
    training_data = get_splited_training_tasks(path)['single_image_reasoning']
    answer_set = set()
    for item in training_data:
        answer_set.add(item['answer'])
    answer_set = list(answer_set)
    private_data = task["single_image_reasoning"]
    single_image_reasoning_ans_list = [{item["id"]:{"answer":[str(np.random.choice(answer_set,1)[0])]}} for item in private_data]
    
    # get answers of image analogy task
    image_analogy_ans_list = [guess_options(item) for item in task['image_analogy']]
    
    # get answers of image difference judgment 
    difference_judgment_ans_list = [guess_options(item) for item in task['difference_judgment']]

    # get answers of "other" task
    other_ans_list = [guess_options(item) for item in task['other']]

    final_list = [vqa_ans_list,visual_reasoning_ans_list, \
                        single_image_reasoning_ans_list,image_analogy_ans_list,difference_judgment_ans_list,other_ans_list]
    
    total_list = []
    for item in final_list:
        total_list.extend(item)
    return total_list


def json_save(path,file):
    """ 
        save the json file following required rules (https://github.com/CognitiveAIGroup/IQTest)
    """
    with open("results/"+path,'w') as f:
        json.dump(file,f)

def generate_ans_file(total_list):
    """
        generate the answer file according to the required rules (https://github.com/CognitiveAIGroup/IQTest)
    """
    ans_split = {"T517-621-private":{},"T622-726-private":{},"T727-769-private":{}}
    for item in total_list:
        key = list(item.keys())[0]

        id = key[key.find("_")+1:]
        file_name = key[0:key.find("_")]
        
        ans_split[file_name].update({id:{"answer":item[key]['answer']}})
   
    for key in ans_split.keys():
        json_save(key+".answer.json",ans_split[key])
    


if __name__ == "__main__":
    pass


   