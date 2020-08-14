# --------------------------------------------------------
# Diagram part baseline
# Licensed under The MIT License 
# Written by Pengbo-Hu
# --------------------------------------------------------


import json 
import numpy as np 



def get_all_training_questions(path):
    """
        output example: 
        [
            {
                'stem': 'A chain is passed around four wheels, 1, 2, 3 and 4, so that the large wheel, 2, moves clockwise. 
                        Which of the wheels turns anti-clockwise?\n\n![](logic-diagram-public/media/image1.png)', 
                'options': ['1', '2', '3', '4'], 
                'answer': [4]
            }, 
            ...
        ]
    """
    public_data = json.load(open(path+'/public.config.json'),encoding='utf-8')  
    data_path_public = public_data['test_suites']['diagram']
    data_path_public_answer = [item+".answer" for item in data_path_public]

    data_list = []
    for q_path,ans_path in zip(data_path_public,data_path_public_answer):
        questions = json.load(open(path+ "/"+q_path+".json"))
        answers=json.load(open(path+"/"+ans_path+".json"))
        for item in questions:
            data_list.append({"stem":item['stem'],"options":item['options'],'answer':answers[str(item['id'])]['answer']})
    
    return data_list


def get_all_private_questions(path):
    """
        output example:
        [
            {
                'stem': '7. Which is the odd one out?', 
                'options': ['![](T517-621-private/media/517A.png)', '![](T517-621-private/media/517B.png)', 
                           '![](T517-621-private/media/517C.png)', '![](T517-621-private/media/517D.png)', 
                           '![](T517-621-private/media/517E.png)'], 
                'id': 'T517-621-private_0'
            },
            ...
        ]
    """
    private_data = json.load(open(path+'/private.config.json'),encoding='utf-8')  
    data_path_private = private_data['test_suites']['diagram']
    data_list = []
    for q_path in data_path_private:
        questions = json.load(open(path+"/"+q_path+".json"))
        for item in questions:
            data_list.append({"stem":item['stem'],"options":item['options'],'id':q_path+"_"+str(item["id"])})
           
 
    return data_list


def options_png_count(options):
    """get the number of png file in options """
    i= 0
    for item in options:
        if 'png' in item:
            i+=1
    return i 




def split_task(data_list):
    """
        split all questions into five different tasks
        1. vqa task example
        {
            'stem': 'What number continues the sequence?\n\n![](T517-621-private/media/520.png)', 
            'options': ['23', '22', '54', '55'], 
            'id': 'T517-621-private_3'
        }
        2. visual reasoning task example
        {
            'stem': 'Which shield below is most like the shield above?\n\n![](T517-621-private/media/524.png)', 
            'options': ['![](T517-621-private/media/524A.png)', '![](T517-621-private/media/524B.png)', 
                       '![](T517-621-private/media/524C.png)', '![](T517-621-private/media/524D.png)', '![](T517-621-private/media/524E.png)'], 
            'id': 'T517-621-private_7'
        }
        3. single image reasoning task example
        {
            'stem': '![](T622-726-private/media/657.png)', 
            'options': [], 
            'id': 'T622-726-private_35'
        }
        4. image analogy task example
        {
            'stem': '![](T517-621-private/media/518-1.png) is to ![](T517-621-private/media/518-2.png) as ![](T517-621-private/media/518-3.png) is to', 
            'options': ['![](T517-621-private/media/518A.png)', '![](T517-621-private/media/518B.png)', 
                       '![](T517-621-private/media/518C.png)', '![](T517-621-private/media/518D.png)', '![](T517-621-private/media/518E.png)'], 
            'id': 'T517-621-private_1'
        }
        5. diference judgment task example
        {
            'stem': '7. Which is the odd one out?', 
            'options': ['![](T517-621-private/media/517A.png)', '![](T517-621-private/media/517B.png)', 
                       '![](T517-621-private/media/517C.png)', '![](T517-621-private/media/517D.png)', 
                       '![](T517-621-private/media/517E.png)'], 
            'id': 'T517-621-private_0'
        }
        6. others 
           This part contain other samples that can not be splited by following rules.
    """
    tasks = {"vqa":[],"visual_reasoning":[],"single_image_reasoning":[],"image_analogy":[],"difference_judgment":[],"other":[]}
    for item in data_list:
        # vqa 143
        if item['stem'].count("png")==1 and options_png_count(item['options'])==0 and item['options']!=[]:
            tasks["vqa"].append(item)
        # single image reasoning 265
        elif item['stem'].count("png")==1 and options_png_count(item['options'])==0 and item['options']==[]:
            tasks["single_image_reasoning"].append(item)
        # visual reasoning 66 
        elif item['stem'].count("png")==1 and options_png_count(item['options'])>0:
            tasks["visual_reasoning"].append(item)
        # image analogy 6
        elif item['stem'].count("png")>2:
            tasks["image_analogy"].append(item)
        # difference judgment 101
        elif item['stem'].count("png")==0 and options_png_count(item['options'])>0:
            tasks["difference_judgment"].append(item)
        # others 6
        else:
            tasks["other"].append(item)
        
    return tasks


if __name__ == "__main__":
    pass