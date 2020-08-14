## IQ Test Baseline Model for Diagram Part 
This part is the random guess baseline from the diagram part of IQ Test.


#### Download dataset 
Dataset can be acquired from following link: 
https://github.com/CognitiveAIGroup/IQTest

After getting dataset, put it into "dataset" folder or set the path on setup phase.

#### Sub-tasks
The dataset contain five sub-tasks:
```
1. vqa (Multi-choice Visual Question Answering) task example
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

```
Task-split script locate on ```utils/split_task.py```.

#### Setup

```
python run.py --path="dataset"
```

To add：
1. ```--path=str```, e.g.```--path='dataset'``` to set the location of dataset.

#### Result file
All result json file are saved at ```result```folder. The method of uploading answer file are presented on https://github.com/CognitiveAIGroup/IQTest

#### Recommended reference articles
Because this task is very hard, we recommend several reference articles as following:
```
@article{tang2020multi,
  title={Multi-Granularity Modularized Network for Abstract Visual Reasoning},
  author={Tang, Xiangru and Wang, Haoyuan and Pan, Xiang and Qi, Jiyang},
  journal={arXiv preprint arXiv:2007.04670},
  year={2020}
}


@inproceedings{zhang2019raven,
  title={RAVEN: A dataset for relational and analogical visual reasoning},
  author={Zhang, Chi and Gao, Feng and Jia, Baoxiong and Zhu, Yixin and Zhu, Song-Chun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5317--5327},
  year={2019}
}

@article{hoshen2017iq,
  title={IQ of neural networks},
  author={Hoshen, Dokhyam and Werman, Michael},
  journal={arXiv preprint arXiv:1710.01692},
  year={2017}
}

```