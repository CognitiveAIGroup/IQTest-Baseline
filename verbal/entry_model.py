from iqtest import iqtest_base
import torch
import re
import numpy
import gensim
from scipy.spatial.distance import cosine as dist_cosine

class IQTestVerbal(iqtest_base.IQTestModelBase):
    model = None

    def pre_run(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            './train_data/numberbatch-en.txt', binary=False)
        return None

    def solve_analogy(self, question):
        """Given a specific analogy question
        use the metric with weight constraint proposed by 
        Speeret al. [Speer et al., 2017]

        the candidate word with the
        highest score is selected as the answer
        """
        max_s = 0;
        searchWord = re.search(r'(.*) is to (.*) as (.*) is to', question['stem'])

        a = searchWord.group(1)
        b = searchWord.group(2)
        c = searchWord.group(3)

        if a not in self.model:
            return [question['id'], [1]]
        if b not in self.model:
            return [question['id'], [1]]
        if c not in self.model:
            return [question['id'], [1]]

        a1 = numpy.array(self.model[a])
        b1 = numpy.array(self.model[b])
        a2 = numpy.array(self.model[c])
        flag = False
        for idx, answer in enumerate(question['options']):
            if answer in self.model:
                flag = True
            else:
                continue

            b2 = numpy.array(self.model[answer])
            #score calculation with pre determined weight
            s = numpy.dot(a1,a2) + numpy.dot(b1,b2) + 0.2 * numpy.dot(b2-a2,b1-a1) + 0.6 * numpy.dot(b2-b1,a2-a1)
            if s > max_s:
                max_s = s
                curr_idx = idx
        if not flag:
            return [question['id'], [1]]
        else:
            return [question['id'], [curr_idx+1]]

    def solve_classification(self, question):
        """Given a specific classification question
        use the method mentioned in
        Speeret al. [Speer et al., 2017]
        """
        def get_distance(elem):
            return elem[2]

        n = len(question["options"])
        dist_list = list()

        for x,option_x in enumerate(question["options"]):
            for y,option_y in enumerate(question["options"]):
                if (option_x not in self.model) or (option_y not in self.model):
                    return [question['id'],[1]]
                if x < y:
                    #cosine similarity between candidate x and y
                    dist_list.append([x,y,dist_cosine(self.model[option_x],self.model[option_y])])
                    dist_list.sort(key = get_distance,reverse = True)

        candidate1 = dist_list[0][0]
        candidate2 = dist_list[0][1]
        for i in range(1,n-1):
            if (candidate1 == dist_list[i][0]) or (candidate1 == dist_list[i][1]):
                return [question['id'],[candidate1+1]]
            elif (candidate2 == dist_list[i][0]) or (candidate2 == dist_list[i][1]):
                return [question['id'],[candidate2+1]]
            else:
                continue
                
        return [question['id'],[1]]

    def solve(self, question):
        """Given a verbal question in question list
        sort into finer catergory

        return the answer in the form of [question id,answer]
        """
        if question['category'] == 'verbal-analogy':
            return self.solve_analogy(question)
        elif question['category'] == 'verbal-classification':
            return self.solve_classification(question)
        else:
            #when model can not handle the question
            return [question['id'], [1]]

class IQTestSeqSample(iqtest_base.IQTestEvalBase):
    def pre_run(self):
        # setup your model here
        pass

    def solve(self, question):
        # question_id, answer_list
        return [question['id'], [1]]

class IQTestDiagramSample(iqtest_base.IQTestModelBase):
    def pre_run(self):
        # setup your model here
        pass

    def solve(self, question):
        # question_id, answer_list
        return [question['id'], [1]]


def get_model_object(category: str) -> object:
    """ model by category
    if don't support category, return None 

    :param category: test category 
    :type category: str
    :return: Model 
    :rtype: object
    """
    if category == 'seq':
        return IQTestSeqSample()
    elif category == 'diagram':
        return IQTestDiagramSample()
    elif category == 'verbal':
        return IQTestVerbal()
    return None

def global_pre_run():
    """ global pre run function used to set up environment
    """
    device = torch.device("cuda:0")
    try:
        data = torch.tensor(numpy.random.rand(10), device=device)
        print(data)
    except:
        pass