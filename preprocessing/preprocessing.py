#coding: utf-8
from collections import defaultdict
import numpy as np
#import pandas as pd
from abc import ABCMeta, abstractmethod
import argparse

NAMES = "first_name.txt"
OBJ = "objects.txt"
PLACES = "house_places.txt"

def retrieve_first_names(input_file):
    with open(input_file,'r') as inp:
        a =  inp.readlines()
    a = map(lambda x : x.split('\t')[0],a)
    a = list(np.random.choice(a,len(a),replace=False))
    return a

def retrieve_file(input_file):
    with open(input_file,'r') as inp:
        a =  inp.readlines()
        aa = list()
        for i in a:
            aa.append(i[:-1])
        aa = list(np.random.choice(aa,len(aa),replace=False))
        return aa


class Scenario(object):
    """Bordes and al. used a rather simple world : 4 characters, 3 objects and 5 rooms"""
    def __init__(self,training_sample,number_of_characters = 4, number_of_objects = 3, number_of_locations = 5):
        self._generate_variables(number_of_characters, number_of_objects, number_of_locations)
        self.training_samples = list()
        if isinstance(training_sample,list):
            for ts in training_sample:
                self.add_training_sample(ts)
        else :
            self.add_training_sample(training_sample)

    def _generate_variables(self,number_of_characters = 4, number_of_objects = 3, number_of_locations = 5):
        self.number_of_characters = number_of_characters
        self.number_of_objects = number_of_objects
        self.number_of_locations = number_of_locations
        self.names = retrieve_first_names(NAMES)[:self.number_of_characters]
        self.objects = retrieve_file(OBJ)[:self.number_of_characters]
        self.locations = retrieve_file(PLACES)[:self.number_of_characters]

    def add_training_sample(self,training_sample):
        assert hasattr(training_sample,'__call__'), "Training sample should be callable"
        assert hasattr(training_sample,'_generate_questions'), "the traning sample should have _generate_questions "
        assert hasattr(training_sample,'_generate_context'), "the traning sample should have _generate_context attr "
        t = training_sample(self.objects,self.names,self.locations)
        self.training_samples.append(t)

    def create_corpus(self,number_of_sentences = 70):
        self._generate_variables()
        self.dataset = defaultdict(dict)
        for i in range(number_of_sentences) :
            self._generate_variables()
            t = np.random.choice(self.training_samples)
            self.dataset[i] = t.generate_training_example()
        return self.dataset

    @classmethod
    def generate_corpus(cls):
        s = cls(Locations)
        ss = s.create_corpus()
        return write_corpus(ss,infile=True)


class TrainingItems(object):
    __metaclass__ = ABCMeta

    def __init__(self,list_of_objects,list_of_characters,list_of_places):
        self.list_of_objects = list_of_objects
        self.list_of_characters = list_of_characters
        self.list_of_places = list_of_places
        self._context = self._generate_context()
        self._question = self._generate_questions()

    @abstractmethod
    def _generate_questions(self):
        pass

    @abstractmethod
    def _generate_context(self):
        pass

    @abstractmethod
    def _generate_answer(self,indice):
        pass

    def generate_training_example(self):
        d = {}
        c = self._generate_context()
        q = self._generate_questions()
        c.append(q["question"])
        d["X"] = '. '.join(c)
        d["Y"] = q["answer"]
        d["fact_ids"] = q["fact_ids"]
        return d


class Locations(TrainingItems):
    def _generate_context(self):
        self._context = list()
        self.c0, self.c1, self.c2 = np.random.choice(self.list_of_characters, 3, replace=False)
        self.p0, self.p1, self.p2 = list(np.random.choice(self.list_of_places, 3, replace=False))
        self.o0 = np.random.choice(self.list_of_objects)

        self._context.append(
          self.c2 + " is in  " + self.p0 )
        self._context.append(self.c0 + " went to " + self.p0)
        self._context.append(self.c1 + " took " + self.o0 + " in " + self.p1)
        self._context.append(self.c1 + " left " + self.o0 + " in " + self.p2)
        self._context.append(self.c1 + " joined " + self.c0 + " in " + self.p0)
        self._context.append(
          self.c2 + " went from " + self.p0 + " to the " + self.p2 + " because he can't stand " + self.c1)
        return self._context

    def _generate_questions(self,level=0):
        self._questions = defaultdict(dict)
        self._questions[0]["question"] = "Why did " + self.c2 + " left ?"
        self._questions[0]["answer"] = " he doesn't like " + self.c1
        self._questions[1]["question"] = "Where is the " + self.o0 + " ?"
        self._questions[1]["answer"] = self.p2

        l = np.random.choice(len(self._questions))
        # Case question 1 - reason x left
        if l == 0 :
            fact_ids = "42,43,44,45,46,47,48,49,50"
        else :
        # Case question 2 - location of object
            fact_ids = "22,23,24,25,26,27,28"
        return {"question": self._questions[l]["question"],
                "answer": self._generate_answer(l) + self._questions[l]["answer"],
                "fact_ids":fact_ids}

    def _generate_answer(self, indice,level=1):
        self._formulation = defaultdict(list)
        if level == 0 :
            return ""
        else :
            self._formulation[0].append("I guess because ")
            self._formulation[0].append("I think it's because ")
            self._formulation[1].append("I think it's in ")
            self._formulation[1].append("Probably in ")
            return self._formulation[indice][np.random.choice(len(self._formulation[indice]))]

def write_corpus(dataset,infile=False):
    """
    Write the generated corpus inside
    """
    assert isinstance(dataset,dict), "the given dataset is not of the right type. {} found, dict required ".format(type(dataset))
    if infile:
        with open("output.txt","w") as out:
            for _,item in dataset.iteritems():
                sol = " | ".join([item["X"],item["Y"], item["fact_ids"]])
                out.write(sol+'\n')
    else :
        out = []
        for _,item in dataset.iteritems():
            sol = " | ".join([item["X"],item["Y"],item['fact_ids']])
            out.append(sol)
        return out


def main(**args):
    a = Scenario.generate_corpus()
    # test = a[3]
    # print(test)
    # ind = test.split("|")[-1]
    # print(ind)
    # ind_int = map(int,map(lambda x :x.decode('utf-8'),ind.split(",")))
    # print(map(lambda x : test.split(" ")[x-1], ind_int))



if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description =__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filename',default='dataset.hd5', help='The filename in which we want to save the created dataset (default : dataset.hd5)')
    args = parser.parse_args()
    a = dict(vars(args))
    main(**a)