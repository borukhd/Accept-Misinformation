#adjust import structure if started as script
from pathlib import Path
from models.LinearCombination.optimizationParameters import OptPars
import os
import sys
import csv
import operator

from ccobra import data
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


""" 
News Item Processing model implementation.
"""
import ccobra
from random import random 
import math
from numpy import mean
import numpy as np
from scipy.optimize._basinhopping import basinhopping
from scipy.optimize import curve_fit



class Hybrid(ccobra.CCobraModel):
    data = None


    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='Hybrid', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {'bestModel' : None}
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])


    def predictS(self, item, **kwargs):
        person = str(int(float(item.identifier)))
        task = item.task_str
        try:
            reply_of_best_model = float(Hybrid.data[person][task][self.parameter['bestModel']])
        except Exception as e:
            print(e)
            print(person)
            print(task)
            #print(Hybrid.data[person].keys())
            return 0.5
        #reply_of_best_model = float(Hybrid.data[person][task][OptPars.parsPerPers[self.name][item.identifier]])
        return reply_of_best_model
        
    def adapt(self, item, target, **kwargs):
        pass
    
    def predict(self, item, **kwargs):
        return 'Accept' if random() < self.predictS(item, **kwargs) else 'Reject'
        
    def pre_train_person(self, dataset):
        if Hybrid.data == None:
            reader = csv.DictReader(open(str(Path(__file__).absolute()).split('models')[0] + 'modeloutputs3.csv', 'r'))
            Hybrid.data = {}
            for row in reader:
                person = str(int(float(row['id']))) 
                if person not in Hybrid.data.keys():
                    Hybrid.data[person] = {}
                Hybrid.data[person][row['task']] = row

        if len(OptPars.parsPerPers.keys()) == 0:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict)
        
        else:
            if len(Hybrid.data.keys()) == 0:
                print('no Hybrid data')
                return 0.5

            model_performances = {}
            person = str(int(float(dataset[0]['item'].identifier))) 
            for task in Hybrid.data[person].keys():
                for modelname in ['CR&time', 'ClassicReas', 'FFT-Max', 'FFT-ZigZag(Z+)', 'HeurRecogn', 'HeurRecogn-lin.', 'S2MR', 'SentimentAnalysis']:
                    try:
                        model_reply = float(Hybrid.data[person][task][modelname])
                    except:
                        print('One error in hybrid')
                        return 0.5
                    if modelname not in model_performances.keys():
                        model_performances[modelname] = []
                    if int(float(Hybrid.data[person][task]['binaryResponse'])) == 0:
                        model_performances[modelname].append(1-model_reply)
                    if int(float(Hybrid.data[person][task]['binaryResponse'])) == 1:
                        model_performances[modelname].append(model_reply)
            sorted_tuples = sorted([(model_performances[a], a) for a in model_performances.keys()], key=meanOfDictEntry)
            self.parameter['bestModel'] = sorted_tuples[-1][1]

        if self.name not in OptPars.parsPerPers.keys():
            OptPars.parsPerPers[self.name] = {}
        OptPars.parsPerPers[self.name][dataset[0]['item'].identifier] = self.parameter.copy()

    def pre_train(self, dataset):

        pass

def meanOfDictEntry(intuple):
    list, entry = intuple
    return np.mean(list)