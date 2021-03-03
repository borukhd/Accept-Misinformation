#adjust import structure if started as script
from models.LinearCombination.optimizationParameters import OptPars, RandomDisplacementBounds
import os
from pathlib import Path
import sys

from scipy.optimize import basinhopping
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


""" 
News Item Processing model implementation.
"""
import ccobra
from random import random 
import math
import numpy as np
import pandas as pd
from models.LinearCombination.sentimentanalyzer import SentimentAnalyzer
from fasttrees.fasttrees import FastFrugalTreeClassifier
from sklearn.model_selection import train_test_split
from models.Heuristic.fftTool import FFTtool
from scipy.optimize import * 

class FFTmax(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    componentKeys = []

    
    def __init__(self, name='Fast-Frugal-Tree-Max', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.fft = None
        self.lastnode = None
        FFTmax.componentKeys = ['Familiarity_All_Combined', 'Familiarity_Party_Combined', 'Partisanship_All_Combined', 'Partisanship_Party_Combined', 'ct','crt','conservatism','education', 'reaction_time', 'Exciting_Party_Combined', 'Importance_Party_Combined', 'Partisanship_Party_Combined','Worrying_Party_Combined', 'panasPos', 'panasNeg', ]  #+ [ a for a in SentimentAnalyzer.relevant]#Keys.person + Keys.task
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])

    def pre_train_person(self, dataset):
        pass

    def pre_train(self, dataset):

        if FFTtool.MAX != None:
            return
        
        if len(OptPars.parsPerPers.keys()) == 0:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict)            
        
        trialList = []
        for pers in dataset:
            trialList.extend([a['aux'] for a in pers])
        return self.fitTreeOnTrials(trialList)


    def fitTreeOnTrials(self, trialList, maxLength=-1, person='global'):


        if self.name in OptPars.parsPerPers.keys() and 'global' in OptPars.parsPerPers[self.name].keys():
            predictionQuality, predictionMargin = OptPars.parsPerPers[self.name]['global']
        else:
            maxLength = -1
            predictionQuality = {}
            predictionMargin = {}
            for a in self.componentKeys:
                print(a, 'start')
                bounds = [(0,5)]
                bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                marginOptimum = basinhopping(parametrizedPredictiveQualityLT, [0.00], niter=OptPars.iterationsFFT, T=OptPars.Tfft, minimizer_kwargs={"args" : (a,trialList), "tol":0.001}, take_step=bounded_step ,disp=0)
                predictionMargin['>' + a] = marginOptimum.x[0]
                predictionQuality['>' + a] = marginOptimum.fun
                bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                marginOptimum = basinhopping(parametrizedPredictiveQualityST, [0.00], niter=OptPars.iterationsFFT, T=OptPars.Tfft, minimizer_kwargs={"args" : (a,trialList), "tol":0.001}, take_step=bounded_step ,disp=0)
                predictionMargin['<' + a] = marginOptimum.x[0]
                predictionQuality['<' + a] = marginOptimum.fun
                print(a, 'done')
        #print(predictionQuality, predictionMargin)
        """

        predictionQuality, predictionMargin = {'>Familiarity_All_Combined': -0.8410767696909273, '<Familiarity_All_Combined': -0.5000443807615739, '>Familiarity_Party_Combined': -0.5000443807615739, '<Familiarity_Party_Combined': -0.9664594097556704, '>Partisanship_All_Combined': -0.5000443807615739, '<Partisanship_All_Combined': -0.9999223361292326, '>Partisanship_Party_Combined': -0.9997072599531616, '<Partisanship_Party_Combined': -0.9994490358126722, '>ct': -0.5000443807615739, '<ct': -0.5001166656944526, '>crt': -0.5000443807615739, '<crt': -0.5001090631475624, '>conservatism': -0.5000571820677036, '<conservatism': -0.5001256913021619, '>education': -0.4999429390208336, '<education': -0.5000868340312217, '>reaction_time': -0.5093220338983051, '<reaction_time': -0.5076076400129492}, {'>Familiarity_All_Combined': 0.2662379350081123, '<Familiarity_All_Combined': 0.0, '>Familiarity_Party_Combined': 4.05274531649115, '<Familiarity_Party_Combined': 2.448674318414391, '>Partisanship_All_Combined': 1.4122760569170363, '<Partisanship_All_Combined': 0.5737368357677288, '>Partisanship_Party_Combined': 2.0476830032419393, '<Partisanship_Party_Combined': 4.273745057197547, '>ct': 2.8458550307470336, '<ct': 1.481851582882988, '>crt': 1.8162947794087823, '<crt': 0.7326754642188853, '>conservatism': 3.3754317937159497, '<conservatism': 1.2782228082177964, '>education': 0.0, '<education': 3.525825279351599, '>reaction_time': 1.6138534997640843, '<reaction_time': 3.6593770894646194}

        """

        if self.name not in OptPars.parsPerPers.keys():
            OptPars.parsPerPers[self.name] = {}
        OptPars.parsPerPers[self.name]['global'] = predictionQuality, predictionMargin

        #print(predictionQuality)
        #   print(predictionMargin)
        #calculate order and direction of cues for both Accept (Pos) and Reject (Neg) exits

        orderedConditions = []
        for a in sorted(predictionQuality.items(), key=lambda x: x[1], reverse=False):
            if a[0][1:] not in [i[1:] for i in orderedConditions] and a[0][1:] in self.componentKeys:
                orderedConditions.append(a[0])


        #assemble tree
        for sa in orderedConditions[:maxLength] if maxLength > 0 else orderedConditions:
            b = sa[1:]
            s = sa[0]
            rep0preds, rep1preds, length0, length1 = predictiveQuality_withoutnode(b, s, predictionMargin[sa], trialList)

            exitLeft = rep1preds/length1 >= rep0preds/length0
            cond = '' if exitLeft else 'not '
            cond += 'item[\'' + b + '\'] ' + s + ' ' + str(predictionMargin[sa])
            newnode = Node(cond,True,False)
            if self.fft == None:
                self.fft = newnode
                self.lastnode = self.fft
                self.fft.length = 1
            else:
                self.fft.length += 1
                if exitLeft:
                    self.lastnode.left = newnode
                    self.lastnode = self.lastnode.left
                else:
                    self.lastnode.right = newnode
                    self.lastnode = self.lastnode.right
        FFTtool.MAX = self.fft
        print(FFTtool.MAX.getstring())

    def predictS(self, item, **kwargs):
        #prepare item features format and partisanship
        aux = kwargs['kwargs'] if len(kwargs.keys()) == 1 else kwargs

        #evaluate FFT from root node on
        return FFTtool.MAX.run(aux, show=False)

    def adapt(self, item, target, **kwargs):
        pass
    
    def predict(self, item, **kwargs):
        return 'Accept' if random() < self.predictS(item, **kwargs) else 'Reject'

    def toCommandList(self,pars):
        optCommands = []
        i = 0
        parKeys = self.sorted_par_keys
        for a in parKeys:
            if len(pars)<=i: 
                print('keys length error', self.name)
                break
            optCommands.append('self.parameter[\'' + a + '\'] = ' + str(pars[i]))
            i += 1
        return optCommands
    
    def executeCommands(self, commands):
        for command in commands:
            exec(command)


def parametrizedPredictiveQualityLT(margin, a, trialList):
    #node = Node('item[\'' + a + '\'] > ' + str(margin[0]), True, False)
    rep0preds, rep1preds, length0, length1 = predictiveQuality_withoutnode(a, '>', margin[0], trialList)
    return -1*max(rep0preds/length0, rep1preds/length1)
def parametrizedPredictiveQualityST(margin, a, trialList):
    #node = Node('item[\'' + a + '\'] < ' + str(margin[0]), True, False)
    rep0preds, rep1preds, length0, length1 = predictiveQuality_withoutnode(a, '<', margin[0], trialList)
    return -1*max(rep0preds/length0, rep1preds/length1)


def predictiveQuality(node, trialList):
    rep0preds = 0
    rep1preds = 0
    length0 = 1
    length1 = 1
    for item in trialList:
        if 1 == node.run(item):
            rep1preds += int(bool(item['truthful'] == 1))
            length1 += 1
        else:
            rep0preds += int(bool(item['truthful'] == 0))
            length0 += 1
    return rep0preds, rep1preds, length0, length1

def predictiveQuality_withoutnode(b, s, predicitonMargin, trialList):
    rep0preds = 0
    rep1preds = 0
    length0 = 1
    length1 = 1
    for item in trialList:
        if s == '>':
            if item[b] > predicitonMargin:
                rep1preds += int(bool(item['truthful'] == 1))
                length1 += 1
            else:
                rep0preds += int(bool(item['truthful'] == 0))
                length0 += 1
        else:
            if item[b] < predicitonMargin:
                rep1preds += int(bool(item['truthful'] == 1))
                length1 += 1
            else:
                rep0preds += int(bool(item['truthful'] == 0))
                length0 += 1
    return rep0preds, rep1preds, length0, length1

class Node:
    def __init__(self, conditionstr, left = True, right = False, length = 1):
        self.condition = conditionstr
        self.left = left
        self.right = right
        self.length = length

    def run(self, item, show = False, length = -1):
        self.show = show
        aux = item

        if self.show:
            print(aux['aux'])
            print(self.condition)

        if length < 0:
            if eval(self.condition):
                if isinstance(self.left,bool):
                    return self.left
                return self.left.run(aux)
            else:
                if isinstance(self.right,bool):
                    return self.right
                return self.right.run(aux)
        elif length == 0:
            if eval(self.condition):
                return True
            else:
                return False
        else:
            if eval(self.condition):
                if isinstance(self.left,bool):
                    return self.left
                return self.left.run(aux, length= length-1)
            else:
                if isinstance(self.right,bool):
                    return self.right
                return self.right.run(aux, length= length-1)    

    def getstring(self):
        a = ''
        if isinstance(self.left,bool):
            a = 'if ' + self.condition + ': \n\treturn ' + str(self.left) + '\n' 
            a += 'return ' + str(self.right) if isinstance(self.right,bool) else self.right.getstring()
        else:
            a = 'if not ' + self.condition + ': \n\treturn ' + str(self.right) + '\n' 
            a += 'return ' + str(self.left) if isinstance(self.left,bool) else self.left.getstring()
        return a 

