#adjust import structure if started as script
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


""" 
News Item Processing model implementation.
"""
import ccobra
from random import random 
import math
from LinearCombination.sentimentanalyzer import SentimentAnalyzer
from LinearCombination.optimizationParameters import OptPars
from scipy.optimize._basinhopping import basinhopping
from numpy import mean
import numpy as np


class LP(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='SentimentAnalysis', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.thresh = 1
        self.parameter = {}
        SentimentAnalyzer.initialize()
        self.relevant = SentimentAnalyzer.relevant
        for a in self.relevant:
            self.parameter[a] = 0

        #dictionary for testing with value from rough optimization on Experiment 1
        optdict = {'negative_emotion': 3.488183752051738, 'fight': 4.795255469272864, 'optimism': 5.4718777782354735, 'sexual': 3.583167795093339, 'money': 5.249519675409447, 'aggression': 5.464386990594476, 'affection': -2.3986185486873572, 'positive_emotion': -1.4074963226019577, 'science': 1.5522568483198222, 'law': 12.874721726587898, 'crime': -2.9929120337902457}
        for a in optdict.keys():
            self.parameter['Sent: ' + a] = optdict[a]
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        analysis = SentimentAnalyzer.analysis(item)
        p = 0
        for a in self.parameter.keys():
            if a.split(' ')[1] not in self.relevant:
                continue
            p += analysis[a]*  self.parameter[a]
        return 1 if self.thresh < p else 0

    def adapt(self, item, target, **kwargs):
        pass

    def adaptS(self, itemPair):
        pass
    
    def predict(self, item, **kwargs):
        return 'Accept' if random() < self.predictS(item, **kwargs) else 'Reject'

    def toCommandList(self,pars):
        optCommands = []
        i = 0
        parKeys = sorted(self.parameter.keys())
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

    def pre_train_person(self, dataset):
        #Optimpizing paramaters per person 
        trialList = []
        for pers in dataset:
            trialList.extend([pers])
        if len(self.parameter.keys()) > 0:
            with np.errstate(divide='ignore'):
                personOptimum = basinhopping(self.itemsOnePersonThisModelPeformance, [1]*len(self.parameter.keys()), niter=OptPars.iterations, stepsize=3, T=4,  minimizer_kwargs={"args" : (trialList)})
            optpars = personOptimum.x
        else: 
            optpars = [] 
        self.executeCommands(self.toCommandList(optpars))

    def itemsOnePersonThisModelPeformance(self, pars, items):
        #input: list of items
        items = [a for a in items]
        performanceOfPerson = []
        self.executeCommands(self.toCommandList(pars))
        for item in items:
            pred = min(1.0,max(self.predictS(item=item['item'], kwargs= item['aux']),0.0)) 
            if item['aux']['binaryResponse']:
                predictionPerf = min(1.0,max(self.predictS(item=item['item'], kwargs=item['aux']),0.0)) 
            elif not item['aux']['binaryResponse']:
                predictionPerf = 1.0 - pred
            else:
                print('Error')
            performanceOfPerson.append(predictionPerf)
        return -1*mean(performanceOfPerson) 




