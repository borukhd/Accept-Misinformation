#adjust import structure if started as script
import os
import sys
from pathlib import Path

from scipy.optimize.optimize import brute
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


""" 
News Item Processing model implementation.
"""
import ccobra
from random import random 
import math
from models.LinearCombination.sentimentanalyzer import SentimentAnalyzer
from models.LinearCombination.optimizationParameters import OptPars, RandomDisplacementBounds
from scipy.optimize._basinhopping import basinhopping
from numpy import mean
import numpy as np


class LP(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    current_mean = []

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
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        analysis = SentimentAnalyzer.an_dict(item.task_str)
        p = 0
        for a in self.parameter.keys():
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

    def pre_train(self, dataset):
        pass

    def pre_train_person(self, dataset_in):
        #Optimpizing paramaters per person 
        if len(OptPars.parsPerPers.keys()) == 0:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict) 
        
        if self.name in OptPars.parsPerPers.keys() and dataset_in[0]['item'].identifier in OptPars.parsPerPers[self.name]:
            commandlist = OptPars.parsPerPers[self.name][dataset_in[0]['item'].identifier]
        else:
            dataset = []
            for a in dataset_in:#[c for b in dataset_in for c in b]:
                a['aux']['id'] = a['item'].identifier
                a['aux']['task'] = a['item'].task_str
                for k in a['aux'].keys():
                    a[k] = a['aux'][k]
                dataset.append(a)
            trialList = dataset
            with np.errstate(divide='ignore'):
                bounds = [(-10,10)*len(self.parameter.keys())]
                bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                personOptimum = basinhopping(self.itemsOnePersonThisModelPeformance, [1/len(self.parameter.keys())] * len(self.parameter.keys()), T=OptPars.T, take_step= bounded_step, niter= OptPars.iterations, minimizer_kwargs ={'args':tuple([trialList]), "method":"L-BFGS-B"})
            optpars = personOptimum.x
            commandlist = self.toCommandList(optpars)
            if self.name not in OptPars.parsPerPers.keys():
                OptPars.parsPerPers[self.name] = {}
            OptPars.parsPerPers[self.name][dataset[0]['item'].identifier] = commandlist
            LP.current_mean.append(personOptimum.fun)
        #print(np.mean(LP.current_mean), dataset_in[0]['item'].identifier, personOptimum.fun, commandlist)
        self.executeCommands(commandlist)

    def itemsOnePersonThisModelPeformance(self, pars, items):
        #input: list of items
        performanceOfPerson = []
        self.executeCommands(self.toCommandList(pars))
        for item in items:
            pred = min(1.0,max(self.predictS(item=item['item'], kwargs= item['aux']),0.0)) 
            predictionPerf = 1.0 - abs(float(item['binaryResponse']) - pred)
            performanceOfPerson.append(predictionPerf)
        return -1*mean(performanceOfPerson) 




