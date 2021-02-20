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

class LC(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='LinearCombination', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.thresh = 1
        self.componentKeys = ['Exciting_Democrats_Combined', 'Exciting_Republicans_Combined', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined', 'Importance_Democrats_Combined', 'Importance_Republicans_Combined', 'Likelihood_Democrats_Combined', 'Likelihood_Republicans_Combined', 'Partisanship_All_Combined', 'Partisanship_All_Partisan', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined', 'Worrying_Democrats_Combined', 'Worrying_Republicans_Combined']
        for a in self.componentKeys:
            if a not in SentimentAnalyzer.relevant:
                self.parameter[a] = 0
        #dictionary for testing with value from rough optimization on Experiment 1
        optdict =  {'Exciting_Party_Combined': -3.0933215184624734, 'Familiarity_Party_Combined': 17.83214047528395, 'Importance_Party_Combined': 3.185192558938423, 'Partisanship_All_Combined': 22.939675300159564, 'Partisanship_All_Partisan': 2.0986712022626524, 'Partisanship_Party_Combined': -15.087158032169091, 'Worrying_Party_Combined': -36.55073351519288}
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        for a in self.componentKeys:
            if a not in kwargs.keys():
                continue
            if kwargs['conservatism'] >= 3.5:
                if 'Republicans' in a:
                    kwargs[a.replace('Republicans', 'Party')] = kwargs[a]
            elif kwargs['conservatism'] <= 3.5:
                if 'Democrats' in a:
                    kwargs[a.replace('Democrats', 'Party')] = kwargs[a]
        p = 0
        for a in self.parameter.keys():
            if a not in kwargs.keys():
                continue
            p += kwargs[a] * self.parameter[a] * 0.3
        if 1 < p:
            return 1
        else:
            return 0


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

