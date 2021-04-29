#adjust import structure if started as script
import os
import sys

from ccobra import data
from scipy.optimize.optimize import brute
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
from pathlib import Path
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from pathlib import Path


""" 
News Item Processing model implementation.
"""
import ccobra
from random import random
import math
from scipy.optimize._basinhopping import basinhopping
from numpy import mean
import numpy as np
from models.LinearCombination.optimizationParameters import OptPars, RandomDisplacementBounds


class RHlinear(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='HeurRecogn-lin.', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.parameter['kappa'] = 1
        self.parameter['alpha'] = 1
        #dictionary for testing with value from rough optimization on Experiment 1
        optdict = {'kappa': -5.192396551875893, 'alpha': 2.2913602334440673}
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        return kwargs['Familiarity_Party_Combined'] * self.parameter['alpha'] + self.parameter['kappa']


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

    def pre_train(self, dataset):
        pass

    def pre_train_person(self, dataset_in): 
        if len(OptPars.parsPerPers.keys()) == 0:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict)

        dataset = []
        for a in dataset_in:
            a['aux']['id'] = a['item'].identifier
            dataset.append(a['aux'])
        
        if self.name in OptPars.parsPerPers.keys() and dataset[0]['id'] in OptPars.parsPerPers[self.name]:
            commandlist = OptPars.parsPerPers[self.name][dataset[0]['id']]
        else:
            #Optimpizing paramaters per person 
            trialList = dataset
            if len(self.parameter.keys()) > 0:
                with np.errstate(divide='ignore'):
                    bounds = [(-15,15)*len(self.parameter.keys())]
                    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                    personOptimum = basinhopping(self.itemsOnePersonThisModelPeformance, [1/len(self.parameter.keys())] * len(self.parameter.keys()), T=OptPars.T, take_step= bounded_step, niter= OptPars.iterations, minimizer_kwargs ={'args':tuple([trialList]), "method":"L-BFGS-B"})
                optpars = personOptimum.x
            else: 
                optpars = [] 
            commandlist = self.toCommandList(optpars)
            if self.name not in OptPars.parsPerPers.keys():
                OptPars.parsPerPers[self.name] = {}
            OptPars.parsPerPers[self.name][dataset[0]['id']] = commandlist
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




