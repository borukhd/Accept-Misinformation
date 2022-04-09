#adjust import structure if started as script
from cgi import print_arguments
import os
import sys

from scipy.optimize.optimize import brute
from pathlib import Path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


""" 
News Item Processing model implementation.
"""
import ccobra
from random import random
import math
from models.LinearCombination.optimizationParameters import OptPars, RandomDisplacementBounds
from scipy.optimize._basinhopping import basinhopping
from numpy import mean
import numpy as np


class PARTR(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='PartR'):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}

        #dictionary for testing 
        optdict = {}
        optdict['offs_r'] = 0.52 
        optdict['offs_f'] = 0.3 
        optdict['cons_p_r'] = 0.25      
        optdict['cons_p_f'] = - 0.25 
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        kwargs = kwargs['kwargs'] if len(kwargs.keys()) == 1 else kwargs

        kwargs['truthful'] = bool(int(float(kwargs['truthful'])))

        if kwargs['truthful']:
            threshold = self.parameter['offs_r'] + self.parameter['cons_p_r'] * kwargs['conservatism']
        if not kwargs['truthful']:
            threshold = self.parameter['offs_f'] + self.parameter['cons_p_f'] * kwargs['conservatism']
            #threshold = 1 - threshold
        return threshold

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

    def pre_train(self, dataset_in):
        if len(OptPars.parsPerPers.keys()) == 0:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict)

        if False and self.name in OptPars.parsPerPers.keys() and 'global' in OptPars.parsPerPers[self.name]:
            commandlist = OptPars.parsPerPers[self.name]['global']
        else:
            dataset = []
            for a in [c for b in dataset_in for c in b]:
                a['aux']['id'] = a['item'].identifier
                for k in a['aux'].keys():
                    a[k] = a['aux'][k]
                dataset.append(a)
            #Optimpizing paramaters per person 
            trialList = dataset
            if len(self.parameter.keys()) > 0:
                with np.errstate(divide='ignore'):
                    bounds = [(-5,5)*len(self.parameter.keys())]
                    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                    personOptimum = basinhopping(self.itemsOnePersonThisModelPeformance, [1/len(self.parameter.keys())] * len(self.parameter.keys()), T=OptPars.T, take_step= bounded_step, niter= OptPars.iterations, minimizer_kwargs ={'args':tuple([trialList]), "method":"L-BFGS-B"})
                optpars = personOptimum.x
            else: 
                optpars = [] 
            commandlist = self.toCommandList(optpars)
            if self.name not in OptPars.parsPerPers.keys():
                OptPars.parsPerPers[self.name] = {}
            OptPars.parsPerPers[self.name]['global'] = commandlist
        self.executeCommands(commandlist)

    def pre_train_person(self, dataset_in):
        return
        if len(OptPars.parsPerPers.keys()) <= 1:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict)

        if self.name in OptPars.parsPerPers.keys() and dataset_in[0]['item'].identifier in OptPars.parsPerPers[self.name].keys():
            commandlist = OptPars.parsPerPers[self.name][dataset_in[0]['item'].identifier]
        else:
            #print('pre_train_person for model', self.name, 'person', str(dataset_in[0]['item'].identifier))
            dataset = []
            for a in dataset_in:
                a['aux']['id'] = a['item'].identifier
                for k in a['aux'].keys():
                    a[k] = a['aux'][k]
                dataset.append(a)
            #Optimpizing paramaters per person 
            trialList = dataset
            if len(self.parameter.keys()) > 0:
                with np.errstate(divide='ignore'):
                    bounds = [(-5,5)*len(self.parameter.keys())]
                    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                    personOptimum = basinhopping(self.itemsOnePersonThisModelPeformance, [1/len(self.parameter.keys())] * len(self.parameter.keys()), T=OptPars.T, take_step= bounded_step, niter= OptPars.iterations, minimizer_kwargs ={'args':tuple([trialList]), "method":"L-BFGS-B"})
                optpars = personOptimum.x
            else: 
                optpars = [] 
            commandlist = self.toCommandList(optpars)
            if self.name not in OptPars.parsPerPers.keys():
                OptPars.parsPerPers[self.name] = {}
            OptPars.parsPerPers[self.name][dataset[0]['item'].identifier] = commandlist
        self.executeCommands(commandlist)


    def itemsOnePersonThisModelPeformance(self, pars, items):
        #input: list of items
        performanceOfPerson = []
        self.executeCommands(self.toCommandList(pars))
        for item in items:
            pred = min(1.0,max(self.predictS(item=item['item'], kwargs= item),0.0)) 
            predictionPerf = 1.0 - abs(float(item['binaryResponse']) - pred)
            performanceOfPerson.append(predictionPerf)
        return -1*mean(performanceOfPerson) 


        