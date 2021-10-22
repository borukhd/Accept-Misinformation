#adjust import structure if started as script
import os
from pathlib import Path
import sys

from numpy.lib.function_base import percentile
from pathlib import Path

from ccobra import data
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
from scipy.optimize._basinhopping import basinhopping
from numpy import mean
import numpy as np
from models.LinearCombination.optimizationParameters import OptPars, RandomDisplacementBounds

class RH(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='HeurRecogn&RT', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.parameter['fam'] = 1
        #dictionary for testing with value from rough optimization on Experiment 1
        optdict = {'fam': 2.390676377924067, 'alpha' : 0}
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        return kwargs['Familiarity_Party_Combined'] < self.parameter['fam'] + kwargs['reaction_time']*self.parameter['alpha'] 


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

        if '&' in self.name:
            if self.name.split('&')[0] in OptPars.parsPerPers.keys() and dataset[0]['id'] in OptPars.parsPerPers[self.name.split('&')[0]]:
                commandlist = OptPars.parsPerPers[self.name.split('&')[0]][dataset[0]['id']]
                self.executeCommands(commandlist)
                #print('valid model order', commandlist)
            else:
                print('invalid model evaluation order')

        if self.name in OptPars.parsPerPers.keys() and dataset[0]['id'] in OptPars.parsPerPers[self.name]:
            commandlist = OptPars.parsPerPers[self.name][dataset[0]['id']]
        else:
            #Optimpizing paramaters per person 
            trialList = dataset
            if len(self.parameter.keys()) > 1:
                with np.errstate(divide='ignore'):
                    bounds = [(-5,5)*1]
                    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                    personOptimum = basinhopping(self.itemsOnePersonThisModelPeformance, [0] * (len(self.parameter.keys())-1),T=OptPars.T, take_step= bounded_step, niter= OptPars.iterations, minimizer_kwargs ={'args':tuple([trialList]), "method":"L-BFGS-B"})
                #print(personOptimum.fun, personOptimum.x, )
                optpars = personOptimum.x
            else: 
                optpars = [] 
            optpars = [a for a in optpars] + [self.parameter[a] for a in self.sorted_par_keys if a != 'alpha']
            commandlist = self.toCommandList(optpars)
            #print('optpars', commandlist)
            if self.name not in OptPars.parsPerPers.keys():
                OptPars.parsPerPers[self.name] = {}
            OptPars.parsPerPers[self.name][dataset[0]['id']] = commandlist
        self.executeCommands(commandlist)

    def itemsOnePersonThisModelPeformance(self, pars, items):
        #input: list of items
        performanceOfPerson = []
        parsall = [a for a in pars] + [self.parameter[a] for a in self.sorted_par_keys if a != 'alpha'] 
        self.executeCommands(self.toCommandList(parsall))
        for item in items:
            for a in item['aux'].keys():
                item[a] = item['aux'][a]
            pred = min(1.0,max(self.predictS(item=item['item'], kwargs= item['aux']),0.0)) 
            predictionPerf = 1.0 - abs(float(item['binaryResponse']) - pred)
            performanceOfPerson.append(predictionPerf)
        return -1*mean(performanceOfPerson) 


