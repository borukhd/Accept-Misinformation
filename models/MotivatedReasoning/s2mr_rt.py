#adjust import structure if started as script
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


class S2MR(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='S2MR&RT'):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}

        #dictionary for testing with value from rough optimization on Experiment 1
        optdict = {'Kc': -0.056536155629514195, 'Kl': -0.09074348945552314, 'Mc': -0.027349753930657073, 'Ml': -0.036805428943731885, '!alpha' : 0}
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        kwargs = kwargs['kwargs'] if len(kwargs.keys()) == 1 else kwargs

        conservativePerson = kwargs['conservatism'] >= 2.9
        liberalPerson = kwargs['conservatism'] <= 2.9
        cons_partisanship = 3.8
        lib_partisanship = 2.2
        if kwargs['Partisanship_Party_Combined']>cons_partisanship and conservativePerson:
            threshold = self.parameter['Kc'] + kwargs['crt'] * self.parameter['Mc']
        elif kwargs['Partisanship_Party_Combined']<lib_partisanship and liberalPerson:
            threshold = self.parameter['Kc'] + kwargs['crt'] * self.parameter['Mc']
        elif kwargs['Partisanship_Party_Combined']<lib_partisanship and conservativePerson:
            threshold = self.parameter['Kl'] + kwargs['crt'] * self.parameter['Ml']
        elif kwargs['Partisanship_Party_Combined']>cons_partisanship and liberalPerson:
            threshold = self.parameter['Kl'] + kwargs['crt'] * self.parameter['Ml']
        else:
            threshold = 0.5
        return threshold + kwargs['reaction_time']*self.parameter['!alpha']

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

    def pre_train_person(self, dataset_in):
        if len(OptPars.parsPerPers.keys()) == 0:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict)
        
        dataset = []
        for a in dataset_in:
            a['aux']['id'] = a['item'].identifier
            dataset.append(a['aux'])

        commandlist = OptPars.parsPerPers['S2MR']['global']
        self.executeCommands(commandlist)


        if self.name in OptPars.parsPerPers.keys() and dataset[0]['id'] in OptPars.parsPerPers[self.name]:
            commandlist = OptPars.parsPerPers[self.name][dataset[0]['id']]
            commandlist.extend(OptPars.parsPerPers['S2MR']['global'])
        else:
            #print('commandlist',commandlist)
            #Optimpizing paramaters per person 
            trialList = dataset
            if len(self.parameter.keys()) > 1:
                with np.errstate(divide='ignore'):
                    bounds = [(-5,5)]
                    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                    personOptimum = basinhopping(self.itemsOnePersonThisModelPeformance, [0],T=OptPars.T, take_step= bounded_step, niter= OptPars.iterations, minimizer_kwargs ={'args':tuple([trialList]), "method":"L-BFGS-B"})
                #print(personOptimum.fun, personOptimum.x, )
                optpars = personOptimum.x
            else: 
                optpars = [] 
            optpars = [a for a in optpars] + [self.parameter[a] for a in self.sorted_par_keys if a != '!alpha']
            commandlist = self.toCommandList(optpars)
            #print('optpars', commandlist)
            if self.name not in OptPars.parsPerPers.keys():
                OptPars.parsPerPers[self.name] = {}
            OptPars.parsPerPers[self.name][dataset[0]['id']] = commandlist
        self.executeCommands(commandlist)

    def itemsOnePersonThisModelPeformance(self, pars, items):
        #input: list of items
        performanceOfPerson = []
        parsall = [a for a in pars] + [self.parameter[a] for a in self.sorted_par_keys if a != '!alpha'] 
        self.executeCommands(self.toCommandList(parsall))
        for item in items:
            for a in item['aux'].keys():
                item[a] = item['aux'][a]
            pred = min(1.0,max(self.predictS(item=item['item'], kwargs= item['aux']),0.0)) 
            predictionPerf = 1.0 - abs(float(item['binaryResponse']) - pred)
            performanceOfPerson.append(predictionPerf)
        return -1*mean(performanceOfPerson) 
        
    def pre_train(self, dataset_in):
        if len(OptPars.parsPerPers.keys()) == 0:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict)

        commandlist = OptPars.parsPerPers[self.name.split('&')[0]]['global']
        self.executeCommands(commandlist)