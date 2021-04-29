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
from scipy.optimize._basinhopping import basinhopping
from numpy import mean
from scipy import mean
from scipy.optimize.minpack import curve_fit
import numpy as np
from models.LinearCombination.optimizationParameters import OptPars, RandomDisplacementBounds

class RT(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    globalpar = {}
    def __init__(self, name='CR&time', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.Cparameter = {}
            
        self.Cparameter['Cr'] = 0.65 
        self.Cparameter['Cf'] = 0.2 
        self.Cparameter['Mr'] = 0.13                              
        self.Cparameter['Mf'] = - 0.12 

        for a in ['alpha']:
            self.parameter[a] = 0
        #dictionary for testing with value from rough optimization on Experiment 1
        optdict = {'alpha': -0.015201555811670677}
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        if kwargs['truthful']:
            threshold = self.Cparameter['Cr'] + self.Cparameter['Mr'] * kwargs['crt']
        if not kwargs['truthful']:
            threshold = self.Cparameter['Cf'] + self.Cparameter['Mf'] * kwargs['crt']
        return threshold + kwargs['reaction_time']*self.parameter['alpha']

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
        if len(items) == 1:
            items = items[0]
        performanceOfPerson = []
        self.executeCommands(self.toCommandList(pars))
        for item in items:
            pred = min(1.0,max(self.predictS(item=item['item'], kwargs= item['aux']),0.0)) 
            predictionPerf = 1.0 - abs(float(item['binaryResponse']) - pred)
            performanceOfPerson.append(predictionPerf)
        return -1*mean(performanceOfPerson) 


    def pre_train(self, dataset):

        if self.name in OptPars.parsPerPers.keys() and 'global' in OptPars.parsPerPers[self.name].keys():
            self.Cparameter = OptPars.parsPerPers[self.name]['global']
        else:

            #Globally fits a linear equation of CRT on real and fake new item measures 
            trialList = []
            for pers in dataset:
                trialList.extend([pers])
            mean_acc_values_fake = {}
            mean_acc_values_real = {}
            acc_values_fake = {}
            acc_values_real = {}
            for alist in trialList:
                if False and len(alist) < max(len(l) for l in trialList):
                    continue
                for a in alist:
                    a['aux']['id'] = a['item'].identifier
                    for k in a['aux'].keys():
                        a[k] = a['aux'][k]
                    crt_value = a['aux']['crt']
                    if a['aux']['truthful'] == 1:
                        if crt_value not in acc_values_real.keys():
                            acc_values_real[crt_value] = []
                        acc_values_real[crt_value].append(abs(a['aux']['binaryResponse']))
                    else:
                        if crt_value not in acc_values_fake.keys():
                            acc_values_fake[crt_value] = []
                        acc_values_fake[crt_value].append(abs(a['aux']['binaryResponse']))
            for key in sorted([a for a in acc_values_real.keys()]):
                if len(acc_values_real[key]) / max(len(l)/2 for l in trialList) < 10:
                    continue
                mean_acc_values_real[key] =  mean(acc_values_real[key])
            for key in sorted([a for a in acc_values_fake.keys()]):
                if len(acc_values_fake[key]) / max(len(l)/2 for l in trialList) < 10:
                    continue
                mean_acc_values_fake[key] =  mean(acc_values_fake[key])
            ry = [mean_acc_values_real[a] for a in mean_acc_values_real.keys()]
            rx = [a for a in mean_acc_values_real.keys()]
            fy = [mean_acc_values_fake[a] for a in mean_acc_values_fake.keys()]
            fx = [a for a in mean_acc_values_fake.keys()]
            realOpt = curve_fit(fit_func,np.array(rx), np.array(ry), method = 'trf')
            fakeOpt = curve_fit(fit_func,np.array(fx), np.array(fy), method = 'trf')
            realLine = realOpt[0]
            fakeLine = fakeOpt[0]
            self.Cparameter['Mr'] = realLine[0]
            self.Cparameter['Cr'] = realLine[1]                         
            self.Cparameter['Mf'] = fakeLine[0]
            self.Cparameter['Cf'] = fakeLine[1]

            #determined graphically; better performance than fitting linear equation as above
            self.parameter['Cr'] = 0.6
            self.parameter['Cf'] = 0.2 
            self.parameter['Mr'] = 0.15    
            self.parameter['Mf'] = - 0.2 

            if self.name not in OptPars.parsPerPers.keys():
                OptPars.parsPerPers[self.name] = {}
            OptPars.parsPerPers[self.name]['global'] = self.Cparameter.copy()
        

def fit_func(crt, m, c):
    return crt*m + c

