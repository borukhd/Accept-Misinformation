#adjust import structure if started as script
from pathlib import Path
from models.LinearCombination.optimizationParameters import OptPars
import os
import sys

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



class CR(ccobra.CCobraModel):
    globalpar = {}

    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='ClassicReas', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}                          
        self.parameter['Cr'] = 0.52 
        self.parameter['Cf'] = 0.3 
        self.parameter['Mr'] = 0.25      
        self.parameter['Mf'] = - 0.25 
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])


    def predictS(self, item, **kwargs):
        if self.name in OptPars.parsPerPers.keys() and 'global' in OptPars.parsPerPers[self.name]:
            self.parameter = OptPars.parsPerPers[self.name]['global']
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        kwargs['truthful'] = bool(int(float(kwargs['truthful'])))
        if kwargs['truthful']:
            threshold = self.parameter['Cr'] + self.parameter['Mr'] * kwargs['crt']
        if not kwargs['truthful']:
            threshold = self.parameter['Cf'] + self.parameter['Mf'] * kwargs['crt']
        return threshold
        

    def adapt(self, item, target, **kwargs):
        pass
    
    def predict(self, item, **kwargs):
        return 'Accept' if random() < self.predictS(item, **kwargs) else 'Reject'
    def pre_train_person(self, dataset):
        pass
    def pre_train(self, dataset):
        if len(OptPars.parsPerPers.keys()) == 0:
            parameterDict = open(str(Path(__file__).absolute()).split('models')[0] + 'models/pars_other.txt', 'r').read()
            OptPars.parsPerPers = eval(parameterDict)

        #Globally fits a linear equation of CRT on real and fake new item measures 
        if self.name in OptPars.parsPerPers.keys() and 'global' in OptPars.parsPerPers[self.name].keys():
            self.parameter = OptPars.parsPerPers[self.name]['global']
        else:
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

                    crt_value = a['crt']
                    if a['truthful'] == 1:
                        if crt_value not in acc_values_real.keys():
                            acc_values_real[crt_value] = []
                        acc_values_real[crt_value].append(abs(a['binaryResponse']))
                    else:
                        if crt_value not in acc_values_fake.keys():
                            acc_values_fake[crt_value] = []
                        acc_values_fake[crt_value].append(abs(a['binaryResponse']))
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
            
            #plt.plot()

            self.parameter['Mr'] = realLine[0]
            self.parameter['Cr'] = realLine[1]                         
            self.parameter['Mf'] = fakeLine[0]
            self.parameter['Cf'] = fakeLine[1]    

            #determined graphically; better performance than fitting linear equation as above
            self.parameter['Cr'] = 0.6
            self.parameter['Cf'] = 0.2 
            self.parameter['Mr'] = 0.15    
            self.parameter['Mf'] = - 0.2 

        if self.name not in OptPars.parsPerPers.keys():
            OptPars.parsPerPers[self.name] = {}
        OptPars.parsPerPers[self.name]['global'] = self.parameter.copy()

def fit_func(crt, m, c):
    return crt*m + c