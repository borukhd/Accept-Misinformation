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
        self.parameter['Cr'] = 0.65 
        self.parameter['Cf'] = 0.2 
        self.parameter['Mr'] = 0.13                              
        self.parameter['Mf'] = - 0.12 
        super().__init__(name, ['misinformation'], ['single-choice'])


    def predictS(self, item, **kwargs):
        self.parameter = CR.globalpar
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        if kwargs['truthful']:
            threshold = self.parameter['Cr'] + self.parameter['Mr'] * kwargs['crt']
        if not kwargs['truthful']:
            threshold = self.parameter['Cf'] + self.parameter['Mf'] * kwargs['crt']
        return threshold
        

    def adapt(self, item, target, **kwargs):
        pass
    
    def predict(self, item, **kwargs):
        return 'Accept' if random() < self.predictS(item, **kwargs) else 'Reject'

    def pre_train(self, dataset):
        #Globally fits a linear equation of CRT on real and fake new item measures 
        if len(CR.globalpar.keys())>0:
            return
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
        self.parameter['Mr'] = realLine[0]
        self.parameter['Cr'] = realLine[1]                         
        self.parameter['Mf'] = fakeLine[0]
        self.parameter['Cf'] = fakeLine[1]
        CR.globalpar = self.parameter.copy()

def fit_func(crt, m, c):
    return crt*m + c