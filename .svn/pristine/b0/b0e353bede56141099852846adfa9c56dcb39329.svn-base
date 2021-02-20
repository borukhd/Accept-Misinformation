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
from Recommender.RS import RS
from LinearCombination.optimizationParameters import OptPars
from scipy.optimize._basinhopping import basinhopping
from numpy import mean
import numpy as np

class RecommenderPlinear(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='RecommenderPersonLinear', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.relevant = ['education', 'crt']
        for a in self.relevant: 
            self.parameter[a] = 0
                
        super().__init__(name, ['misinformation'], ['single-choice'])

    def globalTrain(self, trialList):
        if RS.trained:
            return
        RS.trained = True
        personFeatures = ['crt', 'conservatism', 'ct', 'education', 'reaction_time', 'crt','ct','conservatism','panasPos','panasNeg','education', 'reaction_time','accimp','age','gender']
        for item in trialList:
            if item['item'].identifier not in RS.featuresOfAllPeople.keys():
                RS.featuresOfAllPeople[item['item'].identifier] = {a: item[a] for a in personFeatures if a in item.keys()}
            if item['item'].identifier not in RS.repliesOfAllPeople.keys():
                RS.repliesOfAllPeople[item['item'].identifier] = {}
            RS.repliesOfAllPeople[item['item'].identifier][item['item'].task[0][0]] = float(item['binaryResponse'])
        #print(len(trialList))

    def pre_train(self, dataset):
        trialList = []
        for pers in dataset:
            perslist = []
            for a in pers:
                persdict = {}
                persdict = a['aux']
                persdict['item'] = a['item']
                perslist.append(persdict)
            trialList.extend(perslist)
        return self.globalTrain(trialList)


    def similar(self, person1, person2):
        if (person1 == person2):
            return False
        difference = sum(abs((RS.featuresOfAllPeople[person1][key] - RS.featuresOfAllPeople[person2][key])*self.parameter[key]) for key in self.relevant if RS.featuresOfAllPeople[person2][key] != 'NoInfo' and RS.featuresOfAllPeople[person1][key] != 'NoInfo')
        if difference < 1:
            return True
        else: 
            return False


    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        self.parameter = {'education': 0.41276193531865624, 'crt': -1.4941098613407457}
        self.parameter = {'education': 0.7369543795644158, 'crt': 0.4801283733698878}
        if not RS.trained:
            return 
        repliesOfSimilar = 0
        numberOfReplies = 0
        for person in RS.featuresOfAllPeople.keys():
            if numberOfReplies > 20:
                break
            if ('S' in [a for a in RS.featuresOfAllPeople[person].keys()][0] != 'S' in item.task_str):
                continue
            if item.identifier in RS.featuresOfAllPeople.keys() and self.similar(person, item.identifier) and item.task_str in RS.repliesOfAllPeople[person].keys():
                repliesOfSimilar += RS.repliesOfAllPeople[person][item.task[0][0]]
                numberOfReplies += 1
            else:
                continue
        if numberOfReplies == 0:
            return 0.5
        meanResponse = repliesOfSimilar/numberOfReplies
        return int(meanResponse > 0.5)


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
        #Optimpizing similarity measure paramaters per person 
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





        