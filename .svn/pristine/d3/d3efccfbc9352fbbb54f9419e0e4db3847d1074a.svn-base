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
import numpy as np
import pandas as pd
from LinearCombination.sentimentanalyzer import SentimentAnalyzer
from fasttrees.fasttrees import FastFrugalTreeClassifier
from sklearn.model_selection import train_test_split
from Heuristic.fftTool import FFTtool
from scipy.optimize import * 

class FFTzigzag(ccobra.CCobraModel):
    """ FFTzigzag CCOBRA implementation.
    """
    componentKeys = []
    
    def __init__(self, name='Fast-Frugal-Tree-ZigZag(Z+)', commands = []):
        """ Initializes the model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        SentimentAnalyzer.initialize()
        self.parameter = {}
        #self.parameter['thresh'] = 1
        self.fft = None
        self.lastnode = None
        FFTzigzag.componentKeys = ['crt','ct','conservatism','panasPos','panasNeg','education', 'reaction_time','accimp','age','gender','Exciting_Democrats_Combined', 'Exciting_Republicans_Combined', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined', 'Importance_Democrats_Combined', 'Importance_Republicans_Combined', 'Likelihood_Democrats_Combined', 'Likelihood_Republicans_Combined', 'Partisanship_All_Combined', 'Partisanship_All_Partisan', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined','Sharing_Democrats_Combined', 'Sharing_Republicans_Combined', 'Worrying_Democrats_Combined','Worrying_Republicans_Combined',] + [ a for a in ['Sent: negative_emotion', 'Sent: health', 'Sent: dispute', 'Sent: government', 'Sent: healing', 'Sent: military', 'Sent: fight', 'Sent: meeting', 'Sent: shape_and_size', 'Sent: power', 'Sent: terrorism', 'Sent: competing', 'Sent: office', 'Sent: money', 'Sent: aggression', 'Sent: wealthy', 'Sent: banking', 'Sent: kill', 'Sent: business', 'Sent: speaking', 'Sent: work', 'Sent: valuable', 'Sent: economics', 'Sent: payment', 'Sent: friends', 'Sent: giving', 'Sent: help', 'Sent: school', 'Sent: college', 'Sent: real_estate', 'Sent: reading', 'Sent: gain', 'Sent: science', 'Sent: negotiate', 'Sent: law', 'Sent: crime', 'Sent: stealing', 'Sent: strength'] if a in SentimentAnalyzer.relevant]#Keys.person + Keys.task 
        super().__init__(name, ['misinformation'], ['single-choice'])

    def pre_train(self, dataset):
        #Globally trains zigzag FFT on data for all persons
        if FFTtool.ZigZag != None:
            return
        trialList = []
        for pers in dataset:
            perslist = []
            for a in pers:
                persdict = {}
                persdict = a['aux']
                persdict['item'] = a['item']
                perslist.append(persdict)
            trialList.extend(perslist)
        return self.fitTreeOnTrials(trialList)


    def fitTreeOnTrials(self, trialList, maxLength=-1, person='global'):
        for item in trialList:
            for a in FFTzigzag.componentKeys:
                if 'Sent' in a:
                    if a.split(' ')[1] not in SentimentAnalyzer.relevant:
                        continue
                    item[a] = SentimentAnalyzer.analysis(item['item'])[a.split(' ')[1]]
                if a not in item.keys():
                    continue
                if item['conservatism'] >= 3.5:
                    if 'Republicans' in a:
                        item[a.replace('Republicans', 'Party')] = item[a]
                        item.pop(a,None)
                        item.pop(a.replace('Republicans','Democrats'))
                elif item['conservatism'] <= 3.5:
                    if 'Democrats' in a:
                        item[a.replace('Democrats', 'Party')] = item[a]
                        item.pop(a,None)
                        item.pop(a.replace('Democrats','Republicans'))
        maxLength = -1
        predictionQuality = {}
        predictionMargin = {}
        #precalculated predictive quality of individual cues
        predictionQuality, predictionMargin = {'>crt': -0.9996858309770656, '<crt': -0.9986962190352021, '>conservatism': -0.5007337607167683, '<conservatism': -0.5003962614779187, '>reaction_time': -0.6742837053963789, '<reaction_time': -0.5003962614779187, '>Familiarity_Party_Combined': -0.9999108416547788, '<Familiarity_Party_Combined': -0.5003962614779187, '>Partisanship_All_Combined': -0.9996858309770656, '<Partisanship_All_Combined': -0.9986962190352021, '>Sent: negative_emotion': -0.9996849401386263, '<Sent: negative_emotion': -0.5003962614779187, '>Sent: health': -0.5003962614779187, '<Sent: health': -0.6130589430894309, '>Sent: dispute': -0.6450213266656861, '<Sent: dispute': -0.5003962614779187, '>Sent: government': -0.9986893840104849, '<Sent: government': -0.5003962614779187, '>Sent: healing': -0.9993434011818779, '<Sent: healing': -0.5003962614779187, '>Sent: military': -0.9998210130660462, '<Sent: military': -0.5003962614779187, '>Sent: fight': -0.7681743973878805, '<Sent: fight': -0.5003962614779187, '>Sent: meeting': -0.9997459349593496, '<Sent: meeting': -0.5003962614779187, '>Sent: shape_and_size': -0.9996849401386263, '<Sent: shape_and_size': -0.5003962614779187, '>Sent: power': -0.9998427425695864, '<Sent: power': -0.5003962614779187, '>Sent: terrorism': -0.9996849401386263, '<Sent: terrorism': -0.5003962614779187, '>Sent: competing': -0.9996849401386263, '<Sent: competing': -0.5003962614779187, '>Sent: office': -0.9996846420687481, '<Sent: office': -0.5003962614779187, '>Sent: money': -0.999787007454739, '<Sent: money': -0.5003962614779187, '>Sent: aggression': -0.9999039385206532, '<Sent: aggression': -0.5003962614779187, '>Sent: wealthy': -0.9997457411645054, '<Sent: wealthy': -0.5003962614779187, '>Sent: banking': -0.9993421052631579, '<Sent: banking': -0.5003962614779187, '>Sent: kill': -0.9998209169054442, '<Sent: kill': -0.5003962614779187, '>Sent: business': -0.9998209169054442, '<Sent: business': -0.5003962614779187, '>Sent: speaking': -0.999842668344871, '<Sent: speaking': -0.5003962614779187, '>Sent: work': -0.9998209169054442, '<Sent: work': -0.5003962614779187, '>Sent: valuable': -0.9995617879053462, '<Sent: valuable': -0.5003962614779187, '>Sent: economics': -0.9998593134496342, '<Sent: economics': -0.5003962614779187, '>Sent: payment': -0.9993421052631579, '<Sent: payment': -0.5003962614779187, '>Sent: friends': -0.9986807387862797, '<Sent: friends': -0.5003962614779187, '>Sent: giving': -0.999344262295082, '<Sent: giving': -0.5003962614779187, '>Sent: help': -0.9986893840104849, '<Sent: help': -0.7597607052896725, '>Sent: school': -0.5003962614779187, '<Sent: school': -0.7597607052896725, '>Sent: college': -0.5003962614779187, '<Sent: college': -0.7597607052896725, '>Sent: real_estate': -0.9996851385390428, '<Sent: real_estate': -0.5003962614779187, '>Sent: reading': -0.9986893840104849, '<Sent: reading': -0.5003962614779187, '>Sent: gain': -0.9996851385390428, '<Sent: gain': -0.5003962614779187, '>Sent: science': -0.9996851385390428, '<Sent: science': -0.5003962614779187, '>Sent: negotiate': -0.9986893840104849, '<Sent: negotiate': -0.5003962614779187, '>Sent: law': -0.9998427920138343, '<Sent: law': -0.5003962614779187, '>Sent: crime': -0.9998213966779782, '<Sent: crime': -0.5003962614779187, '>Sent: stealing': -0.9996861268047709, '<Sent: stealing': -0.5003962614779187, '>Sent: strength': -0.9997928319867413, '<Sent: strength': -0.5003962614779187}, {'>crt': 2.108692258820941, '<crt': 4.221334524079497, '>conservatism': 2.4733894476129064, '<conservatism': 0.0, '>reaction_time': 2.0723292361934, '<reaction_time': 0.0, '>Familiarity_Party_Combined': 2.1152623939462805, '<Familiarity_Party_Combined': 0.0, '>Partisanship_All_Combined': 2.1077281741718545, '<Partisanship_All_Combined': 4.307574112948173, '>Sent: negative_emotion': 0.0, '<Sent: negative_emotion': 0.0, '>Sent: health': 2.6557183446611052, '<Sent: health': 9.278284718691123e-06, '>Sent: dispute': 0.0, '<Sent: dispute': 0.0, '>Sent: government': 1.951016698230048, '<Sent: government': 0.0, '>Sent: healing': 0.0, '<Sent: healing': 0.0, '>Sent: military': 0.0, '<Sent: military': 0.0, '>Sent: fight': 0.0, '<Sent: fight': 0.0, '>Sent: meeting': 0.0, '<Sent: meeting': 0.0, '>Sent: shape_and_size': 0.0, '<Sent: shape_and_size': 0.0, '>Sent: power': 0.0, '<Sent: power': 0.0, '>Sent: terrorism': 0.0, '<Sent: terrorism': 0.0, '>Sent: competing': 0.0, '<Sent: competing': 0.0, '>Sent: office': 0.0, '<Sent: office': 0.0, '>Sent: money': 0.0, '<Sent: money': 0.0, '>Sent: aggression': 0.0, '<Sent: aggression': 0.0, '>Sent: wealthy': 0.0, '<Sent: wealthy': 0.0, '>Sent: banking': 0.0, '<Sent: banking': 0.0, '>Sent: kill': 0.0, '<Sent: kill': 0.0, '>Sent: business': 0.0, '<Sent: business': 0.0, '>Sent: speaking': 0.0, '<Sent: speaking': 0.0, '>Sent: work': 0.0, '<Sent: work': 0.0, '>Sent: valuable': 0.0, '<Sent: valuable': 0.0, '>Sent: economics': 0.0, '<Sent: economics': 0.0, '>Sent: payment': 0.0, '<Sent: payment': 0.0, '>Sent: friends': 0.0, '<Sent: friends': 0.0, '>Sent: giving': 1.303250036513945, '<Sent: giving': 0.0, '>Sent: help': 1.3873013971092425, '<Sent: help': 9.278284717033553e-06, '>Sent: school': 2.0401223021398005, '<Sent: school': 9.278284719291338e-06, '>Sent: college': 1.4173002482924701, '<Sent: college': 9.278284719291338e-06, '>Sent: real_estate': 0.0, '<Sent: real_estate': 0.0, '>Sent: reading': 0.0, '<Sent: reading': 0.0, '>Sent: gain': 0.0, '<Sent: gain': 0.0, '>Sent: science': 0.0, '<Sent: science': 0.0, '>Sent: negotiate': 1.7659083823077841, '<Sent: negotiate': 0.0, '>Sent: law': 0.0, '<Sent: law': 0.0, '>Sent: crime': 0.0, '<Sent: crime': 0.0, '>Sent: stealing': 0.0, '<Sent: stealing': 0.0, '>Sent: strength': 0.0, '<Sent: strength': 0.0}
        orderedConditionsPos = []
        orderedConditionsNeg = []
        #calculate order and direction of cues for both Accept (Pos) and Reject (Neg) exits
        for a in sorted(predictionQuality.items(), key=lambda x: x[1], reverse=False):
            if a[0][1:] not in item.keys():
                continue
            b = a[0][1:]
            s = a[0][0]
            cond = 'item[\'aux\'][\'' + b + '\'] ' + s + ' ' + str(predictionMargin[a[0]])
            newnode = Node(cond,True,False)
            rep0preds, rep1preds, length0, length1 = predictiveQuality(newnode, trialList)
            #determine exit direction
            if rep1preds/length1 >= rep0preds/length0:
                if a[0][1:] not in [i[1:] for i in orderedConditionsPos + orderedConditionsNeg] and a[0][1:] in [a.replace('Democrats','Party').replace('Republicans','Party') for a in FFTzigzag.componentKeys]:
                    orderedConditionsPos.append(a[0])
            else:
                if a[0][1:] not in [i[1:] for i in orderedConditionsNeg +orderedConditionsPos] and a[0][1:] in  [a.replace('Democrats','Party').replace('Republicans','Party') for a in FFTzigzag.componentKeys]:
                    orderedConditionsNeg.append(a[0])
        orderedConditions = []
        for i in range(max(len(orderedConditionsNeg), len(orderedConditionsPos))):
            if len(orderedConditionsNeg) > i:
                orderedConditions.append(orderedConditionsNeg[i])
            if len(orderedConditionsPos) > i:
                orderedConditions.append(orderedConditionsPos[i])

        exitLeft = True #for first exit, as Z+ version implemented
        #assemble tree
        for sa in orderedConditions[:maxLength] if maxLength > 0 else orderedConditions:
            b = sa[1:]
            s = sa[0]
            cond = 'item[\'aux\'][\'' + b + '\'] ' + s + ' ' + str(predictionMargin[sa])
            newnode = Node(cond,True,False)
            if self.fft == None:
                self.fft = newnode
                self.lastnode = self.fft
            else:
                if not exitLeft:
                    self.lastnode.left = newnode
                    self.lastnode = self.lastnode.left
                else:
                    self.lastnode.right = newnode
                    self.lastnode = self.lastnode.right
            exitLeft = not exitLeft
        FFTtool.ZigZag = self.fft

    def predictS(self, item, **kwargs):
        #prepare item features format and partisanship

        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        try:
            if 'aux' not in item.keys():
                item['aux'] = item
        except:
            tempitem = item
            item = {}
            item['item'] = tempitem
            item['aux'] = kwargs
        
        for a in FFTzigzag.componentKeys:
            if 'Sent' in a:
                if a.split(' ')[1] not in SentimentAnalyzer.relevant:
                    continue
                item['aux'][a] = SentimentAnalyzer.analysis(item['item'])[a.split(' ')[1]]
            if a.replace('Republicans', 'Party') not in item['aux'].keys() and a.replace('Democrats', 'Party') not in item['aux'].keys():
                continue
            if item['aux']['conservatism'] >= 3.5:
                if 'Republicans' in a:
                    item['aux'][a.replace('Republicans', 'Party')] = item['aux'][a]
                    item['aux'].pop(a,None)
                    item['aux'].pop(a.replace('Republicans','Democrats'))
            elif item['aux']['conservatism'] <= 3.5:
                if 'Democrats' in a:
                    item['aux'][a.replace('Democrats', 'Party')] = item['aux'][a]
                    item['aux'].pop(a,None)
                    item['aux'].pop(a.replace('Democrats','Republicans'))

        #evaluate FFT from root node on
        return FFTtool.ZigZag.run(item, **kwargs, show=False)

    def adapt(self, item, target, **kwargs):
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
        


def parametrizedPredictiveQualityLT(margin, a, trialList):
    node = Node('item[\'aux\'][\'' + a + '\'] > ' + str(margin[0]), True, False)
    rep0preds, rep1preds, length0, length1 = predictiveQuality(node, trialList)
    return -1*max(rep0preds/length0, rep1preds/length1)
def parametrizedPredictiveQualityST(margin, a, trialList):
    node = Node('item[\'aux\'][\'' + a + '\'] < ' + str(margin[0]), True, False)
    rep0preds, rep1preds, length0, length1 = predictiveQuality(node, trialList)
    return -1*max(rep0preds/length0, rep1preds/length1)


def predictiveQuality(node, trialList):
    rep0preds = 0
    rep1preds = 0
    length0 = 1
    length1 = 1
    for item in trialList:
        if 'aux' not in item.keys():
            item['aux'] = item
        if item['aux']['conservatism'] >= 3.5:
            if 'Republicans' in node.condition:
                node.condition = node.condition.replace('Republicans','Party')
        elif item['aux']['conservatism'] <= 3.5:
            if 'Democrats' in node.condition:
                node.condition = node.condition.replace('Democrats', 'Party')
        
        if node.condition.split('\'')[3] not in item['aux'].keys():
            continue
        if 1 == node.run(item):
            rep1preds += int(bool(item['aux']['truthful'] == 1))
            length1 += 1
        else:
            rep0preds += int(bool(item['aux']['truthful']  == 0))
            length0 += 1
    return rep0preds, rep1preds, length0, length1

class Node:
    def __init__(self, conditionstr, left, right, show = False):
        self.condition = conditionstr
        self.left = left
        self.right = right
        self.show = show
    
    def run(self, item, show = False, **kwargs):
        self.show = show
        try:
            if 'aux' not in item.keys():
                item['aux'] = item
        except:
            tempitem = item
            item = {}
            item['item'] = tempitem
            item['aux'] = kwargs

        if item['aux']['conservatism'] >= 3.5:
            if 'Republicans' in self.condition:
                self.condition = self.condition.replace('Republicans','Party')
        elif item['aux']['conservatism'] <= 3.5:
            if 'Democrats' in self.condition:
                self.condition = self.condition.replace('Democrats', 'Party')

        if self.show:
            print(item['aux'])
            print(self.condition)

        if eval(self.condition):
            if isinstance(self.left,bool):
                return self.left
            return self.left.run(item)
        else:
            if isinstance(self.right,bool):
                return self.right
            return self.right.run(item)

    def getstring(self):
        a = ''
        if isinstance(self.left,bool):
            a = 'If ' + self.condition.split('\'')[3] + self.condition.split(']')[2] + ' then return ' + str(self.left) + ', else: ' 
            a += 'Return ' + str(self.right) if isinstance(self.right,bool) else self.right.getstring()
        else:
            a = 'If ' + self.condition.split('\'')[3] + self.condition.split(']')[2] + ' then return ' + str(self.right) + ', else: '
            a += 'Return ' + str(self.left) if isinstance(self.left,bool) else self.left.getstring()
        return a 