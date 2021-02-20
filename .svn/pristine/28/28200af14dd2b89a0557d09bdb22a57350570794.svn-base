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

finaltree = None

class FFTmax(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    componentKeys = []
    

    def __init__(self, name='Fast-Frugal-Tree-Max', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        SentimentAnalyzer.initialize()
        self.parameter = {}
        self.fft = None
        self.lastnode = None
        FFTmax.componentKeys = ['crt','ct','conservatism','panasPos','panasNeg','education', 'reaction_time','accimp','age','gender','Exciting_Democrats_Combined', 'Exciting_Republicans_Combined', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined', 'Importance_Democrats_Combined', 'Importance_Republicans_Combined', 'Likelihood_Democrats_Combined', 'Likelihood_Republicans_Combined', 'Partisanship_All_Combined', 'Partisanship_All_Partisan', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined','Sharing_Democrats_Combined', 'Sharing_Republicans_Combined', 'Worrying_Democrats_Combined','Worrying_Republicans_Combined', ] + [ a for a in ['Sent: negative_emotion', 'Sent: health', 'Sent: dispute', 'Sent: government', 'Sent: healing', 'Sent: military', 'Sent: fight', 'Sent: meeting', 'Sent: shape_and_size', 'Sent: power', 'Sent: terrorism', 'Sent: competing', 'Sent: office', 'Sent: money', 'Sent: aggression', 'Sent: wealthy', 'Sent: banking', 'Sent: kill', 'Sent: business', 'Sent: speaking', 'Sent: work', 'Sent: valuable', 'Sent: economics', 'Sent: payment', 'Sent: friends', 'Sent: giving', 'Sent: help', 'Sent: school', 'Sent: college', 'Sent: real_estate', 'Sent: reading', 'Sent: gain', 'Sent: science', 'Sent: negotiate', 'Sent: law', 'Sent: crime', 'Sent: stealing', 'Sent: strength'] if a in SentimentAnalyzer.relevant]#Keys.person + Keys.task 
        super().__init__(name, ['misinformation'], ['single-choice'])

    def pre_train(self, dataset):
        #Globally trains max FFT on data for all persons
        if FFTtool.MAX != None:
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
            for a in FFTmax.componentKeys:
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
        predictionQuality, predictionMargin = {'>crt': -0.5765148663906662, '<crt': -0.5835007682307056, '>conservatism': -0.5840998087605864, '<conservatism': -0.576172936416883, '>ct': -0.5755846761552449, '<ct': -0.6117876278616659, '>education': -0.6196549137284321, '<education': -0.576172936416883, '>gender': -0.5751785605274403, '<gender': -0.6629834254143646, '>accimp': -0.6245434623813002, '<accimp': -0.576257736867164, '>panasPos': -0.423766680755993, '<panasPos': -0.576172936416883, '>panasNeg': -0.423766680755993, '<panasNeg': -0.576172936416883, '>Exciting_Party_Combined': -0.6378066378066378, '<Exciting_Party_Combined': -0.6442516268980477, '>Familiarity_Party_Combined': -0.8807965084560829, '<Familiarity_Party_Combined': -0.576172936416883, '>Importance_Party_Combined': -0.9197396963123644, '<Importance_Party_Combined': -0.7952691680261011, '>Partisanship_All_Combined': -0.824295010845987, '<Partisanship_All_Combined': -0.7604868506846337, '>Partisanship_All_Partisan': -0.824295010845987, '<Partisanship_All_Partisan': -0.7213447326474424, '>Partisanship_Party_Combined': -0.9166666666666666, '<Partisanship_Party_Combined': -0.7128539213229766, '>Worrying_Party_Combined': -0.576172936416883, '<Worrying_Party_Combined': -0.7948717948717948}, {'>crt': 0.9611886396917466, '<crt': 0.6196903610860507, '>conservatism': 3.5312980367111253, '<conservatism': 0.0, '>ct': 5.0, '<ct': 4.227544258038495, '>education': 2.2412854095104953, '<education': 0.0, '>gender': 4.550795699820794, '<gender': 5.0, '>accimp': 1.7730933959075976, '<accimp': 5.0, '>panasPos': 0.0, '<panasPos': 0.0, '>panasNeg': 0.0, '<panasNeg': 0.0, '>Exciting_Party_Combined': 1.9445809043351385, '<Exciting_Party_Combined': 3.2554860566661565, '>Familiarity_Party_Combined': 2.2340655206116744, '<Familiarity_Party_Combined': 0.0, '>Importance_Party_Combined': 2.271886359375083, '<Importance_Party_Combined': 4.180200375448119, '>Partisanship_All_Combined': 4.098278058073108, '<Partisanship_All_Combined': 3.825921609404447, '>Partisanship_All_Partisan': 1.2084637950209594, '<Partisanship_All_Partisan': 0.8947571888022123, '>Partisanship_Party_Combined': 4.263257855394618, '<Partisanship_Party_Combined': 3.8420666367941125, '>Worrying_Party_Combined': 5.0, '<Worrying_Party_Combined': 3.937920683250629}
        #calculate order and direction of cues
        orderedConditions = []
        for a in sorted(predictionQuality.items(), key=lambda x: x[1], reverse=False):
            if a[0][1:] not in item.keys():
                continue
            if a[0][1:] not in [i[1:] for i in orderedConditions] and a[0][1:] in [a.replace('Democrats','Party').replace('Republicans','Party') for a in FFTmax.componentKeys]:
                orderedConditions.append(a[0])
        #assemble tree
        for sa in orderedConditions[:maxLength] if maxLength > 0 else orderedConditions:
            b = sa[1:]
            s = sa[0]
            cond = 'item[\'aux\'][\'' + b + '\'] ' + s + ' ' + str(predictionMargin[sa])
            newnode = Node(cond,True,False)
            rep0preds, rep1preds, length0, length1 = predictiveQuality(newnode, trialList)
            if self.fft == None:
                self.fft = newnode
                self.lastnode = self.fft
            else:
                if rep1preds/length1 >= rep0preds/length0:
                    self.lastnode.left = newnode
                    self.lastnode = self.lastnode.left
                elif rep1preds/length1 <= rep0preds/length0:
                    self.lastnode.right = newnode
                    self.lastnode = self.lastnode.right
        FFTtool.MAX = self.fft

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
        
        for a in FFTmax.componentKeys:
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
        return FFTtool.MAX.run(item, **kwargs, show=False)

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