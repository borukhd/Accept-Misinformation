
from typing import Sequence
from pandas.core import base
from numpy import mean, std
from math import floor, ceil
from scipy.stats import median_absolute_deviation
from numpy.lib.function_base import median
from scipy.stats.stats import ModeResult

models = ['CR&time', 'ClassicReas', 'FFT-Max', 'FFT-ZigZag(Z+)', 'HeurRecogn', 'HeurRecogn-lin.', 'S2MR', 'SentimentAnalysis', 'HeurRecogn&RT', 'S2MR&RT', 'SentimentAnalysis&RT']


def itemsList(source, models):
    linecount = 0
    indcount = 0
    line1 = True
    lines = open(source)
    ind = {}


    perPerson = {}

    baseline = ['BaselineRandom', 'CorrectReply','AlwaysReject']
    if '3' in source:
        models = models + ['WMSupprByMood']

        if 'S2MR&RT' in models:
            models = models + ['WMSupprByMood&RT']

    for line in lines:
        listLine = line.replace('\r','').replace('\n','').split(',')
        if line1:
            line1 = False
            for key in listLine:
                ind[key] = indcount
                indcount += 1
            continue
        linecount += 1
        
        person = listLine[ind['id']]
        if person not in perPerson.keys():
            perPerson[person] = {}

        for model in models:
            if model not in perPerson[person].keys():
                perPerson[person][model] = []
            #print(ind.keys())
            perPerson[person][model].append(1-abs(float((listLine[ind['binaryResponse']])) - float(listLine[ind[model]])))

    maxModels = {}
    maxPerfs = {}
    for pers in perPerson.keys():
        maxperf, maxmodel = 0, None
        for model in perPerson[pers].keys():
            if mean(perPerson[pers][model]) > maxperf:
                maxperf, maxmodel = mean(perPerson[pers][model]), model
        maxPerfs[pers] = maxperf
        maxModels[pers] = maxmodel
    
    numberOfModelAsMax = {}
    for pers in maxModels.keys():
        if maxModels[pers] not in numberOfModelAsMax.keys():
            numberOfModelAsMax[maxModels[pers]] = 0
        numberOfModelAsMax[maxModels[pers]] += 1
    
    allPersPerfList = [maxPerfs[a] for a in maxPerfs.keys()]

    #print(numberOfModelAsMax)
    percOfModelsAsMax = {}
    for a in numberOfModelAsMax.keys():
        percOfModelsAsMax[a] = float(numberOfModelAsMax[a]) / sum(numberOfModelAsMax[a] for a in numberOfModelAsMax.keys())

    print(sorted([(v,k) for k,v in percOfModelsAsMax.items()], reverse=True)[:8])
    print('mean', round(mean(allPersPerfList), 2), 'median', round(median(allPersPerfList), 2),'MAD', round(median_absolute_deviation(allPersPerfList), 2))


def itemsList2models(source, models):
    linecount = 0
    indcount = 0
    line1 = True
    lines = open(source)
    ind = {}


    perPerson = {}

    baseline = ['BaselineRandom', 'CorrectReply','AlwaysReject']

    if '3' in source:
        models = models + ['WMSupprByMood']
        if 'S2MR&RT' in models:
            models = models + ['WMSupprByMood&RT']

    for line in lines:
        listLine = line.replace('\r','').replace('\n','').split(',')
        if line1:
            line1 = False
            for key in listLine:
                ind[key] = indcount
                indcount += 1
            continue
        linecount += 1

        person = listLine[ind['id']]
        if person not in perPerson.keys():
            perPerson[person] = {}


        for model in models:# ind.keys():
            if model not in perPerson[person].keys():
                perPerson[person][model] = []
            perPerson[person][model].append(1-abs(float(listLine[ind['binaryResponse']]) - float(listLine[ind[model]])))

        for model in baseline:
            if model not in perPerson[person].keys():
                perPerson[person][model] = []
            if model == 'BaselineRandom':
                perPerson[person][model].append(1-abs(float((listLine[ind['binaryResponse']])) - float(0.5)))
            if model == 'CorrectReply':
                perPerson[person][model].append(1-abs(float((listLine[ind['binaryResponse']])) - float('T' in listLine[ind['truthful']])))
            if model == 'AlwaysReject':
                perPerson[person][model].append(1-abs(float((listLine[ind['binaryResponse']])) - float(0)))

    pairs = []
    pairdone = []

    for model1 in models:
            for model2 in models:
                    if model1 == model2:
                        continue
                    if model1 + model2 in pairdone or model2 + model1 in pairdone:
                        continue
                    pairdone.append(model1 + model2)
                    maxModels = {}
                    maxPerfs = {}
                    for pers in perPerson.keys():
                        maxperf, maxmodel = 0, None
                        for model in [model1,model2]:
                            if model not in perPerson[pers].keys():
                                continue
                            if mean(perPerson[pers][model]) > maxperf:
                                maxperf, maxmodel = mean(perPerson[pers][model]), model
                        maxPerfs[pers] = maxperf
                        maxModels[pers] = maxmodel
                    
                    numberOfModelAsMax = {}
                    for pers in maxModels.keys():
                        if maxModels[pers] not in numberOfModelAsMax.keys():
                            numberOfModelAsMax[maxModels[pers]] = 0
                        numberOfModelAsMax[maxModels[pers]] += 1
                    
                    allPersPerfList = [maxPerfs[a] for a in maxPerfs.keys()]
                    if len([a for a in numberOfModelAsMax.keys() if a != None]) < 2:
                        continue
                    pairs.append((numberOfModelAsMax,mean(allPersPerfList), std(allPersPerfList), median(allPersPerfList), median_absolute_deviation(allPersPerfList)))
    pairs.sort(key=order)
    print(pairs[:5])

    for model in sorted(a for a in models + baseline):
        meanresperpers = [mean(perPerson[pers][model]) for pers in perPerson.keys()]
        print(model, ':', int(20-len(model))*' ', 'mean', round(mean(meanresperpers), 2), 'median', round(median(meanresperpers), 2),'MAD', round(median_absolute_deviation(meanresperpers), 2))

def order(itemOfList):
    dictionary, meanV, stdV, medainV, madV = itemOfList
    return -meanV

def listMostPopularByParts(source, models,  numberDiv = 4):
    linecount = 0
    indcount = 0
    line1 = True
    firstperson = True
    firstpersonName = ''
    taskcounter = -1
    lines = open(source)
    ind = {}


    perPerson = {}

    baseline = ['BaselineRandom', 'CorrectReply','AlwaysReject']
    if '3' in source:
        models = models + ['WMSupprByMood']
        if 'S2MR&RT' in models:
            models = models + ['WMSupprByMood&RT']

    for line in lines:
        listLine = line.replace('\r','').replace('\n','').split(',')
        if line1:
            line1 = False
            for key in listLine:
                ind[key] = indcount
                indcount += 1
            continue
        linecount += 1
        person = listLine[ind['id']]
        if firstperson:
            firstpersonName = person
            firstperson = False
        taskcounter += 1.0
        if person != firstpersonName:
            break
    print("taskcounter",taskcounter)
    for line in lines:
        listLine = line.replace('\r','').replace('\n','').split(',')
        linecount += 1
        person = listLine[ind['id']]

        if person not in perPerson.keys():
            perPerson[person] = {}
            for a in range(1,numberDiv+1):
                perPerson[person][a] = {}

        for model in models:
            for i in range(1,numberDiv+1):
                if model not in perPerson[person][i].keys():
                    perPerson[person][i][model] = []
            perPerson[person][1+floor(numberDiv*(int(listLine[ind['sequence']])%taskcounter)/taskcounter)][model].append(1-abs(float((listLine[ind['binaryResponse']])) - float(listLine[ind[model]])))

    maxModels = {}
    maxPerfs = {}
    numberOfModelAsMax = {}
    percOfModelsAsMax = {}
    allPersPerfList = {}
    for i in range(1,numberDiv+1):#
        maxPerfs[i] = {}
        maxModels[i] = {}
        for pers in perPerson.keys():
            maxperf, maxmodel = 0, None
            for model in perPerson[pers][i].keys():
                if mean(perPerson[pers][i][model]) > maxperf:
                    maxperf, maxmodel = mean(perPerson[pers][i][model]), model
            maxPerfs[i][pers] = maxperf
            maxModels[i][pers] = maxmodel

        numberOfModelAsMax[i] = {}
        for pers in maxModels[i].keys():
            if maxModels[i][pers] not in numberOfModelAsMax[i].keys():
                numberOfModelAsMax[i][maxModels[i][pers]] = 0
            numberOfModelAsMax[i][maxModels[i][pers]] += 1
    
        allPersPerfList[i] = [maxPerfs[i][a] for a in maxPerfs[i].keys()]

        #print(numberOfModelAsMax)
        percOfModelsAsMax[i] = {}
        for a in numberOfModelAsMax[i].keys():
            percOfModelsAsMax[i][a]  = float(numberOfModelAsMax[i][a]) / sum(numberOfModelAsMax[i][a] for a in numberOfModelAsMax[i].keys())
        print(sorted([(round(v*100, 3),k) for k,v in percOfModelsAsMax[i].items()], reverse=True)[:5])

    print('mean', round(mean(allPersPerfList[i]), 2), 'median', round(median(allPersPerfList[i]), 2),'MAD', round(median_absolute_deviation(allPersPerfList[i]), 2))

def listNumberChangesIn12and23Thirds(source):
    return



for source in ['modeloutputs12.csv','modeloutputs3.csv']:
    print(source, ':')
    itemsList(source, models )
    itemsList2models(source, models)

for source in ['modeloutputs12.csv']:
    listMostPopularByParts(source,models, 3)
    listNumberChangesIn12and23Thirds(source)
