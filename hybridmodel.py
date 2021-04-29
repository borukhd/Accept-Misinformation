
from pandas.core import base
from numpy import mean, std
from math import floor, ceil
from scipy.stats import median_absolute_deviation
from numpy.lib.function_base import median

def itemsList(source):
    linecount = 0
    indcount = 0
    line1 = True
    lines = open(source)
    ind = {}


    perPerson = {}

    baseline = ['BaselineRandom', 'CorrectReply','AlwaysReject']
    models = ['CR&time', 'ClassicReas', 'FFT-Max', 'FFT-ZigZag(Z+)', 'HeurRecogn', 'HeurRecogn-lin.', 'S2MR', 'SentimentAnalysis']
    if '3' in source:
        models = models + ['WMSupprByMood']

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

    print(percOfModelsAsMax)
    print('mean', round(mean(allPersPerfList), 2), 'median', round(median(allPersPerfList), 2),'MAD', round(median_absolute_deviation(allPersPerfList), 2))


def itemsList2models(source):
    linecount = 0
    indcount = 0
    line1 = True
    lines = open(source)
    ind = {}


    perPerson = {}

    baseline = ['BaselineRandom', 'CorrectReply','AlwaysReject']

    models = ['CR&time', 'ClassicReas', 'FFT-Max', 'FFT-ZigZag(Z+)', 'HeurRecogn', 'HeurRecogn-lin.', 'S2MR', 'SentimentAnalysis']
    if '3' in source:
        models = models + ['WMSupprByMood']

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

    for model in models + baseline:
        meanresperpers = [mean(perPerson[pers][model]) for pers in perPerson.keys()]
        print(model, ':', int(20-len(model))*' ', 'mean', round(mean(meanresperpers), 2), 'median', round(median(meanresperpers), 2),'MAD', round(median_absolute_deviation(meanresperpers), 2))

def order(itemOfList):
    dictionary, meanV, stdV, medainV, madV = itemOfList
    return -meanV



for source in ['modeloutputs12.csv','modeloutputs3.csv']:
    print(source, ':')
    itemsList(source)
    itemsList2models(source)