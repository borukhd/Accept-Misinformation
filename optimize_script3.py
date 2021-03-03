from models.Heuristic.surp import SURP
from random import betavariate
from models.LinearCombination.optimizationParameters import OptPars
import pandas as pd 
import os 
from ccobra.data import Item
from models.ClassicalReasoning.cr import CR
from models.Heuristic.hr import RH
from models.Heuristic.hrlinear import RHlinear
from models.Heuristic.rt import RT
from models.Heuristic.fftmax3 import FFTmax
from models.Heuristic.fftzigzag3 import FFTzigzag
from models.LinearCombination.sent3 import LP
from models.LinearCombination.sentimentanalyzer3 import SentimentAnalyzer
from models.MotivatedReasoning.s2mr import S2MR
from models.hybrid3 import Hybrid
from numpy.core.numeric import correlate
from numpy.lib.function_base import average

import numpy as np

from numpy.core.fromnumeric import mean

from numpy.lib.twodim_base import tri
from scipy.optimize.optimize import brute
numberExceptions = 0



dir_path = os.path.dirname(os.path.realpath(__file__))

models = [SURP, CR, FFTmax, FFTzigzag, S2MR, LP, RT, RH, RHlinear, Hybrid,  ]## 

dataTrials = pd.read_csv("allitems3.csv")

sentim = SentimentAnalyzer()
sentim.initialize()

persons = [int(a) for a in dataTrials['id']]
training_list = []
for ident in set(persons):
    data_for_pers = dataTrials[ident == dataTrials['id']]
    pers_rows = []
    for index, row in data_for_pers.iterrows():
        aux = {a : row[a] for a in row.keys()}
        for a in list(aux.keys()).copy():
            if a in ['Familiarity_All_Combined', 'Partisanship_All_Combined', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined', 'truthful', 'crt', 'conservatism', 'ct','education', 'reaction_time', 'binaryResponse', 'accurate', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined', 'Exciting_Democrats_Combined', 'Exciting_Republicans_Combined','Importance_Democrats_Combined', 'Importance_Republicans_Combined', 'Worrying_Democrats_Combined','Worrying_Republicans_Combined', 'panasPos', 'panasNeg']:
                if a not in aux.keys():
                    #print('2',row['id'],row['conservatism'], a, aux)
                    continue
                aux[a] = float(aux[a])
                if row['conservatism'] >= 2.9:
                    if 'Republicans' in a:
                        aux[a.replace('Republicans', 'Party')] = aux[a]
                        aux.pop(a,None)
                        aux.pop(a.replace('Republicans','Democrats'))
                elif row['conservatism'] <= 2.9:
                    if 'Democrats' in a:
                        aux[a.replace('Democrats', 'Party')] = aux[a]
                        aux.pop(a,None)
                        aux.pop(a.replace('Democrats','Republicans'))
        """
        """
        if aux['Familiarity_Task'] != 'unknown':
            if int(float(aux['Familiarity_Task'])) == 0:
                aux['Familiarity_Party_Combined'] = float(1)
            if int(float(aux['Familiarity_Task'])) == 1:
                aux['Familiarity_Party_Combined'] = float(2.5)
            if int(float(aux['Familiarity_Task'])) == 2:
                aux['Familiarity_Party_Combined'] = float(4)
        else: 
            pass
            #print(aux)
        aux['item'] = Item(ident, "misinformation", row['task'], 'single-choice', '0|1', index)
        aux['aux'] = aux
        aux['trial'] = aux['item'].sequence_number
        aux['id'] = int(aux['id'])
        for a in sentim.relevant:
            aux[a] = sentim.an_dict(aux['task'])[a]
        pers_rows.append(aux)
    training_list.append(pers_rows)
    if not len(training_list) % 500:
        print(len(training_list))
    if len(training_list) < -50:
        break
#"""
f = open("models/pars_other.txt", "w")
f.write("{}")
f.close()
#"""
parameterDict = open('models/pars_other.txt', 'r').read()
OptPars.parsPerPers = eval(parameterDict)

print('Total trials:', sum([len(a) for a in training_list]))
trial_resp = {}
for model in models:
    model_inst = model()
    trial_resp[model_inst.name] = {}
    print(model_inst.name, 'training started')
    model_inst.pre_train(training_list.copy())
    print(model_inst.name, model_inst.parameter)
    """try:
        for l in range(model_inst.fft.length):
            average_accuracy = []
            model_tendency = []
            person_tendency = []
            average_pers_accuracy = []
            print('length:',l)
            for triallist in training_list.copy():
                if len(triallist) != 20 and len(triallist) != 24:
                    continue
                model_inst.pre_train_person(triallist.copy())
                pers_accuracy = []
                for trial in triallist:
                    trial['length'] = l
                    prediction_m = model_inst.predictS(trial['item'], kwargs=trial)
                    #prediction = 1 if trial['Familiarity_Party_Combined'] > 2.5130623903980265 else 0
                    #print(prediction-prediction_m)
                    bounded_prediction = min(1, max(0, prediction_m))
                    person_response = float(trial['binaryResponse'])
                    #print(bounded_prediction)
                    correctness = 1-abs(person_response-float(bounded_prediction))
                    average_accuracy.append(correctness)
                    pers_accuracy.append(correctness)
                    person_tendency.append(person_response)
                    model_tendency.append(bounded_prediction)
                    trial_resp[model_inst.name][trial['trial']] = bounded_prediction
                if not trial['id'] % 500:
                    print(trial['id'],'pers done')
                average_pers_accuracy.append(np.mean(pers_accuracy))
            print('average model accuracy', np.mean(average_pers_accuracy))
            print('median model accuracy', np.median(average_pers_accuracy))
            print('person tendency', np.mean(person_tendency))
            print('model tendency', np.mean(model_tendency))
            print(model_inst.name, np.mean(average_accuracy))
            print(len(model_tendency), ' trials in persons:', len(average_pers_accuracy))
            
    except:"""
    average_accuracy = []
    model_tendency = []
    person_tendency = []
    average_pers_accuracy = []
    tl = training_list.copy()
    for triallist in tl:
        if triallist[0]['id'] < 900:
            if len(triallist) != 20: 
                continue
        elif triallist[0]['id'] >= 2000:
            if len(triallist) != 35:
                continue
        else:
            if len(triallist) != 24:
                continue
        model_inst.pre_train_person(triallist.copy())
        pers_accuracy = []
        for trial in triallist:
            prediction_m = model_inst.predictS(trial['item'], kwargs=trial)
            #prediction = 1 if trial['Familiarity_Party_Combined'] > 2.5130623903980265 else 0
            #print(prediction-prediction_m)
            bounded_prediction = min(1, max(0, prediction_m))
            person_response = float(trial['binaryResponse'])
            #print(bounded_prediction)
            correctness = 1-abs(person_response-float(bounded_prediction))
            average_accuracy.append(correctness)
            pers_accuracy.append(correctness)
            person_tendency.append(person_response)
            model_tendency.append(bounded_prediction)

            trial_resp[model_inst.name][trial['trial']] = bounded_prediction
        if not trial['id'] % 500:
            print(trial['id'],'pers done')
        average_pers_accuracy.append(np.mean(pers_accuracy))
    print('average model accuracy', np.mean(average_pers_accuracy))
    print('median model accuracy', np.median(average_pers_accuracy))
    print('person tendency', np.mean(person_tendency))
    print('model tendency', np.mean(model_tendency))
    print(model_inst.name, np.mean(average_accuracy))
    print(len(model_tendency), ' trials in persons:', len(average_pers_accuracy))
        
    try:
        f = open("models/pars_other.txt", "w")
        f.write(str(OptPars.parsPerPers))
        f.close()
    except IOError:
        print("I/O error")


f = open("modeloutputs3.csv", "w")
firstline = ''
firstline_list = []
allfeats = []
for aux in OptPars.parsPerPers.values():
    allfeats = [a for a in allfeats + list(aux.keys())]
features = ['id', 'task', 'sequence', 'choices', 'response_type', 'domain', 'response', 'truthful', 'binaryResponse', 'Familiarity_All_Combined', 'Familiarity_Party_Combined', 'Familiarity_Task', 'Partisanship_All_Combined', 'Partisanship_Party_Combined', 'Exciting_Party_Combined','Worrying_Party_Combined','Importance_Party_Combined', 'conservatism','panasPos','panasNeg', 'crt', 'ct', 'education', 'reaction_time'] + SentimentAnalyzer.relevant
for model in sorted(OptPars.parsPerPers.keys()):
    firstline_list.append(model)
    didonce = False
    for par_set in OptPars.parsPerPers[model].keys():
        break
        if par_set == 'global':
            try:
                a, b = OptPars.parsPerPers[model][par_set]
                OptPars.parsPerPers[model][par_set] = {'quality' : a, 'margin' : b}
            except:
                pass
            for par_name in sorted(OptPars.parsPerPers[model][par_set].keys()):
                firstline_list.append(model + '_parameterValue_' + str(par_name))
        elif not didonce:
            didonce = True
            for par_name in [a.split('\'')[1] for a in sorted(OptPars.parsPerPers[model][par_set])]:
                new_item = model + '_parameterValue_' + str(par_name)
                firstline_list.append(new_item)

for feature in sorted(features):
    firstline_list.append(feature)
for a in firstline_list:
    firstline += a + ','
f.write(firstline[:-1] + '\n')
for trial in [a for b in training_list for a in b]:
    trial['sequence'] = trial['trial']
    trial['choices'] = 'Accept|Reject'
    trial['response_type'] = 'single-choice'
    trial['domain'] = 'misinformation'
    trial['response'] = 'Accept' if bool(int(float(trial['binaryResponse']))) else 'Reject'
    if trial['trial'] not in [a for a in trial_resp.values()][0].keys():
        continue
    linelist = []
    for model in sorted(OptPars.parsPerPers.keys()):
        linelist.append(trial_resp[model][trial['trial']])
        for par_set in OptPars.parsPerPers[model].keys():
            break
            if par_set == 'global':
                if type(OptPars.parsPerPers[model][par_set]) is not dict:
                    a, b = OptPars.parsPerPers[model][par_set]
                    OptPars.parsPerPers[model][par_set] = {'<' : str(a).replace(',',';'), '>' : str(b).replace(',',';')}
                    print(OptPars.parsPerPers[model][par_set])
                for par_name in sorted(OptPars.parsPerPers[model][par_set].keys()):
                    linelist.append(OptPars.parsPerPers[model][par_set][par_name])
            elif par_set == trial['id']:
                for a in sorted(OptPars.parsPerPers[model][par_set]):
                    linelist.append(a.split(' ')[-1] )

    for feature in sorted(features):
        feat = '' if feature not in trial.keys() else trial[feature]
        linelist.append(feat)
    
    line = ''
    for a in linelist:
        line += str(a).replace(',',';') + ','
    f.write(line[:-1] + '\n')
f.close()