import pandas as pd 
import os 
import csv
from ccobra.data import Item
from models.ClassicalReasoning.cr import CR
from models.Heuristic.hr import RH
from models.Heuristic.hrlinear import RHlinear
from models.Heuristic.rt import RT
from models.Heuristic.fftmax import FFTmax
from models.Heuristic.fftzigzag import FFTzigzag
from models.LinearCombination.sent import LP
from models.LinearCombination.sentimentanalyzer import SentimentAnalyzer
from models.MotivatedReasoning.s2mr import S2MR
numberExceptions = 0

def rowToItem(row, n):
    global numberExceptions
    aux = {}
    task= row['task']
    for a in ['id','_Accurate','Familiarity_All_Combined', 'Familiarity_Task', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined', 'Conserv', 'crt','ct','conservatism','panasPos','panasNeg','education', 'reaction_time','accimp','age','gender','Exciting_Democrats_Combined', 'Exciting_Republicans_Combined','Importance_Democrats_Combined', 'Importance_Republicans_Combined', 'Likelihood_Democrats_Combined', 'Likelihood_Republicans_Combined', 'Partisanship_All_Combined', 'Partisanship_All_Combined', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined','Sharing_Democrats_Combined', 'Sharing_Republicans_Combined', 'Worrying_Democrats_Combined','Worrying_Republicans_Combined', ] + SentimentAnalyzer.relevant:
        if int(row['id']) < 900 : 
            if '_L' in row['task'] or '_C' in row['task']:
                pass
            elif float(row['_L']) == float(1):
                task = row['task'] + '_L'
            elif float(row['_C']) == float(1):
                task = row['task'] + '_C'
            else:
                return 'Incomplete'
            #if row['_N'] == '1':
            #    continue
            if a in row.keys():
                try:
                    aux[a] = float(row[a])
                except Exception as e:
                    numberExceptions +=1
                    print('Exception:', e, a)
                    print(numberExceptions) 
                    return 'Incomplete'
        else:
            try:
                if a in row.keys():
                    aux[a] = float(row[a])
                    #print(aux[a])
                    if a == 'reaction_time':
                        if str(aux[a]) == 'nan':
                            try:
                                aux[a] = float(row['_A_RT_2'])
                            except Exception as e:
                                print('Exception:', e)
                                numberExceptions +=1
                                print(numberExceptions)
                                return 'Incomplete'

                    if a == 'conservatism':
                        if str(aux[a]) == 'nan':
                            try:
                                aux[a] = float(row['Conserv'])
                            except Exception as e:
                                print('Exception:', e)
                                numberExceptions +=1
                                print(numberExceptions)
                                return 'Incomplete'
                    if a == 'crt':
                        if str(aux[a]) == 'nan':
                            try:
                                aux[a] = float(row['CRT_ACC'])
                            except Exception as e:
                                print('Exception:', e)
                                numberExceptions +=1
                                print(numberExceptions)
                                return 'Incomplete'
            except Exception as e:
                print(a)
                print(row[a])
                print(row)
                print(aux)
                print('Exception:', e)
                numberExceptions +=1
                print(numberExceptions)
                #raise(e)
                return 'Incomplete'
    aux['Partisanship_All_Combined'] = abs(partisanship_All[task])
    aux['Partisanship_Democrats_Combined'] = partisanship_Dem[task]
    aux['Partisanship_Republicans_Combined'] = partisanship_Rep[task]
    aux['Familiarity_Democrats_Combined'] = float(familiarity_Dem[task])
    aux['Familiarity_Republicans_Combined'] = float(familiarity_Rep[task])
    aux['Familiarity_All_Combined'] = float(familiarity_All[task])
    try:
        fam_dict = {
            0 : 0,
            1 : 2,
            2 : 1,
            3 : 0
        }
        aux['Familiarity_Task'] = float(fam_dict[int(aux['Familiarity_Task'])])
    except Exception as e:
        aux['Familiarity_Task'] = 'unknown'
    aux['truthful'] = int('R' in task)
    aux['aux'] = aux
    aux['task'] = task
    aux['binaryResponse'] = int(aux['_Accurate'])
    aux['accurate'] = 1 if (bool(aux['_Accurate']) and 'R' in task) or (not bool(aux['_Accurate']) and 'F' in task) else 0#bool(aux['_Accurate'])#
    return aux    


dir_path = os.path.dirname(os.path.realpath(__file__))

models = [RH, CR, RHlinear, RT, LP, S2MR, FFTmax, FFTzigzag]

dataTrials = pd.read_csv("st_reformatted.csv")
pretest = pd.read_csv("data/pretest.csv")

partisanship_All = {}
partisanship_Dem = {}
partisanship_Rep = {}
familiarity_All = {}
familiarity_Dem = {}
familiarity_Rep = {}
for index, row in pretest.iterrows():
    partisanship_All[row["ItemNum"]] = abs(float(row["PartisanDiff"]))
    partisanship_Dem[row["ItemNum"]] = float(row["P_Clinton"])
    partisanship_Rep[row["ItemNum"]] = float(row["P_Trump"])
    familiarity_All[row["ItemNum"]] = abs(float(row["FamiliarityDiff"]))
    familiarity_Dem[row["ItemNum"]] = float(row["F_Clinton"])
    familiarity_Rep[row["ItemNum"]] = float(row["F_Trump"])


used = ['crt','ClintonTrump','conservatism','Education', 'reaction_time','accimp','age','gender', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined','Partisanship_All_Combined', 'Partisanship_All_Partisan', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined',"Conserv","CRT_ACC",'_1', '_2' ,'_A_RT_2' ,'binaryResponse','_Accurate' , 'id','_RT_2' , "CRT", "_C", '_L', '_N', 'task','trial','choices','domain']


dataTrials = dataTrials.drop(columns=[a for a in list(dataTrials.columns) if (a not in used)])
dataTrials = dataTrials.rename(columns={
    "_1": "Familiarity_Task", 
    "CRT": "crt",
    "ClintonTrump": "ct",
    "Conserv": "conservatism",
    "_RT_2": "reaction_time",
    "Education": "education",
    })



persons = list(dataTrials['id'])
training_list = []
for ident in set(persons):
    training_list.append([rowToItem(row, numberExceptions) for index, row in dataTrials[dataTrials['id']==ident].iterrows() if rowToItem(row, numberExceptions) != 'Incomplete'])
    if not len(training_list) % 50:
        print(len(training_list))
non_mod_tl = training_list.copy()

for index, row in dataTrials.iterrows():
    aux = rowToItem(row, numberExceptions)
    if aux == 'Incomplete':
        continue
    #trial_item = aux['item'] 
    #prediction = model_inst.predictS(trial_item, kwargs = aux)


fields = list(aux.keys())
csv_file = "allitems.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=fields)
        writer.writeheader()
        for data in [item for sublist in non_mod_tl for item in sublist]:
            writer.writerow(data)
except IOError:
    print("I/O error")
