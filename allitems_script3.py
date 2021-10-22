import pandas as pd 
import os 
import csv
from models.LinearCombination.sentimentanalyzer3 import SentimentAnalyzer

from pandas.core.frame import DataFrame
from ccobra.data import Item

numberExceptions = 0

def rowToItem(row, n):
    global numberExceptions
    aux = {}
    task= row['task']
    for a in row.keys():
        if 'PANAS' in a:
            if str(row[a]) == 'nan':
                return 'Incomplete'
            else:
                panasPos = aux['panasPos'] if 'panasPos' in aux.keys() else 0
                panasNeg = aux['panasNeg'] if 'panasNeg' in aux.keys() else 0
                for b in ['NERVOUS','AFRAID','GUILTY','DISTRESSED','JITTERY','HOSTILE','ALERT','IRRITABLE','UPSET','SCARED','ASHAMED']:
                    if b in a:
                        panasNeg += int(float(row[a]))
                for b in ['ACTIVE','EXCITED','DETERMINED','ATTENTIVE','INTERESTED','PROUD','ENTHUSIASTIC','STRONG','INSPIRED']:
                    if b in a:
                        panasPos += int(float(row[a]))
                aux['panasNeg'] = panasNeg
                aux['panasPos'] = panasPos
        if 'conserv1' == a:
            if str(row[a]) == 'nan':
                return 'Incomplete'
            else:
                conservatism1 = row['conserv1']
        if 'conserv2' == a:
            if str(row[a]) == 'nan':
                return 'Incomplete'
            else:
                conservatism2 = row['conserv2']
    aux['conservatism'] = (conservatism1 + conservatism2)/2
    for a in ['id','_Accurate','Familiarity_All_Combined', 'Familiarity_Task', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined', 'Conserv', 'crt','ct','conservatism','panasPos','panasNeg','education', 'reaction_time','accimp','age','gender','Exciting_Democrats_Combined', 'Exciting_Republicans_Combined','Importance_Democrats_Combined', 'Importance_Republicans_Combined', 'Partisanship_All_Combined', 'Partisanship_All_Combined', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined', 'Worrying_Democrats_Combined','Worrying_Republicans_Combined', 'CRT1_1','CRT1_2','CRT1_3','CRT3_1','CRT3_2', 'CRT3_3'] + SentimentAnalyzer.relevant:
        try:
            if a in row.keys():
                if 'CRT' in a:
                    if str(row[a]) == 'nan':
                        return 'Incomplete'
                    else:
                        crt = aux['crt'] if 'crt' in aux.keys() else 0
                        if a == 'CRT1_1':
                            if str(int(row[a])).lower() == '8':
                                crt += float(1/6)
                        if a == 'CRT1_2':
                            if str(int(row[a])).lower()  == '10':
                                crt += float(1/6)
                        if a == 'CRT1_3':
                            if str(int(row[a])).lower() == '39':
                                crt += float(1/6)
                        if a == 'CRT3_1':
                            if str(int(row[a])).lower() == '2':
                                crt += float(1/6)
                        if a == 'CRT3_2':
                            if str(int(row[a])).lower() == '8':
                                crt += float(1/6)
                        if a == 'CRT3_3':
                            if row[a].lower() == 'emily':
                                crt += float(1/6)
                        aux['crt'] = crt
                else:
                    aux[a] = float(row[a])
                if a == 'reaction_time':
                    if str(aux[a]) == 'nan':
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

                if a == '_Accurate':
                    if str(aux[a]) == 'nan':
                        if str(row['_Accurate1']) == 'nan' or str(row['_Accurate1'])  == '':
                            return 'Incomplete'
                        try:
                            aux[a] = float(row['_Accurate1'])
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
    aux['Worrying_All_Combined'] = abs(Worrying_All[task])
    aux['Worrying_Democrats_Combined'] = Worrying_Dem[task]
    aux['Worrying_Republicans_Combined'] = Worrying_Rep[task]
    aux['Importance_Democrats_Combined'] = float(Importance_Dem[task])
    aux['Importance_Republicans_Combined'] = float(Importance_Rep[task])
    aux['Importance_All_Combined'] = float(Importance_All[task])
    aux['Exciting_Democrats_Combined'] = float(Exciting_Dem[task])
    aux['Exciting_Republicans_Combined'] = float(Exciting_Rep[task])
    aux['Exciting_All_Combined'] = float(Exciting_All[task])
    
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
    print(aux)
    return aux    


dir_path = os.path.dirname(os.path.realpath(__file__))


f = open('data/pretest3.csv', 'r')
fn = open('data/pretest3n.csv', 'w')
fn.close()
fn = open('data/pretest3n.csv', 'a')
lineno = 0
for line in f:
    if lineno > 1:
        fn.write(line)
    else:
        linelist = line.split(',')
        for index in range(2, len(linelist)):
            if linelist[index] == '':
                linelist[index] = linelist[index-1]
            elif linelist[index] == '\n':
                linelist[index] = linelist[index-1]
        linestring = ''
        for a in linelist:
            linestring += a + ','
        fn.write(linestring[:-1] + '\n')


    lineno += 1
fn.close()
pretest = pd.read_csv('data/pretest3n.csv', header=[0,1,2])
dataTrials = pd.read_csv('st_reformatted3.csv')

partisanship_All = {}
partisanship_Dem = {}
partisanship_Rep = {}
familiarity_All = {}
familiarity_Dem = {}
familiarity_Rep = {}
Worrying_All = {}
Worrying_Dem = {}
Worrying_Rep = {}
Importance_All = {}
Importance_Dem = {}
Importance_Rep = {}
Exciting_All = {}
Exciting_Dem = {}
Exciting_Rep = {}
for index, row in pretest.iterrows():
    text = row['Unnamed: 2_level_0', 'Unnamed: 2_level_1', 'Headline']
    task = None
    for key in dataTrials.iloc[0].keys():
        if '_' not in key: 
            continue
        k = key.split('_')[:2]
        a , b = k
        b = b.replace('\'','').replace('\"','')
        if 'R' != b[0] and 'F' != b[0] or len(b) < 4:
            continue
        if ' ' != b[2] and ' ' != b[3]:
            continue
        part = b.lower().replace('-','')[5:-2]
        full = text.replace('\'','').replace('\"','').lower().replace('-','')
        if part in full:
            task = a.replace('R','Real').replace('F','Fake')
            break
    if task == None:
        continue
    partisanship_All[task] = abs(float(row['Partisanship', 'All', 'Partisan']))
    partisanship_Dem[task] = float(row['Partisanship', 'Democrats', 'Combined'])
    partisanship_Rep[task] = float(row['Partisanship', 'Republicans', 'Combined'])
    familiarity_Dem[task] = float(row['Familiarity', 'Democrats', 'Combined'])
    familiarity_Rep[task] = float(row['Familiarity', 'Republicans', 'Combined'])
    familiarity_All[task] = abs(familiarity_Rep[task] + familiarity_Dem[task])/2
    Worrying_Dem[task] = float(row['Worrying', 'Democrats', 'Combined'])
    Worrying_Rep[task] = float(row['Worrying', 'Republicans', 'Combined'])
    Worrying_All[task] = abs(Worrying_Rep[task] + Worrying_Dem[task])/2
    Importance_Dem[task] = float(row['Importance', 'Democrats', 'Combined'])
    Importance_Rep[task] = float(row['Importance', 'Republicans', 'Combined'])
    Importance_All[task] = abs(Importance_Rep[task] + Importance_Dem[task])/2
    Exciting_Dem[task] = float(row['Exciting', 'Democrats', 'Combined'])
    Exciting_Rep[task] = float(row['Exciting', 'Republicans', 'Combined'])
    Exciting_All[task] = abs(Exciting_Rep[task] + Exciting_Dem[task])/2


#used = ['crt','ClintonTrump','conservatism','Education', 'reaction_time','accimp','age','gender', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined','Partisanship_All_Combined', 'Partisanship_All_Partisan', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined','Conserv','CRT_ACC','_1', '_2' ,'_A_RT_2' ,'binaryResponse','_Accurate' , 'id','_RT_2' , 'CRT', '_C', '_L', '_N', 'task','trial','choices','domain']


dataTrials = dataTrials.drop(columns=[a for a in list(dataTrials.columns) if (a in ['used'] or 'sharing' in a)])
#print([a for a in dataTrials.columns])

replacedict = {
    '_1': 'Familiarity_Task', 
    'CRT': 'crt',
    'ClintonTrump': 'ct',
    'Conserv': 'conservatism',
    '_RT_2_Timing-Last Click': 'reaction_time',
    '_To the best of your knowledge_is the claim in the above headline accurate?': '_Accurate',
    '_To the best of your knowledge_is the claim in the above headline accurate?.1': '_Accurate1',
    'CRT3_3_Emilys father has three daughters. The first two are named April and May. What is the third daug...': 'CRT3_3',
    'CRT3_2_A farmer had 15 sheep and all but 8 died. How many are left?': 'CRT3_2',
    'CRT3_1_If youre running a race and you pass the person in second place_what place are you in? [Please...': 'CRT3_1',
    'CRT1_3_On a loaf of bread_there is a patch of mold. Every day_the patch doubles in size. If it takes 4...': 'CRT1_3',
    'CRT1_2_If it takes 10 seconds for 10 printers to print out 10 pages of paper_how many seconds will it t...': 'CRT1_2',
    'CRT1_1_The ages of Mark and Adam add up to 28 years total. Mark is 20 years older than Adam. How many ye...': 'CRT1_1',
    'Economic_Conserv_On economic issues I am:': 'conserv1',
    'Social_Conserv_On social issues I am:': 'conserv2',
    'POTUS2016_Who did you vote for in the 2016 Presidential Election? Reminder: This survey is anonymous.': 'ct',
    'Education_What is the highest level of school you have completed or the highest degree you have received? ': 'education',
    'Gender_What is your gender?': 'gender'
    }
#for a in dataTrials.columns:
#    if '_RT_2' in a:


dataTrials = dataTrials.rename(columns=replacedict)



persons = list(dataTrials['id'])
training_list = []
for ident in set(persons):
    training_list.append([rowToItem(row, numberExceptions) for index, row in dataTrials[dataTrials['id']==ident].iterrows() if rowToItem(row, numberExceptions) != 'Incomplete'])
    if not len(training_list) % 50:
        print(len(training_list))
non_mod_tl = training_list.copy()

for index, row in dataTrials.iterrows():
    aux1 = rowToItem(row, numberExceptions)
    if aux1 == 'Incomplete':
        continue
    else:
        aux = aux1
        break
    #trial_item = aux['item'] 
    #prediction = model_inst.predictS(trial_item, kwargs = aux)


fields = list(aux.keys())
csv_file = 'allitems3.csv'
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=fields)
        writer.writeheader()
        for data in [item for sublist in non_mod_tl for item in sublist]:
            writer.writerow(data)
except IOError:
    print('I/O error')
