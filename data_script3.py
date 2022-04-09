from typing import Collection, Counter

import pandas as pd 
import os 
import csv

dir_path = os.path.dirname(os.path.realpath(__file__))

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

counter = 0
dataTrials = []
columns= []
for number in ['3']:
    data_pd = pd.read_csv("data/st"+number+".csv", header=[0,1], encoding= 'unicode_escape') 

    data_pd.columns = [a[0]+'_' + a[1] for a in data_pd.columns]

    columns.extend([a for a in data_pd.columns])

    item_rows = []
    for tup in data_pd.columns:
        a = tup
        if '_' in a and ("Fake" in a and a.split("Fake")[-1] and len(a.split("Fake"))>1 and RepresentsInt(a.split("Fake")[-1][0]) or ("Real" in a and  a.split("Real")[-1] and len(a.split("Real"))>1 and RepresentsInt(a.split("Real")[-1][0]))):
            item_rows.append(a)
    print(item_rows)


    item_rows = sorted(list(set(item_rows)))
    data = data_pd.to_dict('index') 
    items = {}
    for it in item_rows:
        name = it.split('_')[0]
        feature = it[it.index('_'):]
        if name not in items.keys():
            items[name] = []
        items[name].append(feature)
        columns.append(feature)

    for index in range(len( list(data.keys()))):
        for item in items.keys():
            dataN = {}
            for feature in items[item]:
                dataN[feature] = data[index][item + feature]
            for feature in data[0].keys():
                if feature not in item_rows:
                    dataN[feature] = data[index][feature]
            dataN['task'] = str(item)
            dataN['id'] = index + (int(number) -1) * 1000
            dataN['trial'] = counter
            counter += 1
            
            dataTrials.append(dataN)
        print(index)
        
csv_columns = [a for a in data_pd.columns]

fields= list(set(columns)) + ['id', 'task', 'trial']
csv_file = "st_reformatted3.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=fields)
        writer.writeheader()
        for data in dataTrials:
            writer.writerow(data)
except IOError:
    print("I/O error")

with open(csv_file) as f:
    lines = f.readlines()

lines[0] = lines[0].replace('\'','').replace(')','').replace('(','').replace(', ','_')

with open(csv_file, "w") as f:
    f.writelines(lines)