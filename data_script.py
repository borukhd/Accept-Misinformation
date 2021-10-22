from typing import Collection, Counter, Sequence

from pandas.io.parsers import count_empty_vals
import pandas as pd 
import os 
import csv

dir_path = os.path.dirname(os.path.realpath(__file__))


def hamming_distance(string1, string2): 
    # Start with a distance of zero, and count up
    if len(string1) != len(string2):
        return 100000
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance


def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

dataTrials = []
columns= []
counter = 0

for number in ['1','2']:
    data_pd = pd.read_csv("data/st"+number+".csv") 
    columns.extend(list(data_pd.columns))
    item_rows = [a for a in data_pd.columns if '_' in a and ("Fake" in a and a.split("Fake")[-1] and len(a.split("Fake"))>1 and RepresentsInt(a.split("Fake")[-1][0]) or ("Real" in a and  a.split("Real")[-1] and len(a.split("Real"))>1 and RepresentsInt(a.split("Real")[-1][0])))] 
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
        if number == '1':
            givenString = data[index]["DO_BR_FL_88"]  
            mover1 = 255
            mover2 = 510
            givenXString = givenString[:mover1] + 'X' + givenString[mover1:mover2] + 'X' + givenString[mover2:]
        
            recoveredString = givenXString.replace('lXF','l|F').replace('eXF','e|F').replace('eXR','e|R').replace('lXR','l|R')
            stringsequence = recoveredString.split('|')
            for sequenceCount in range(len(stringsequence)): 
                for correctSubStr in ['Liberal', 'Conservative', 'Neutral', 'Real', 'Fake']:
                    maximumCheckDistance = len(correctSubStr)
                    for substrstart in range(len(stringsequence[sequenceCount])):
                        #print(hamming_distance(stringsequence[sequenceCount][substrstart:substrstart+maximumCheckDistance], correctSubStr),
                        #    stringsequence[sequenceCount][substrstart:substrstart+maximumCheckDistance], correctSubStr)
                        if hamming_distance(stringsequence[sequenceCount][substrstart:substrstart+maximumCheckDistance], correctSubStr) == 1:
                            #print('distance 1', stringsequence[sequenceCount][substrstart:maximumCheckDistance], correctSubStr)
                            stringsequence[sequenceCount] = stringsequence[sequenceCount][:substrstart] + correctSubStr + stringsequence[sequenceCount][substrstart+maximumCheckDistance:]
                    stringsequence[sequenceCount] = stringsequence[sequenceCount].replace(' - ' + correctSubStr, '')
            recoveredString= stringsequence
            for a in range(len(recoveredString)):
                if 'X' in recoveredString[a]:
                    for typ in ['Real', 'Fake']:
                        if typ in recoveredString[a]:
                            for num in range(1,16):
                                if typ + str(num) not in recoveredString:
                                    recoveredString[a] = typ + str(num)

        else:
            givenString = data[index]["DO_BR_FL_167"]  
            mover1 = 255
            mover2 = mover1
            givenXString = givenString[:mover1] + 'X' + givenString[mover1:mover2] + givenString[mover2:]
        
            recoveredString = givenXString.replace('WXF','W|F').replace('WXR','W|R').replace('yXF','W|F').replace('yXR','W|R').replace('e0','e10').replace('l0','l10')
            stringsequence = recoveredString.split('|')
            if 24 != (len(stringsequence)):
                print(recoveredString)
            for sequenceCount in range(len(stringsequence)): 
                for a in ['Accuracy', 'NW', 'Real', 'Fake']:
                    correctSubStr = '' + a# + '|' + b
                    maximumCheckDistance = len(correctSubStr)
                    for substrstart in range(len(stringsequence[sequenceCount])):
                        #print(hamming_distance(stringsequence[sequenceCount][substrstart:substrstart+maximumCheckDistance], correctSubStr),
                        #    stringsequence[sequenceCount][substrstart:substrstart+maximumCheckDistance], correctSubStr)
                        if hamming_distance(stringsequence[sequenceCount][substrstart:substrstart+maximumCheckDistance], correctSubStr) == 1:
                            #print('distance 1', stringsequence[sequenceCount][substrstart:maximumCheckDistance], correctSubStr)
                            stringsequence[sequenceCount] = stringsequence[sequenceCount][:substrstart] + correctSubStr + stringsequence[sequenceCount][substrstart+maximumCheckDistance:]
                    stringsequence[sequenceCount] = stringsequence[sequenceCount].replace('_' + a, '').replace(' ','')
            recoveredString= stringsequence
            for a in range(len(recoveredString)):
                if 'X' in recoveredString[a]:
                    for typ in ['Real', 'Fake']:
                        if typ in recoveredString[a]:
                            for num in range(1,13):
                                if typ + str(num) not in recoveredString:
                                    recoveredString[a] = typ + str(num)

        """
        for a in range(len(recoveredString)):
            if 'X' in recoveredString[a]:
                print(recoveredString)
                continue
            if recoveredString.count(recoveredString[a]) != 1:
                print(recoveredString)
                continue
        """


        for item in items.keys():
            dataN = {}
            for feature in items[item]:
                dataN[feature] = data[index][item + feature]
            for feature in data[0].keys():
                if feature not in item_rows:
                    dataN[feature] = data[index][feature]
            
            dataN['task'] = str(item)
            dataN['id'] = index + (int(number) -1) * 1000
            #print(recoveredString)
            dataN['trial'] = counter + recoveredString.index(str(item))
            
            dataTrials.append(dataN)
        counter += len(items.keys())
        
        print(index)
        
csv_columns = [a for a in data_pd.columns]

fields= list(set(columns)) + ['id', 'task', 'trial']
csv_file = "st_reformatted.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=fields)
        writer.writeheader()
        for data in dataTrials:
            writer.writerow(data)
except IOError:
    print("I/O error")
