import collections
from models.LinearCombination.optimizationParameters import OptPars
from models.LinearCombination.sentimentanalyzer import SentimentAnalyzer as SA12
from models.LinearCombination.sentimentanalyzer3 import SentimentAnalyzer as SA3
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")



# Generate a large random dataset
def do(filename, onlyModels = False):
    models = ['CR&time', 'ClassicReas', 'FFT-Max', 'FFT-ZigZag(Z+)', 'HeurRecogn', 'HeurRecogn-lin.', 'S2MR', 'SentimentAnalysis', 'WMSupprByMood']

    inputfilename = filename

    d = pd.read_csv(inputfilename)
    for a in d.keys():
        a_new = a.replace('_Combined','')
        a_new = a_new.replace('Party','Own')
        d = d.rename(columns={a: a_new})
    d = d.rename(columns={'truthful': 'real news'})
    d = d.rename(columns={'binaryResponse': '\"Accept\"'})
    d = d.rename(columns={'WMSuppressionByMood': 'WMSupprByMood'})

    if '3' not in inputfilename:
        SA12.initialize()
        rel = SA12.relevant
        outname = 'corr12'
    else:
        SA3.initialize()
        rel = SA3.relevant
        outname = 'corr3'
        d = d.drop(columns='Familiarity_Own')

    for a in d.columns:
        if a in rel:
            d = d.drop(columns=a)
    d = d.drop(columns='sequence')
    d = d.drop(columns='id')
    d = d.drop(columns='ct')
    
    if onlyModels:
        for a in d.columns:
            if a not in models + ['real news', '\"Accept\"']:
                d = d.drop(columns=a)


    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    #mask = np.tril(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 7))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, #mask=mask, 
                cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.subplots_adjust(left=0.22, bottom=0.22, right=1, top=0.99, wspace=0.2, hspace=0.2)
    
    plt.show()
    plt.savefig(outname)

for a in ['modeloutputs12.csv', 'modeloutputs3.csv']:
    do(a, True)