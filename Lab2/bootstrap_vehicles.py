# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:59:36 2019

@author: cl18417
"""

import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np


def bootstrap(sample, sample_size, iterations):
    new_samples = np.empty([iterations, sample_size])
    means = np.empty([iterations, 1])
    
    for i in range(iterations):
        new_samples[i] = np.random.choice(sample, size=sample_size)
        means[i] = np.mean(new_samples[i])
        
    data_mean = np.mean(means)
    
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    
    return data_mean, lower, upper

if __name__ == "__main__":
    df = pd.read_csv('./vehicles.csv')

    dataCurrent = df.values.T[0]
    
    boot = bootstrap(dataCurrent, dataCurrent.shape[0], 15000)
    print('Current mean:', boot[0])
    print('Current lower:', boot[1])
    print('Current upper:', boot[2])
    
    dataNew = df.values.T[1]
    dataNew = dataNew[~np.isnan(dataNew)]
    
    boot = bootstrap(dataNew, dataNew.shape[0], 15000)
    print('\nNew mean:', boot[0])
    print('New lower:', boot[1])
    print('New upper:', boot[2])