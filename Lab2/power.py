# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:36:53 2019

@author: cl18417
"""

import numpy as np

def power(sample1, sample2, reps, size, alpha):
    obs = sample2.mean() - sample1.mean()
    countP = 0
    
    for i in range(reps):
        count = 0
        
        for j in range(reps):
            newSample = np.concatenate(sample1, sample2)
            
            newSample1 = np.random.choice(newSample, size=size)
            newSample2 = np.random.choice(newSample, size=size)
        
            perm = newSample2.mean() - newSample1.mean()
        
            if perm > obs:
                count += 1
                
        pvalue = count/reps
        if pvalue < 1 - alpha:
            countP += 1
    
    print(countP/reps)