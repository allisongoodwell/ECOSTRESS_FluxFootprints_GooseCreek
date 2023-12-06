#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:46:11 2023

Performance functions

@author: allison
"""

def KGE(observed, simulated):
    """
    The observed and simulated inputs should be arrays of the same length, representing the observed
    and simulated time series data.

    """
    observed_mean = observed.mean()
    simulated_mean = simulated.mean()
    covariance = ((observed - observed_mean) * (simulated - simulated_mean)).mean()
    observed_stdev = observed.std()
    simulated_stdev = simulated.std()
    correlation = covariance / (observed_stdev * simulated_stdev)
    alpha = simulated_stdev / observed_stdev
    beta = simulated_mean / observed_mean
    
    KGE = 1 - ((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
    
    return KGE, correlation, alpha, beta

def NSE(observed, predicted):
    """
    Calculates the Nash-Sutcliffe efficiency (NSE) between observed and predicted values.
    """
    
    observed_mean = observed.mean()
    #numerator = sum([(observed[i] - predicted[i])**2 for i in range(len(observed))])
    
    numerator = ((observed - predicted)**2).sum()

    #denominator = sum([(observed[i] - mean_obs)**2 for i in range(len(observed))])
    
    denominator = ((observed - observed_mean)**2).sum()
    NSE = 1 - numerator / denominator
    
    return NSE

def RMSE(observed, predicted):
    
    """
    Calculates the root mean squared error
    """
    
    resids_err = (observed - predicted)**2
    mean_resids_err = resids_err.mean()
    RMSE = mean_resids_err**0.5
    
    return RMSE
    