#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 23:40:45 2022

@author: mohamedazizbhouri
"""

import numpy as np

is_input = 0 # 0 or 1
case_var = 'moist' # 'moist' or 'heat' if is_input == 0
sim = 'CAM' # 'CAM' or 'SPCAM'

if sim == 'CAM':
    if is_input == 1:
        X = np.load('data_CAM5_8K/all_inputs.npy')
        X = np.array(X,dtype=np.float64)
        
        mu_X_CAM5, sigma_X_CAM5 = [], []
        # we do not use the vectorial format of computing mean and std on purpose
        # in order to avoid the risk of precision error given the dataset size
        for i in range(X.shape[1]):
            mu_X_CAM5.append(np.mean(X[:,i]))
            sigma_X_CAM5.append(np.std(X[:,i]))
        mu_X_CAM5 = np.array(np.array(mu_X_CAM5),dtype=np.float32)
        sigma_X_CAM5 = np.array(np.array(sigma_X_CAM5),dtype=np.float32)
        np.save('norm/mu_X_CAM5.npy',mu_X_CAM5)
        np.save('norm/sigma_X_CAM5.npy',sigma_X_CAM5)
    else:
        if case_var == 'moist':
            X = np.load('data_CAM5_8K/all_outputs_moist.npy')
            X = np.array(X,dtype=np.float64)
            
            mu_y_moist_CAM5, sigma_y_moist_CAM5 = [], []
            for i in range(X.shape[1]):
                mu_y_moist_CAM5.append(np.mean(X[:,i]))
                sigma_y_moist_CAM5.append(np.std(X[:,i]))
            mu_y_moist_CAM5 = np.array(np.array(mu_y_moist_CAM5),dtype=np.float32)
            sigma_y_moist_CAM5 = np.array(np.array(sigma_y_moist_CAM5),dtype=np.float32)
            np.save('norm/mu_y_moist_CAM5.npy',mu_y_moist_CAM5)
            np.save('norm/sigma_y_moist_CAM5.npy',sigma_y_moist_CAM5)
        else:
            X = np.load('data_CAM5_8K/all_outputs_heat.npy')
            X = np.array(X,dtype=np.float64)
            
            mu_y_heat_CAM5, sigma_y_heat_CAM5 = [], []
            for i in range(X.shape[1]):
                mu_y_heat_CAM5.append(np.mean(X[:,i]))
                sigma_y_heat_CAM5.append(np.std(X[:,i]))
            mu_y_heat_CAM5 = np.array(np.array(mu_y_heat_CAM5),dtype=np.float32)
            sigma_y_heat_CAM5 = np.array(np.array(sigma_y_heat_CAM5),dtype=np.float32)
            np.save('norm/mu_y_heat_CAM5.npy',mu_y_heat_CAM5)
            np.save('norm/sigma_y_heat_CAM5.npy',sigma_y_heat_CAM5)
elif sim == 'SPCAM':
    if is_input == 1:
        X = np.load('data_SPCAM5_hist/three_month_inputs.npy')
        X = np.array(X,dtype=np.float64)
        
        mu_X_SPCAM5, sigma_X_SPCAM5 = [], []
        # we do not use the vectorial format of computing mean and std on purpose
        # in order to avoid the risk of precision error given the dataset size
        for i in range(X.shape[1]):
            mu_X_SPCAM5.append(np.mean(X[:,i]))
            sigma_X_SPCAM5.append(np.std(X[:,i]))
        mu_X_SPCAM5 = np.array(np.array(mu_X_SPCAM5),dtype=np.float32)
        sigma_X_SPCAM5 = np.array(np.array(sigma_X_SPCAM5),dtype=np.float32)
        np.save('norm/mu_X_SPCAM5.npy',mu_X_SPCAM5)
        np.save('norm/sigma_X_SPCAM5.npy',sigma_X_SPCAM5)
    else:
        if case_var == 'moist':
            X = np.load('data_SPCAM5_hist/three_month_outputs_moist.npy')
            X = np.array(X,dtype=np.float64)
            
            mu_y_moist_SPCAM5, sigma_y_moist_SPCAM5 = [], []
            for i in range(X.shape[1]):
                mu_y_moist_SPCAM5.append(np.mean(X[:,i]))
                sigma_y_moist_SPCAM5.append(np.std(X[:,i]))
            mu_y_moist_SPCAM5 = np.array(np.array(mu_y_moist_SPCAM5),dtype=np.float32)
            sigma_y_moist_SPCAM5 = np.array(np.array(sigma_y_moist_SPCAM5),dtype=np.float32)
            np.save('norm/mu_y_moist_SPCAM5.npy',mu_y_moist_SPCAM5)
            np.save('norm/sigma_y_moist_SPCAM5.npy',sigma_y_moist_SPCAM5)
        else:
            X = np.load('data_SPCAM5_hist/three_month_outputs_heat.npy')
            X = np.array(X,dtype=np.float64)
            
            mu_y_heat_SPCAM5, sigma_y_heat_SPCAM5 = [], []
            for i in range(X.shape[1]):
                mu_y_heat_SPCAM5.append(np.mean(X[:,i]))
                sigma_y_heat_SPCAM5.append(np.std(X[:,i]))
            mu_y_heat_SPCAM5 = np.array(np.array(mu_y_heat_SPCAM5),dtype=np.float32)
            sigma_y_heat_SPCAM5 = np.array(np.array(sigma_y_heat_SPCAM5),dtype=np.float32)
            np.save('norm/mu_y_heat_SPCAM5.npy',mu_y_heat_SPCAM5)
            np.save('norm/sigma_y_heat_SPCAM5.npy',sigma_y_heat_SPCAM5)