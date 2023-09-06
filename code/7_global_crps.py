#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:20:34 2023

@author: mohamedazizbhouri
"""

import numpy as onp
import time

from matplotlib import pyplot as plt
plt.close('all')

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 32,
                     'lines.linewidth': 2,
                     'axes.labelsize': 32,
                     'axes.titlesize': 32,
                     'xtick.labelsize': 32,
                     'ytick.labelsize': 32,
                     'legend.fontsize': 32,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex"
                     })

dim_y = 48 
dim_heat = 26
dim_moist = 22
n_remove = 4
ind_input = onp.concatenate( (onp.arange(26),n_remove+26+onp.arange(26-n_remove),onp.array([52,53,54,55])) ) 
dim_xH = ind_input.shape[0]
dim_xL = ind_input.shape[0]

ind_output_heat = onp.arange(26)
ind_output_moist = n_remove+onp.arange(26-n_remove)
    
mu_error_out = onp.concatenate((onp.zeros((1,dim_heat),dtype=onp.float32),
                                onp.zeros((1,dim_moist),dtype=onp.float32)),axis=1)

sigma_error_out = onp.concatenate((1/1004.6*onp.ones((1,dim_heat),dtype=onp.float32),
                                    1/2.26e6*onp.ones((1,dim_moist),dtype=onp.float32)),axis=1)

is_reshape_single_pred = 1

is_MF = 0
is_LF = 0
is_SF = 1

if is_reshape_single_pred == 1:
    
    test_SPCAM =['2003_02_06','2003_02_12','2003_02_18','2003_02_24','2003_02_28',
              '2003_03_06','2003_03_12','2003_03_18','2003_03_24','2003_03_30','2003_03_31',
              '2003_04_06','2003_04_12','2003_04_18','2003_04_24','2003_04_30',
              '2003_05_06','2003_05_12','2003_05_18','2003_05_24','2003_05_30','2003_05_31',
              '2003_06_06','2003_06_12','2003_06_18','2003_06_24','2003_06_30',
              '2003_07_06','2003_07_12','2003_07_18','2003_07_24','2003_07_30','2003_07_31',
              '2003_08_06','2003_08_12','2003_08_18','2003_08_24','2003_08_30','2003_08_31',
              '2003_09_06','2003_09_12','2003_09_18','2003_09_24','2003_09_30',
              '2003_10_06','2003_10_12','2003_10_18','2003_10_24','2003_10_30','2003_10_31',
              '2003_11_06','2003_11_12','2003_11_18','2003_11_24','2003_11_30',
              '2003_12_06','2003_12_12','2003_12_18','2003_12_24','2003_12_30','2003_12_31',
              '2004_01_06','2004_01_12','2004_01_18','2004_01_24','2004_01_30','2004_01_31']

    Npts_per_file = onp.load('data_SPCAM5_4K/Npts_per_file_test.npy')
        
    def reshape_loc_onp(pred, dim_y):
        pred_loc = pred[:Npts_per_file[0],:]
        pred = pred[Npts_per_file[0]:,:]
        nt_total = pred_loc.shape[0]//(lat*lon)
        pred_array = onp.reshape(pred_loc.T, (dim_y,nt_total,lat,lon))
        
        for i in range(len(test_SPCAM)-1):
            print(i,len(test_SPCAM)-1)
            pred_loc = pred[:Npts_per_file[i+1],:]
            pred = pred[Npts_per_file[i+1]:,:]
            nt_total = pred_loc.shape[0]//(lat*lon)
            
            pred_array = onp.concatenate( (pred_array, onp.reshape(pred_loc.T, (dim_y,nt_total,lat,lon))),axis=1)
        return pred_array 

    case_var = 'all'
    lat = 96
    lon = 144
    N_dt_day = 24 # we have a dt=1hour
    def daily_avg(test):
        test_daily = []
        N_time_steps = test.shape[1]
        for i in range(test.shape[0]):
            test_daily.append( onp.mean( test[i,:,:,:].reshape( (N_time_steps//N_dt_day, N_dt_day, lat, lon) ), axis=1 ) )
        return onp.array(test_daily) # dim_y x N_day x lat x lon
      
    if is_MF == 1 or is_LF == 1:
        mu_SF_out = onp.concatenate((onp.load('norm/mu_y_heat_CAM5.npy')[None,ind_output_heat],
                                     onp.load('norm/mu_y_moist_CAM5.npy')[None,ind_output_moist]),axis=1)
        sigma_SF_out = onp.concatenate((onp.load('norm/sigma_y_heat_CAM5.npy')[None,ind_output_heat],
                                        onp.load('norm/sigma_y_moist_CAM5.npy')[None,ind_output_moist]),axis=1)
    
    if is_SF == 1:
        mu_SF_out = onp.concatenate((onp.load('norm/mu_y_heat_SPCAM5.npy')[None,ind_output_heat],
                                     onp.load('norm/mu_y_moist_SPCAM5.npy')[None,ind_output_moist]),axis=1)
        sigma_SF_out = onp.concatenate((onp.load('norm/sigma_y_heat_SPCAM5.npy')[None,ind_output_heat],
                                        onp.load('norm/sigma_y_moist_SPCAM5.npy')[None,ind_output_moist]),axis=1)
    
    tt = time.time()
    for i in range(32):
        ieff = i + 0
        print(ieff,time.time()-tt)
        tt = time.time()
        if is_MF == 1:
            samples_test_H = onp.concatenate( (onp.load('MF_param/MF_param_'+str(ieff)+'/test_pred_1.npy')[0,:,:],
                                               onp.load('MF_param/MF_param_'+str(ieff)+'/test_pred_2.npy')[0,:,:]),axis=0)
        if is_SF == 1:
            samples_test_H = onp.concatenate( (onp.load('SF_param/SF_param_'+str(ieff)+'/test_pred_1.npy')[0,:,:],
                                               onp.load('SF_param/SF_param_'+str(ieff)+'/test_pred_1.npy')[0,:,:]),axis=0)
        if is_LF == 1:
            samples_test_H = onp.concatenate( (onp.load('MF_param/MF_param_'+str(ieff)+'/LF_test_pred_1.npy')[0,:,:],
                                               onp.load('MF_param/MF_param_'+str(ieff)+'/LF_test_pred_2.npy')[0,:,:]),axis=0)
        samples_test_H = mu_SF_out + sigma_SF_out * samples_test_H
        samples_test_H = (samples_test_H - mu_error_out) / sigma_error_out
        
        samples_test_H = reshape_loc_onp(samples_test_H, dim_y)
        samples_test_H = daily_avg(samples_test_H)
        samples_test_H = samples_test_H.reshape((dim_y, samples_test_H.shape[1]*lat*lon))
        samples_test_H = samples_test_H.T
        print(samples_test_H.shape)
        # Npts x dim_y
        if is_MF == 1:
            onp.save('MF_param/MF_param_'+str(ieff)+'/test_pred_reshaped.npy', samples_test_H)
        if is_SF == 1:
            onp.save('SF_param/SF_param_'+str(ieff)+'/test_pred_reshaped.npy', samples_test_H)
        if is_LF == 1:
            onp.save('MF_param/MF_param_'+str(ieff)+'/LF_test_pred_reshaped.npy', samples_test_H)
    
    test = onp.load('data_SPCAM5_4K/all_outputs_reshaped.npy')
    test = daily_avg(test)
    test = test.reshape((dim_y, test.shape[1]*lat*lon)) # dim_y x N_samples
    test = test.T
    test = (test - mu_error_out) / sigma_error_out
    onp.save('data_SPCAM5_4K/all_outputs_reshaped_temp_avg.npy', test)

else:
    
    test = onp.load('data_SPCAM5_4K/all_outputs_reshaped_temp_avg.npy')
    test = onp.array(test,dtype=onp.float64)
         
    def crps(outputs, target, weights=None):
        """
        Computes the Continuous Ranked Probability Score (CRPS) between the target and the ecdf for each output variable and then takes a weighted average over them.
    
        Input
        -----
        outputs - float[B, F, S] samples from the model
        target - float[B, F] ground truth target
        """
        tt = time.time()
        n = outputs.shape[2]
        y_hats = onp.sort(outputs, axis=-1)
        print('sort',time.time()-tt)
        
        tt = time.time()
        # E[Y - y]
        mae = onp.abs(target[..., None] - y_hats).mean(axis=(0, -1))
        print('abs',time.time()-tt)
        
        tt = time.time()
        # E[Y - Y'] ~= sum_i sum_j |Y_i - Y_j| / (2 * n * (n-1))
        diff = y_hats[..., 1:] - y_hats[..., :-1]
        print('abs2',time.time()-tt)
        
        tt = time.time()
        count = onp.arange(1, n) * onp.arange(n - 1, 0, -1)
        print('arange',time.time()-tt)
        
        tt = time.time()
        crps = mae - (diff * count).sum(axis=-1).mean(axis=0) / (2 * n * (n-1))
        print('crps final',time.time()-tt)
        return crps
    
    if is_MF == 1:
        ieff = 0
        pred_daily = onp.load('MF_param/MF_param_'+str(ieff)+'/test_pred_reshaped.npy')[:,:,None]
        for i in range(31):
            print(i)
            ieff = i+1
            pred_daily = onp.concatenate( (pred_daily,onp.load('MF_param/MF_param_'+str(ieff)+'/test_pred_reshaped.npy')[:,:,None]),axis=2)
        pred_daily = onp.array(pred_daily,dtype=onp.float64)
    
        crps_f = crps(pred_daily, test)
        onp.save('glob_errors/crps_rpn_MF.npy',crps_f)
    
    if is_SF == 1:
        ieff = 0
        pred_daily = onp.load('SF_param/SF_param_'+str(ieff)+'/test_pred_reshaped.npy')[:,:,None]
        for i in range(31):
            print(i)
            ieff = i+1
            pred_daily = onp.concatenate( (pred_daily,onp.load('SF_param/SF_param_'+str(ieff)+'/test_pred_reshaped.npy')[:,:,None]),axis=2)
        pred_daily = onp.array(pred_daily,dtype=onp.float64)
    
        crps_f = crps(pred_daily, test)
        print(crps_f.shape)
        print(onp.array(crps_f).shape)
        onp.save('glob_errors/crps_rpn_SF.npy',crps_f)
    
    if is_LF == 1:
        ieff = 0
        pred_daily = onp.load('MF_param/MF_param_'+str(ieff)+'/LF_test_pred_reshaped.npy')[:,:,None]
        for i in range(31):
            print(i)
            ieff = i+1
            pred_daily = onp.concatenate( (pred_daily,onp.load('MF_param/MF_param_'+str(ieff)+'/LF_test_pred_reshaped.npy')[:,:,None]),axis=2)
        pred_daily = onp.array(pred_daily,dtype=onp.float64)
    
        crps_f = crps(pred_daily, test)
        print(crps_f.shape)
        print(onp.array(crps_f).shape)
        onp.save('glob_errors/crps_rpn_LF.npy',crps_f)
      