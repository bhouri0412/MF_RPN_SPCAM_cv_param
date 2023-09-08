#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:59:59 2023

@author: mohamedazizbhouri
"""
import numpy as onp

lat = 96
lon = 144

dim_heat = 26
dim_moist = 22
mu_error_out = onp.concatenate((onp.zeros((1,dim_heat),dtype=onp.float32),
                                onp.zeros((1,dim_moist),dtype=onp.float32)),axis=1)
mu_error_out = mu_error_out.T[:,:,None,None]
sigma_error_out = onp.concatenate((1/1004.6*onp.ones((1,dim_heat),dtype=onp.float32),
                                    1/2.26e6*onp.ones((1,dim_moist),dtype=onp.float32)),axis=1)    
sigma_error_out = sigma_error_out.T[:,:,None,None]

is_det = 0
is_SF = 0
is_MF = 0
is_LF = 1

N_dt_day = 24 # we have a dt=1hour

test = (onp.load('data_SPCAM5_4K/all_outputs_reshaped.npy') - mu_error_out) / sigma_error_out
test = onp.array(test,dtype=onp.float64)

def daily_avg(test):
    test_daily = []
    N_time_steps = test.shape[1]
    for i in range(test.shape[0]):
        test_daily.append( onp.mean( test[i,:,:,:].reshape( (N_time_steps//N_dt_day, N_dt_day, lat, lon) ), axis=1 ) )
    return onp.array(test_daily)

test = daily_avg(test)

if is_LF == 1:
    pred = (onp.load('MF_param/mean_RPN_LF_reshaped.npy') - mu_error_out) / sigma_error_out
    pred = onp.array(pred,dtype=onp.float64)
    pred = daily_avg(pred)
    pred = onp.mean(pred, axis = 3)
    test = onp.mean(test, axis = 3)
    
    print('Compute R^2')
    r2 = 1 - onp.sum( (test-pred)**2, axis=1)/onp.sum( (test-onp.mean(test,axis=1)[:,None,:])**2, axis=1)
    onp.save('MF_results/r2_LF_pres_lat.npy', r2)

if is_det == 1:
    pred = (onp.load('SF_param/SF_param_det/test_pred_reshaped.npy') - mu_error_out) / sigma_error_out
    pred = onp.array(pred,dtype=onp.float64)
    pred = daily_avg(pred)
    pred = onp.mean(pred, axis = 3)
    test = onp.mean(test, axis = 3)
    
    print('Compute R^2')
    r2 = 1 - onp.sum( (test-pred)**2, axis=1)/onp.sum( (test-onp.mean(test,axis=1)[:,None,:])**2, axis=1)
    onp.save('SF_results/r2_det_pres_lat.npy',r2)
    
if is_SF == 1:
    pred = (onp.load('SF_param/mean_RPN_SF_reshaped.npy') - mu_error_out) / sigma_error_out
    pred = onp.array(pred,dtype=onp.float64)
    pred = daily_avg(pred)
    pred = onp.mean(pred, axis = 3)
    test = onp.mean(test, axis = 3)
    
    print('Compute R^2')
    r2 = 1 - onp.sum( (test-pred)**2, axis=1)/onp.sum( (test-onp.mean(test,axis=1)[:,None,:])**2, axis=1)
    onp.save('SF_results/r2_SF_pres_lat.npy', r2)
    
if is_MF == 1:
    pred = (onp.load('MF_param/mean_RPN_MF_reshaped.npy') - mu_error_out) / sigma_error_out
    pred = onp.array(pred,dtype=onp.float64)
    pred = daily_avg(pred)
    pred = onp.mean(pred, axis = 3)
    test = onp.mean(test, axis = 3)
    
    print('Compute R^2')
    r2 = 1 - onp.sum( (test-pred)**2, axis=1)/onp.sum( (test-onp.mean(test,axis=1)[:,None,:])**2, axis=1)
    onp.save('MF_results/r2_MF_pres_lat.npy', r2)
