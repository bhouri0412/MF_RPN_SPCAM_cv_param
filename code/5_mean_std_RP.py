#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:19:00 2022

@author: mohamedazizbhouri
"""

import numpy as onp
import time

n_remove = 4
ind_output_heat = onp.arange(26)
ind_output_moist = n_remove+onp.arange(26-n_remove)

is_MF = 1
is_SF = 0
is_LF = 0

if is_MF == 1:
    mu_SF_out = onp.concatenate((onp.load('norm/mu_y_heat_CAM5.npy')[None,ind_output_heat],
                                 onp.load('norm/mu_y_moist_CAM5.npy')[None,ind_output_moist]),axis=1)
    sigma_SF_out = onp.concatenate((onp.load('norm/sigma_y_heat_CAM5.npy')[None,ind_output_heat],
                                    onp.load('norm/sigma_y_moist_CAM5.npy')[None,ind_output_moist]),axis=1)
    
    is_comp_mean = 1
    id_step = 1
    N_rpn = 128
    print('loading data')
    
    if is_comp_mean == 1: # RPN mean computation
        samples_test_H = onp.load('MF_param/MF_param_'+str(0)+'/test_pred_'+str(id_step)+'.npy')[0,:,:]/N_rpn
        t = time.time()
        for i in range(N_rpn-1):
            print(id_step,i,N_rpn-1)
            samples_test_H = samples_test_H +onp.load('MF_param/MF_param_'+str(i+1)+'/test_pred_'+str(id_step)+'.npy')[0,:,:]/N_rpn
            print(time.time()-t)
            t = time.time()
        samples_test_H = mu_SF_out + sigma_SF_out * samples_test_H
        onp.save('MF_param/mean_RPN_MF_'+str(id_step)+'.npy',samples_test_H)
    else: # RPN std computation
        mu = onp.load('MF_param/mean_RPN_MF_'+str(id_step)+'.npy')
        
        sigma = (mu_SF_out + onp.load('MF_param/MF_param_'+str(0)+'/test_pred_'+str(id_step)+'.npy')[0,:,:] * sigma_SF_out - mu)**2/N_rpn
        t = time.time()
        for i in range(N_rpn-1):
            print(id_step,i,N_rpn-1)
            sigma = sigma + (mu_SF_out + onp.load('MF_param/MF_param_'+str(0)+'/test_pred_'+str(id_step)+'.npy')[0,:,:] * sigma_SF_out - mu)**2/N_rpn
            print(time.time()-t)
            t = time.time()
        sigma = onp.sqrt(sigma)
        onp.save('MF_param/std_RPN_MF_'+str(id_step)+'.npy',sigma)

if is_SF == 1:
    mu_SF_out = onp.concatenate((onp.load('norm/mu_y_heat_SPCAM5.npy')[None,ind_output_heat],
                                 onp.load('norm/mu_y_moist_SPCAM5.npy')[None,ind_output_moist]),axis=1)
    sigma_SF_out = onp.concatenate((onp.load('norm/sigma_y_heat_SPCAM5.npy')[None,ind_output_heat],
                                    onp.load('norm/sigma_y_moist_SPCAM5.npy')[None,ind_output_moist]),axis=1)
    
    is_comp_mean = 1
    N_rpn = 128
    print('loading data')
    
    if is_comp_mean == 1: # RPN mean computation
        samples_test_H = onp.concatenate( (onp.load('SF_param/SF_param_'+str(0)+'/test_pred_1.npy'),
                                          onp.load('SF_param/SF_param_'+str(0)+'/test_pred_2.npy')), axis=1)[0,:,:]/N_rpn
        t = time.time()
        for i in range(N_rpn-1):
            print(i,N_rpn-1)
            samples_test_H = samples_test_H + onp.concatenate( (onp.load('SF_param/SF_param_'+str(i+1)+'/test_pred_1.npy'),
                                                                onp.load('SF_param/SF_param_'+str(i+1)+'/test_pred_2.npy')), axis=1)[0,:,:]/N_rpn
            print(time.time()-t)
            t = time.time()
        samples_test_H = mu_SF_out + sigma_SF_out * samples_test_H
        onp.save('SF_param/mean_RPN_SF.npy',samples_test_H)
    else: # RPN std computation
        mu = onp.load('SF_param/mean_RPN_SF.npy')
        sigma = (mu_SF_out + onp.concatenate( (onp.load('SF_param/SF_param_'+str(0)+'/test_pred_1.npy'),
                                               onp.load('SF_param/SF_param_'+str(0)+'/test_pred_2.npy')), axis=1)[0,:,:] * sigma_SF_out - mu)**2/N_rpn
        t = time.time()
        for i in range(N_rpn-1):
            print(i,N_rpn-1)
            sigma = sigma + (mu_SF_out + onp.concatenate( (onp.load('SF_param/SF_param_'+str(i+1)+'/test_pred_1.npy'),
                                                           onp.load('SF_param/SF_param_'+str(i+1)+'/test_pred_2.npy')), axis=1)[0,:,:] * sigma_SF_out - mu)**2/N_rpn
            print(time.time()-t)
            t = time.time()
        sigma = onp.sqrt(sigma)
        onp.save('SF_param/std_RPN_SF.npy',sigma)

if is_LF == 1:
    mu_MF_out = onp.concatenate((onp.load('norm/mu_y_heat_CAM5.npy')[None,ind_output_heat],
                                 onp.load('norm/mu_y_moist_CAM5.npy')[None,ind_output_moist]),axis=1)
    sigma_MF_out = onp.concatenate((onp.load('norm/sigma_y_heat_CAM5.npy')[None,ind_output_heat],
                                    onp.load('norm/sigma_y_moist_CAM5.npy')[None,ind_output_moist]),axis=1)
    
    is_comp_mean = 1
    id_step = 2
    N_rpn = 128
    print('loading data')
    
    if is_comp_mean == 1: # RPN mean computation
        samples_test_H = onp.load('MF_param/MF_param_'+str(0)+'/LF_test_pred_'+str(id_step)+'.npy')[0,:,:]/N_rpn
        t = time.time()
        for i in range(N_rpn-1):
            print(id_step,i,N_rpn-1)
            samples_test_H = samples_test_H + onp.load('MF_param/MF_param_'+str(i+1)+'/LF_test_pred_'+str(id_step)+'.npy')[0,:,:]/N_rpn
            print(time.time()-t)
            t = time.time()
        samples_test_H = mu_MF_out + sigma_MF_out * samples_test_H
        onp.save('MF_param/mean_RPN_LF_'+str(id_step)+'.npy',samples_test_H)
    else: # RPN std computation
        mu = onp.load('MF_param/mean_RPN_LF_'+str(id_step)+'.npy')
        
        sigma = (mu_MF_out + onp.load('MF_param/MF_param_'+str(0)+'/LF_test_pred_'+str(id_step)+'.npy')[0,:,:] * sigma_MF_out - mu)**2/N_rpn
        t = time.time()
        for i in range(N_rpn-1):
            print(id_step,i,N_rpn-1)
            sigma = sigma + (mu_MF_out + onp.load('MF_param/MF_param_'+str(i+1)+'/LF_test_pred_'+str(id_step)+'.npy')[0,:,:] * sigma_MF_out - mu)**2/N_rpn
            print(time.time()-t)
            t = time.time()
        sigma = onp.sqrt(sigma)
        onp.save('MF_param/std_RPN_LF_'+str(id_step)+'.npy',sigma)
    