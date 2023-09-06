#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:04:50 2022

@author: mohamedazizbhouri
"""

import os
current_dirs_parent = os.path.dirname(os.getcwd())
current_dirs_parent = os.path.dirname(current_dirs_parent)
current_dirs_parent = os.path.dirname(current_dirs_parent)

CAM_train_input = 0
CAM_train_output = 0
CAM_4K_input = 0
CAM_4K_output = 0

SPCAM_train_input = 0
SPCAM_train_output = 0
SPCAM_test_input = 0
SPCAM_test_output = 0

is_output_heat = 1

if SPCAM_train_input == 1 or SPCAM_train_output == 1:
    pp_SPCAM = 'data_SPCAM5_hist/'
if SPCAM_test_input == 1 or SPCAM_test_output == 1:
    pp_SPCAM ='data_SPCAM5_4K/'
if CAM_train_input == 1 or CAM_train_output == 1:
    pp_CAM = 'data_CAM5_8K/' 
if CAM_4K_input == 1 or CAM_4K_output == 1:
    pp_CAM = 'data_CAM5_4K/' 
    
import numpy as np

train_CAM =['2003_02_06','2003_02_12','2003_02_18','2003_02_24','2003_02_28',
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

train_SPCAM =['2003_02_06','2003_02_12','2003_02_18','2003_02_24','2003_02_28',
              '2003_03_06','2003_03_12','2003_03_18','2003_03_24','2003_03_30','2003_03_31',
              '2003_04_06','2003_04_12','2003_04_18','2003_04_24','2003_04_30']

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

import gc

ind_input = np.concatenate( (np.arange(52),np.array([104,105,106,107])) )
if is_output_heat == 1:
    ind_output = np.arange(26) # heat tend
else:
    ind_output = 26+np.arange(26) # moist tend

if CAM_train_input == 1 or CAM_4K_input == 1: 
    train_CAM_in = np.load(pp_CAM+'inputs_'+train_CAM[0]+'.npy')[:,ind_input]
    gc.collect()
    for i in range(len(train_CAM)-1):
        print(train_CAM[i+1])
        train_CAM_in = np.concatenate((train_CAM_in,
                                       np.load(pp_CAM+'inputs_'+train_CAM[i+1]+'.npy')[:,ind_input]),axis=0)
        gc.collect()
    np.save(pp_CAM+'all_inputs.npy',train_CAM_in)
    print('Final data shape: ', train_CAM_in.shape)

if CAM_train_output == 1 or CAM_4K_output == 1:    
    train_CAM_out = np.load(pp_CAM+'outputs_'+train_CAM[0]+'.npy')[:,ind_output]
    gc.collect()
    for i in range(len(train_CAM)-1):
        print(train_CAM[i+1])
        train_CAM_out = np.concatenate((train_CAM_out,
                                        np.load(pp_CAM+'outputs_'+train_CAM[i+1]+'.npy')[:,ind_output]),axis=0)
        gc.collect()
    
    if is_output_heat == 1:
        np.save(pp_CAM+'all_outputs_heat.npy',train_CAM_out)
    else:
        np.save(pp_CAM+'all_outputs_moist.npy',train_CAM_out)
    print('Final data shape: ', train_CAM_out.shape) 
    
if SPCAM_train_input == 1:
    train_SPCAM_in = np.load(pp_SPCAM+'inputs_'+train_SPCAM[0]+'.npy')[:,ind_input] 
    gc.collect()
    for i in range(len(train_SPCAM)-1):
        print(train_SPCAM[i+1])
        train_SPCAM_in = np.concatenate((train_SPCAM_in,
                                         np.load(pp_SPCAM+'inputs_'+train_SPCAM[i+1]+'.npy')[:,ind_input]),axis=0)
        gc.collect()
    np.save(pp_SPCAM+'three_month_inputs.npy',train_SPCAM_in)
    print('Final data shape: ', train_SPCAM_in.shape)
    
if SPCAM_train_output == 1:   
    train_SPCAM_out = np.load(pp_SPCAM+'outputs_'+train_SPCAM[0]+'.npy')[:,ind_output]
    gc.collect()
    for i in range(len(train_SPCAM)-1):
        print(train_SPCAM[i+1])
        train_SPCAM_out = np.concatenate((train_SPCAM_out,
                                          np.load(pp_SPCAM+'outputs_'+train_SPCAM[i+1]+'.npy')[:,ind_output]),axis=0)
        gc.collect()
    if is_output_heat == 1:
        np.save(pp_SPCAM+'three_month_outputs_heat.npy',train_SPCAM_out)
    else:
        np.save(pp_SPCAM+'three_month_outputs_moist.npy',train_SPCAM_out)
    print('Final data shape: ', train_SPCAM_out.shape) 

if SPCAM_test_input == 1:
    test_SPCAM_in = np.load(pp_SPCAM+'inputs_'+test_SPCAM[0]+'.npy')[:,ind_input]
    gc.collect()
    for i in range(len(test_SPCAM)-1):
        print(test_SPCAM[i+1])
        test_SPCAM_in = np.concatenate((test_SPCAM_in,
                                        np.load(pp_SPCAM+'inputs_'+test_SPCAM[i+1]+'.npy')[:,ind_input]),axis=0)
        gc.collect()
    np.save(pp_SPCAM+'all_inputs.npy',test_SPCAM_in)
    print('Final data shape: ', test_SPCAM_in.shape) 
    
if SPCAM_test_output == 1:   
    test_SPCAM_out = np.load(pp_SPCAM+'outputs_'+test_SPCAM[0]+'.npy')[:,ind_output]
    gc.collect()
    for i in range(len(test_SPCAM)-1):
        print(test_SPCAM[i+1])
        test_SPCAM_out = np.concatenate((test_SPCAM_out,
                                         np.load(pp_SPCAM+'outputs_'+test_SPCAM[i+1]+'.npy')[:,ind_output]),axis=0)
        gc.collect()
    if is_output_heat == 1:
        np.save(pp_SPCAM+'all_outputs_heat.npy',test_SPCAM_out)
    else:
        np.save(pp_SPCAM+'all_outputs_moist.npy',test_SPCAM_out)
    print('Final data shape: ', test_SPCAM_out.shape) 
