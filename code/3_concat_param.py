#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:23:32 2023

@author: mohamedazizbhouri
"""

import numpy as np

N_param_gp = 128

is_MF = 0
is_LF = 0
is_SF = 1

if is_MF == 1:
    HF_params_0_0 = np.load('MF_param/MF_param_0/HF_params_0_0.npy')
    HF_params_0_1 = np.load('MF_param/MF_param_0/HF_params_0_1.npy')
    HF_params_1_0 = np.load('MF_param/MF_param_0/HF_params_1_0.npy')
    HF_params_1_1 = np.load('MF_param/MF_param_0/HF_params_1_1.npy')
    HF_params_2_0 = np.load('MF_param/MF_param_0/HF_params_2_0.npy')
    HF_params_2_1 = np.load('MF_param/MF_param_0/HF_params_2_1.npy')
    HF_params_3_0 = np.load('MF_param/MF_param_0/HF_params_3_0.npy')
    HF_params_3_1 = np.load('MF_param/MF_param_0/HF_params_3_1.npy')
    HF_params_4_0 = np.load('MF_param/MF_param_0/HF_params_4_0.npy')
    HF_params_4_1 = np.load('MF_param/MF_param_0/HF_params_4_1.npy')
    HF_params_5_0 = np.load('MF_param/MF_param_0/HF_params_5_0.npy')
    HF_params_5_1 = np.load('MF_param/MF_param_0/HF_params_5_1.npy')
    HF_params_6_0 = np.load('MF_param/MF_param_0/HF_params_6_0.npy')
    HF_params_6_1 = np.load('MF_param/MF_param_0/HF_params_6_1.npy')
    HF_params_7_0 = np.load('MF_param/MF_param_0/HF_params_7_0.npy')
    HF_params_7_1 = np.load('MF_param/MF_param_0/HF_params_7_1.npy')
    
    HF_params_prior_0_0 = np.load('MF_param/MF_param_0/HF_params_prior_0_0.npy')
    HF_params_prior_0_1 = np.load('MF_param/MF_param_0/HF_params_prior_0_1.npy')
    HF_params_prior_1_0 = np.load('MF_param/MF_param_0/HF_params_prior_1_0.npy')
    HF_params_prior_1_1 = np.load('MF_param/MF_param_0/HF_params_prior_1_1.npy')
    HF_params_prior_2_0 = np.load('MF_param/MF_param_0/HF_params_prior_2_0.npy')
    HF_params_prior_2_1 = np.load('MF_param/MF_param_0/HF_params_prior_2_1.npy')
    HF_params_prior_3_0 = np.load('MF_param/MF_param_0/HF_params_prior_3_0.npy')
    HF_params_prior_3_1 = np.load('MF_param/MF_param_0/HF_params_prior_3_1.npy')
    HF_params_prior_4_0 = np.load('MF_param/MF_param_0/HF_params_prior_4_0.npy')
    HF_params_prior_4_1 = np.load('MF_param/MF_param_0/HF_params_prior_4_1.npy')
    HF_params_prior_5_0 = np.load('MF_param/MF_param_0/HF_params_prior_5_0.npy')
    HF_params_prior_5_1 = np.load('MF_param/MF_param_0/HF_params_prior_5_1.npy')
    HF_params_prior_6_0 = np.load('MF_param/MF_param_0/HF_params_prior_6_0.npy')
    HF_params_prior_6_1 = np.load('MF_param/MF_param_0/HF_params_prior_6_1.npy')
    HF_params_prior_7_0 = np.load('MF_param/MF_param_0/HF_params_prior_7_0.npy')
    HF_params_prior_7_1 = np.load('MF_param/MF_param_0/HF_params_prior_7_1.npy')
    
    for i in range(N_param_gp-1):
        
        print(i)
        HF_params_0_0 = np.concatenate( (HF_params_0_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_0_0.npy')), axis = 0)
        HF_params_0_1 = np.concatenate( (HF_params_0_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_0_1.npy')), axis = 0)
        HF_params_1_0 = np.concatenate( (HF_params_1_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_1_0.npy')), axis = 0)
        HF_params_1_1 = np.concatenate( (HF_params_1_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_1_1.npy')), axis = 0)
        HF_params_2_0 = np.concatenate( (HF_params_2_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_2_0.npy')), axis = 0)
        HF_params_2_1 = np.concatenate( (HF_params_2_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_2_1.npy')), axis = 0)
        HF_params_3_0 = np.concatenate( (HF_params_3_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_3_0.npy')), axis = 0)
        HF_params_3_1 = np.concatenate( (HF_params_3_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_3_1.npy')), axis = 0)
        HF_params_4_0 = np.concatenate( (HF_params_4_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_4_0.npy')), axis = 0)
        HF_params_4_1 = np.concatenate( (HF_params_4_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_4_1.npy')), axis = 0)
        HF_params_5_0 = np.concatenate( (HF_params_5_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_5_0.npy')), axis = 0)
        HF_params_5_1 = np.concatenate( (HF_params_5_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_5_1.npy')), axis = 0)    
        HF_params_6_0 = np.concatenate( (HF_params_6_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_6_0.npy')), axis = 0)
        HF_params_6_1 = np.concatenate( (HF_params_6_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_6_1.npy')), axis = 0)    
        HF_params_7_0 = np.concatenate( (HF_params_7_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_7_0.npy')), axis = 0)
        HF_params_7_1 = np.concatenate( (HF_params_7_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_7_1.npy')), axis = 0)
    
        HF_params_prior_0_0 = np.concatenate( (HF_params_prior_0_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_0_0.npy')), axis = 0)
        HF_params_prior_0_1 = np.concatenate( (HF_params_prior_0_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_0_1.npy')), axis = 0)
        HF_params_prior_1_0 = np.concatenate( (HF_params_prior_1_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_1_0.npy')), axis = 0)
        HF_params_prior_1_1 = np.concatenate( (HF_params_prior_1_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_1_1.npy')), axis = 0)        
        HF_params_prior_2_0 = np.concatenate( (HF_params_prior_2_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_2_0.npy')), axis = 0)
        HF_params_prior_2_1 = np.concatenate( (HF_params_prior_2_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_2_1.npy')), axis = 0)        
        HF_params_prior_3_0 = np.concatenate( (HF_params_prior_3_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_3_0.npy')), axis = 0)
        HF_params_prior_3_1 = np.concatenate( (HF_params_prior_3_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_3_1.npy')), axis = 0)        
        HF_params_prior_4_0 = np.concatenate( (HF_params_prior_4_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_4_0.npy')), axis = 0)
        HF_params_prior_4_1 = np.concatenate( (HF_params_prior_4_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_4_1.npy')), axis = 0)        
        HF_params_prior_5_0 = np.concatenate( (HF_params_prior_5_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_5_0.npy')), axis = 0)
        HF_params_prior_5_1 = np.concatenate( (HF_params_prior_5_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_5_1.npy')), axis = 0)        
        HF_params_prior_6_0 = np.concatenate( (HF_params_prior_6_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_6_0.npy')), axis = 0)
        HF_params_prior_6_1 = np.concatenate( (HF_params_prior_6_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_6_1.npy')), axis = 0)        
        HF_params_prior_7_0 = np.concatenate( (HF_params_prior_7_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_7_0.npy')), axis = 0)
        HF_params_prior_7_1 = np.concatenate( (HF_params_prior_7_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/HF_params_prior_7_1.npy')), axis = 0)
    
    np.save('MF_param/saved_param/HF_params_0_0.npy', HF_params_0_0)
    np.save('MF_param/saved_param/HF_params_0_1.npy', HF_params_0_1)
    np.save('MF_param/saved_param/HF_params_1_0.npy', HF_params_1_0)
    np.save('MF_param/saved_param/HF_params_1_1.npy', HF_params_1_1)
    np.save('MF_param/saved_param/HF_params_2_0.npy', HF_params_2_0)
    np.save('MF_param/saved_param/HF_params_2_1.npy', HF_params_2_1)
    np.save('MF_param/saved_param/HF_params_3_0.npy', HF_params_3_0)
    np.save('MF_param/saved_param/HF_params_3_1.npy', HF_params_3_1)
    np.save('MF_param/saved_param/HF_params_4_0.npy', HF_params_4_0)
    np.save('MF_param/saved_param/HF_params_4_1.npy', HF_params_4_1)
    np.save('MF_param/saved_param/HF_params_5_0.npy', HF_params_5_0)
    np.save('MF_param/saved_param/HF_params_5_1.npy', HF_params_5_1)
    np.save('MF_param/saved_param/HF_params_6_0.npy', HF_params_6_0)
    np.save('MF_param/saved_param/HF_params_6_1.npy', HF_params_6_1)
    np.save('MF_param/saved_param/HF_params_7_0.npy', HF_params_7_0)
    np.save('MF_param/saved_param/HF_params_7_1.npy', HF_params_7_1)
    
    np.save('MF_param/saved_param/HF_params_prior_0_0.npy', HF_params_prior_0_0)
    np.save('MF_param/saved_param/HF_params_prior_0_1.npy', HF_params_prior_0_1)
    np.save('MF_param/saved_param/HF_params_prior_1_0.npy', HF_params_prior_1_0)
    np.save('MF_param/saved_param/HF_params_prior_1_1.npy', HF_params_prior_1_1)
    np.save('MF_param/saved_param/HF_params_prior_2_0.npy', HF_params_prior_2_0)
    np.save('MF_param/saved_param/HF_params_prior_2_1.npy', HF_params_prior_2_1)
    np.save('MF_param/saved_param/HF_params_prior_3_0.npy', HF_params_prior_3_0)
    np.save('MF_param/saved_param/HF_params_prior_3_1.npy', HF_params_prior_3_1)
    np.save('MF_param/saved_param/HF_params_prior_4_0.npy', HF_params_prior_4_0)
    np.save('MF_param/saved_param/HF_params_prior_4_1.npy', HF_params_prior_4_1)
    np.save('MF_param/saved_param/HF_params_prior_5_0.npy', HF_params_prior_5_0)
    np.save('MF_param/saved_param/HF_params_prior_5_1.npy', HF_params_prior_5_1)
    np.save('MF_param/saved_param/HF_params_prior_6_0.npy', HF_params_prior_6_0)
    np.save('MF_param/saved_param/HF_params_prior_6_1.npy', HF_params_prior_6_1)
    np.save('MF_param/saved_param/HF_params_prior_7_0.npy', HF_params_prior_7_0)
    np.save('MF_param/saved_param/HF_params_prior_7_1.npy', HF_params_prior_7_1)

if is_LF == 1:
    LF_params_0_0 = np.load('MF_param/MF_param_0/LF_params_0_0.npy')
    LF_params_0_1 = np.load('MF_param/MF_param_0/LF_params_0_1.npy')
    LF_params_1_0 = np.load('MF_param/MF_param_0/LF_params_1_0.npy')
    LF_params_1_1 = np.load('MF_param/MF_param_0/LF_params_1_1.npy')
    LF_params_2_0 = np.load('MF_param/MF_param_0/LF_params_2_0.npy')
    LF_params_2_1 = np.load('MF_param/MF_param_0/LF_params_2_1.npy')
    LF_params_3_0 = np.load('MF_param/MF_param_0/LF_params_3_0.npy')
    LF_params_3_1 = np.load('MF_param/MF_param_0/LF_params_3_1.npy')
    LF_params_4_0 = np.load('MF_param/MF_param_0/LF_params_4_0.npy')
    LF_params_4_1 = np.load('MF_param/MF_param_0/LF_params_4_1.npy')
    LF_params_5_0 = np.load('MF_param/MF_param_0/LF_params_5_0.npy')
    LF_params_5_1 = np.load('MF_param/MF_param_0/LF_params_5_1.npy')
    LF_params_6_0 = np.load('MF_param/MF_param_0/LF_params_6_0.npy')
    LF_params_6_1 = np.load('MF_param/MF_param_0/LF_params_6_1.npy')
    LF_params_7_0 = np.load('MF_param/MF_param_0/LF_params_7_0.npy')
    LF_params_7_1 = np.load('MF_param/MF_param_0/LF_params_7_1.npy')
    
    LF_params_prior_0_0 = np.load('MF_param/MF_param_0/LF_params_prior_0_0.npy')
    LF_params_prior_0_1 = np.load('MF_param/MF_param_0/LF_params_prior_0_1.npy')
    LF_params_prior_1_0 = np.load('MF_param/MF_param_0/LF_params_prior_1_0.npy')
    LF_params_prior_1_1 = np.load('MF_param/MF_param_0/LF_params_prior_1_1.npy')
    LF_params_prior_2_0 = np.load('MF_param/MF_param_0/LF_params_prior_2_0.npy')
    LF_params_prior_2_1 = np.load('MF_param/MF_param_0/LF_params_prior_2_1.npy')
    LF_params_prior_3_0 = np.load('MF_param/MF_param_0/LF_params_prior_3_0.npy')
    LF_params_prior_3_1 = np.load('MF_param/MF_param_0/LF_params_prior_3_1.npy')
    LF_params_prior_4_0 = np.load('MF_param/MF_param_0/LF_params_prior_4_0.npy')
    LF_params_prior_4_1 = np.load('MF_param/MF_param_0/LF_params_prior_4_1.npy')
    LF_params_prior_5_0 = np.load('MF_param/MF_param_0/LF_params_prior_5_0.npy')
    LF_params_prior_5_1 = np.load('MF_param/MF_param_0/LF_params_prior_5_1.npy')
    LF_params_prior_6_0 = np.load('MF_param/MF_param_0/LF_params_prior_6_0.npy')
    LF_params_prior_6_1 = np.load('MF_param/MF_param_0/LF_params_prior_6_1.npy')
    LF_params_prior_7_0 = np.load('MF_param/MF_param_0/LF_params_prior_7_0.npy')
    LF_params_prior_7_1 = np.load('MF_param/MF_param_0/LF_params_prior_7_1.npy')
    
    for i in range(N_param_gp-1):
        
        print(i)
        LF_params_0_0 = np.concatenate( (LF_params_0_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_0_0.npy')), axis = 0)
        LF_params_0_1 = np.concatenate( (LF_params_0_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_0_1.npy')), axis = 0)
        LF_params_1_0 = np.concatenate( (LF_params_1_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_1_0.npy')), axis = 0)
        LF_params_1_1 = np.concatenate( (LF_params_1_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_1_1.npy')), axis = 0)
        LF_params_2_0 = np.concatenate( (LF_params_2_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_2_0.npy')), axis = 0)
        LF_params_2_1 = np.concatenate( (LF_params_2_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_2_1.npy')), axis = 0)
        LF_params_3_0 = np.concatenate( (LF_params_3_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_3_0.npy')), axis = 0)
        LF_params_3_1 = np.concatenate( (LF_params_3_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_3_1.npy')), axis = 0)
        LF_params_4_0 = np.concatenate( (LF_params_4_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_4_0.npy')), axis = 0)
        LF_params_4_1 = np.concatenate( (LF_params_4_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_4_1.npy')), axis = 0)
        LF_params_5_0 = np.concatenate( (LF_params_5_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_5_0.npy')), axis = 0)
        LF_params_5_1 = np.concatenate( (LF_params_5_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_5_1.npy')), axis = 0)    
        LF_params_6_0 = np.concatenate( (LF_params_6_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_6_0.npy')), axis = 0)
        LF_params_6_1 = np.concatenate( (LF_params_6_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_6_1.npy')), axis = 0)    
        LF_params_7_0 = np.concatenate( (LF_params_7_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_7_0.npy')), axis = 0)
        LF_params_7_1 = np.concatenate( (LF_params_7_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_7_1.npy')), axis = 0)
    
        LF_params_prior_0_0 = np.concatenate( (LF_params_prior_0_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_0_0.npy')), axis = 0)
        LF_params_prior_0_1 = np.concatenate( (LF_params_prior_0_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_0_1.npy')), axis = 0)
        LF_params_prior_1_0 = np.concatenate( (LF_params_prior_1_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_1_0.npy')), axis = 0)
        LF_params_prior_1_1 = np.concatenate( (LF_params_prior_1_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_1_1.npy')), axis = 0)        
        LF_params_prior_2_0 = np.concatenate( (LF_params_prior_2_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_2_0.npy')), axis = 0)
        LF_params_prior_2_1 = np.concatenate( (LF_params_prior_2_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_2_1.npy')), axis = 0)        
        LF_params_prior_3_0 = np.concatenate( (LF_params_prior_3_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_3_0.npy')), axis = 0)
        LF_params_prior_3_1 = np.concatenate( (LF_params_prior_3_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_3_1.npy')), axis = 0)        
        LF_params_prior_4_0 = np.concatenate( (LF_params_prior_4_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_4_0.npy')), axis = 0)
        LF_params_prior_4_1 = np.concatenate( (LF_params_prior_4_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_4_1.npy')), axis = 0)        
        LF_params_prior_5_0 = np.concatenate( (LF_params_prior_5_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_5_0.npy')), axis = 0)
        LF_params_prior_5_1 = np.concatenate( (LF_params_prior_5_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_5_1.npy')), axis = 0)        
        LF_params_prior_6_0 = np.concatenate( (LF_params_prior_6_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_6_0.npy')), axis = 0)
        LF_params_prior_6_1 = np.concatenate( (LF_params_prior_6_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_6_1.npy')), axis = 0)        
        LF_params_prior_7_0 = np.concatenate( (LF_params_prior_7_0,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_7_0.npy')), axis = 0)
        LF_params_prior_7_1 = np.concatenate( (LF_params_prior_7_1,
                                         np.load('MF_param/MF_param_'+str(i+1)+'/LF_params_prior_7_1.npy')), axis = 0)
    
    np.save('MF_param/saved_param/LF_params_0_0.npy', LF_params_0_0)
    np.save('MF_param/saved_param/LF_params_0_1.npy', LF_params_0_1)
    np.save('MF_param/saved_param/LF_params_1_0.npy', LF_params_1_0)
    np.save('MF_param/saved_param/LF_params_1_1.npy', LF_params_1_1)
    np.save('MF_param/saved_param/LF_params_2_0.npy', LF_params_2_0)
    np.save('MF_param/saved_param/LF_params_2_1.npy', LF_params_2_1)
    np.save('MF_param/saved_param/LF_params_3_0.npy', LF_params_3_0)
    np.save('MF_param/saved_param/LF_params_3_1.npy', LF_params_3_1)
    np.save('MF_param/saved_param/LF_params_4_0.npy', LF_params_4_0)
    np.save('MF_param/saved_param/LF_params_4_1.npy', LF_params_4_1)
    np.save('MF_param/saved_param/LF_params_5_0.npy', LF_params_5_0)
    np.save('MF_param/saved_param/LF_params_5_1.npy', LF_params_5_1)
    np.save('MF_param/saved_param/LF_params_6_0.npy', LF_params_6_0)
    np.save('MF_param/saved_param/LF_params_6_1.npy', LF_params_6_1)
    np.save('MF_param/saved_param/LF_params_7_0.npy', LF_params_7_0)
    np.save('MF_param/saved_param/LF_params_7_1.npy', LF_params_7_1)
    
    np.save('MF_param/saved_param/LF_params_prior_0_0.npy', LF_params_prior_0_0)
    np.save('MF_param/saved_param/LF_params_prior_0_1.npy', LF_params_prior_0_1)
    np.save('MF_param/saved_param/LF_params_prior_1_0.npy', LF_params_prior_1_0)
    np.save('MF_param/saved_param/LF_params_prior_1_1.npy', LF_params_prior_1_1)
    np.save('MF_param/saved_param/LF_params_prior_2_0.npy', LF_params_prior_2_0)
    np.save('MF_param/saved_param/LF_params_prior_2_1.npy', LF_params_prior_2_1)
    np.save('MF_param/saved_param/LF_params_prior_3_0.npy', LF_params_prior_3_0)
    np.save('MF_param/saved_param/LF_params_prior_3_1.npy', LF_params_prior_3_1)
    np.save('MF_param/saved_param/LF_params_prior_4_0.npy', LF_params_prior_4_0)
    np.save('MF_param/saved_param/LF_params_prior_4_1.npy', LF_params_prior_4_1)
    np.save('MF_param/saved_param/LF_params_prior_5_0.npy', LF_params_prior_5_0)
    np.save('MF_param/saved_param/LF_params_prior_5_1.npy', LF_params_prior_5_1)
    np.save('MF_param/saved_param/LF_params_prior_6_0.npy', LF_params_prior_6_0)
    np.save('MF_param/saved_param/LF_params_prior_6_1.npy', LF_params_prior_6_1)
    np.save('MF_param/saved_param/LF_params_prior_7_0.npy', LF_params_prior_7_0)
    np.save('MF_param/saved_param/LF_params_prior_7_1.npy', LF_params_prior_7_1)

if is_SF == 1:
    SF_params_0_0 = np.load('SF_param/SF_param_0/params_0_0.npy')
    SF_params_0_1 = np.load('SF_param/SF_param_0/params_0_1.npy')
    SF_params_1_0 = np.load('SF_param/SF_param_0/params_1_0.npy')
    SF_params_1_1 = np.load('SF_param/SF_param_0/params_1_1.npy')
    SF_params_2_0 = np.load('SF_param/SF_param_0/params_2_0.npy')
    SF_params_2_1 = np.load('SF_param/SF_param_0/params_2_1.npy')
    SF_params_3_0 = np.load('SF_param/SF_param_0/params_3_0.npy')
    SF_params_3_1 = np.load('SF_param/SF_param_0/params_3_1.npy')
    SF_params_4_0 = np.load('SF_param/SF_param_0/params_4_0.npy')
    SF_params_4_1 = np.load('SF_param/SF_param_0/params_4_1.npy')
    SF_params_5_0 = np.load('SF_param/SF_param_0/params_5_0.npy')
    SF_params_5_1 = np.load('SF_param/SF_param_0/params_5_1.npy')
    SF_params_6_0 = np.load('SF_param/SF_param_0/params_6_0.npy')
    SF_params_6_1 = np.load('SF_param/SF_param_0/params_6_1.npy')
    SF_params_7_0 = np.load('SF_param/SF_param_0/params_7_0.npy')
    SF_params_7_1 = np.load('SF_param/SF_param_0/params_7_1.npy')
    
    SF_params_prior_0_0 = np.load('SF_param/SF_param_0/params_prior_0_0.npy')
    SF_params_prior_0_1 = np.load('SF_param/SF_param_0/params_prior_0_1.npy')
    SF_params_prior_1_0 = np.load('SF_param/SF_param_0/params_prior_1_0.npy')
    SF_params_prior_1_1 = np.load('SF_param/SF_param_0/params_prior_1_1.npy')
    SF_params_prior_2_0 = np.load('SF_param/SF_param_0/params_prior_2_0.npy')
    SF_params_prior_2_1 = np.load('SF_param/SF_param_0/params_prior_2_1.npy')
    SF_params_prior_3_0 = np.load('SF_param/SF_param_0/params_prior_3_0.npy')
    SF_params_prior_3_1 = np.load('SF_param/SF_param_0/params_prior_3_1.npy')
    SF_params_prior_4_0 = np.load('SF_param/SF_param_0/params_prior_4_0.npy')
    SF_params_prior_4_1 = np.load('SF_param/SF_param_0/params_prior_4_1.npy')
    SF_params_prior_5_0 = np.load('SF_param/SF_param_0/params_prior_5_0.npy')
    SF_params_prior_5_1 = np.load('SF_param/SF_param_0/params_prior_5_1.npy')
    SF_params_prior_6_0 = np.load('SF_param/SF_param_0/params_prior_6_0.npy')
    SF_params_prior_6_1 = np.load('SF_param/SF_param_0/params_prior_6_1.npy')
    SF_params_prior_7_0 = np.load('SF_param/SF_param_0/params_prior_7_0.npy')
    SF_params_prior_7_1 = np.load('SF_param/SF_param_0/params_prior_7_1.npy')
    
    for i in range(N_param_gp-1):
        
        print(i)
        SF_params_0_0 = np.concatenate( (SF_params_0_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_0_0.npy')), axis = 0)
        SF_params_0_1 = np.concatenate( (SF_params_0_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_0_1.npy')), axis = 0)
        SF_params_1_0 = np.concatenate( (SF_params_1_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_1_0.npy')), axis = 0)
        SF_params_1_1 = np.concatenate( (SF_params_1_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_1_1.npy')), axis = 0)
        SF_params_2_0 = np.concatenate( (SF_params_2_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_2_0.npy')), axis = 0)
        SF_params_2_1 = np.concatenate( (SF_params_2_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_2_1.npy')), axis = 0)
        SF_params_3_0 = np.concatenate( (SF_params_3_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_3_0.npy')), axis = 0)
        SF_params_3_1 = np.concatenate( (SF_params_3_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_3_1.npy')), axis = 0)
        SF_params_4_0 = np.concatenate( (SF_params_4_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_4_0.npy')), axis = 0)
        SF_params_4_1 = np.concatenate( (SF_params_4_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_4_1.npy')), axis = 0)
        SF_params_5_0 = np.concatenate( (SF_params_5_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_5_0.npy')), axis = 0)
        SF_params_5_1 = np.concatenate( (SF_params_5_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_5_1.npy')), axis = 0)    
        SF_params_6_0 = np.concatenate( (SF_params_6_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_6_0.npy')), axis = 0)
        SF_params_6_1 = np.concatenate( (SF_params_6_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_6_1.npy')), axis = 0)    
        SF_params_7_0 = np.concatenate( (SF_params_7_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_7_0.npy')), axis = 0)
        SF_params_7_1 = np.concatenate( (SF_params_7_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_7_1.npy')), axis = 0)
    
        SF_params_prior_0_0 = np.concatenate( (SF_params_prior_0_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_0_0.npy')), axis = 0)
        SF_params_prior_0_1 = np.concatenate( (SF_params_prior_0_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_0_1.npy')), axis = 0)
        SF_params_prior_1_0 = np.concatenate( (SF_params_prior_1_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_1_0.npy')), axis = 0)
        SF_params_prior_1_1 = np.concatenate( (SF_params_prior_1_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_1_1.npy')), axis = 0)        
        SF_params_prior_2_0 = np.concatenate( (SF_params_prior_2_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_2_0.npy')), axis = 0)
        SF_params_prior_2_1 = np.concatenate( (SF_params_prior_2_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_2_1.npy')), axis = 0)        
        SF_params_prior_3_0 = np.concatenate( (SF_params_prior_3_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_3_0.npy')), axis = 0)
        SF_params_prior_3_1 = np.concatenate( (SF_params_prior_3_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_3_1.npy')), axis = 0)        
        SF_params_prior_4_0 = np.concatenate( (SF_params_prior_4_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_4_0.npy')), axis = 0)
        SF_params_prior_4_1 = np.concatenate( (SF_params_prior_4_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_4_1.npy')), axis = 0)        
        SF_params_prior_5_0 = np.concatenate( (SF_params_prior_5_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_5_0.npy')), axis = 0)
        SF_params_prior_5_1 = np.concatenate( (SF_params_prior_5_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_5_1.npy')), axis = 0)        
        SF_params_prior_6_0 = np.concatenate( (SF_params_prior_6_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_6_0.npy')), axis = 0)
        SF_params_prior_6_1 = np.concatenate( (SF_params_prior_6_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_6_1.npy')), axis = 0)        
        SF_params_prior_7_0 = np.concatenate( (SF_params_prior_7_0,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_7_0.npy')), axis = 0)
        SF_params_prior_7_1 = np.concatenate( (SF_params_prior_7_1,
                                         np.load('SF_param/SF_param_'+str(i+1)+'/params_prior_7_1.npy')), axis = 0)
    
    np.save('SF_param/saved_param/SF_params_0_0.npy', SF_params_0_0)
    np.save('SF_param/saved_param/SF_params_0_1.npy', SF_params_0_1)
    np.save('SF_param/saved_param/SF_params_1_0.npy', SF_params_1_0)
    np.save('SF_param/saved_param/SF_params_1_1.npy', SF_params_1_1)
    np.save('SF_param/saved_param/SF_params_2_0.npy', SF_params_2_0)
    np.save('SF_param/saved_param/SF_params_2_1.npy', SF_params_2_1)
    np.save('SF_param/saved_param/SF_params_3_0.npy', SF_params_3_0)
    np.save('SF_param/saved_param/SF_params_3_1.npy', SF_params_3_1)
    np.save('SF_param/saved_param/SF_params_4_0.npy', SF_params_4_0)
    np.save('SF_param/saved_param/SF_params_4_1.npy', SF_params_4_1)
    np.save('SF_param/saved_param/SF_params_5_0.npy', SF_params_5_0)
    np.save('SF_param/saved_param/SF_params_5_1.npy', SF_params_5_1)
    np.save('SF_param/saved_param/SF_params_6_0.npy', SF_params_6_0)
    np.save('SF_param/saved_param/SF_params_6_1.npy', SF_params_6_1)
    np.save('SF_param/saved_param/SF_params_7_0.npy', SF_params_7_0)
    np.save('SF_param/saved_param/SF_params_7_1.npy', SF_params_7_1)
    
    np.save('SF_param/saved_param/SF_params_prior_0_0.npy', SF_params_prior_0_0)
    np.save('SF_param/saved_param/SF_params_prior_0_1.npy', SF_params_prior_0_1)
    np.save('SF_param/saved_param/SF_params_prior_1_0.npy', SF_params_prior_1_0)
    np.save('SF_param/saved_param/SF_params_prior_1_1.npy', SF_params_prior_1_1)
    np.save('SF_param/saved_param/SF_params_prior_2_0.npy', SF_params_prior_2_0)
    np.save('SF_param/saved_param/SF_params_prior_2_1.npy', SF_params_prior_2_1)
    np.save('SF_param/saved_param/SF_params_prior_3_0.npy', SF_params_prior_3_0)
    np.save('SF_param/saved_param/SF_params_prior_3_1.npy', SF_params_prior_3_1)
    np.save('SF_param/saved_param/SF_params_prior_4_0.npy', SF_params_prior_4_0)
    np.save('SF_param/saved_param/SF_params_prior_4_1.npy', SF_params_prior_4_1)
    np.save('SF_param/saved_param/SF_params_prior_5_0.npy', SF_params_prior_5_0)
    np.save('SF_param/saved_param/SF_params_prior_5_1.npy', SF_params_prior_5_1)
    np.save('SF_param/saved_param/SF_params_prior_6_0.npy', SF_params_prior_6_0)
    np.save('SF_param/saved_param/SF_params_prior_6_1.npy', SF_params_prior_6_1)
    np.save('SF_param/saved_param/SF_params_prior_7_0.npy', SF_params_prior_7_0)
    np.save('SF_param/saved_param/SF_params_prior_7_1.npy', SF_params_prior_7_1)