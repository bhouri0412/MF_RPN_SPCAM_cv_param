#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:59:59 2023

@author: mohamedazizbhouri
"""
import numpy as onp

lat = 96
lon = 144
dim_y = 48

is_create_Npts_per_file = 0

is_det = 0
is_rpn_SF = 0
is_rpn_MF = 0
is_test_data = 0
is_LF = 1

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

if is_create_Npts_per_file == 1:
    print('create Npts_per_file')
    Npts_per_file = []
    for i in range(len(test_SPCAM)):
        Npts_per_file.append( onp.load('data_SPCAM5_4K/inputs_'+test_SPCAM[i]+'.npy').shape[0] )
    Npts_per_file = onp.array(Npts_per_file)
    onp.save('data_SPCAM5_4K/Npts_per_file_test.npy',Npts_per_file)
    print('Npts_per_file created')
else:
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
    
if is_det == 1:
    print('load and reshape det')
    n_run = 128
    case_var = 'all'
    samples_test_H = reshape_loc_onp( onp.concatenate( (onp.load('SF_param/SF_param_det/test_pred_1.npy')[0,:,:],
                                                        onp.load('SF_param/SF_param_det/test_pred_2.npy')[0,:,:]), axis=0 ), dim_y )
    onp.save('SF_param/SF_param_det/test_pred_reshaped.npy', samples_test_H)

if is_LF == 1:
    print('load and reshape det')
    
    mean_rpn_LF = reshape_loc_onp( onp.concatenate( (onp.load('MF_param/mean_RPN_LF_1.npy'),
                                                  onp.load('MF_param/mean_RPN_LF_2.npy')), axis=0), dim_y ) # dim_y x nt x lon x lat
    std_rpn_LF = reshape_loc_onp( onp.concatenate( (onp.load('MF_param/std_RPN_LF_1.npy'),
                                                  onp.load('MF_param/std_RPN_LF_2.npy')), axis=0), dim_y ) # dim_y x nt x lon x lat
    
    onp.save('MF_param/mean_RPN_LF_reshaped.npy', mean_rpn_LF)
    onp.save('MF_param/std_RPN_LF_reshaped.npy', std_rpn_LF)
        
if is_rpn_SF == 1:
    print('load and reshape rpn SF')
    mean_rpn = reshape_loc_onp( onp.load('SF_param/mean_RPN_SF.npy'), dim_y ) # dim_y x nt x lon x lat
    std_rpn = reshape_loc_onp( onp.load('SF_param/std_RPN_SF.npy'), dim_y )
    
    onp.save('SF_param/mean_RPN_SF_reshaped.npy', mean_rpn)
    onp.save('SF_param/std_RPN_SF_reshaped.npy', std_rpn)
    
if is_rpn_MF == 1:
    print('load and reshape rpn MF')
    mean_rpn_MF = reshape_loc_onp( onp.concatenate( (onp.load('MF_param/mean_RPN_MF_1.npy'),
                                                  onp.load('MF_param/mean_RPN_MF_2.npy')), axis=0), dim_y ) # dim_y x nt x lon x lat
    std_rpn_MF = reshape_loc_onp( onp.concatenate( (onp.load('MF_param/std_RPN_MF_1.npy'),
                                                  onp.load('MF_param/std_RPN_MF_2.npy')), axis=0), dim_y ) # dim_y x nt x lon x lat
    
    onp.save('MF_param/mean_RPN_MF_reshaped.npy', mean_rpn_MF)
    onp.save('MF_param/std_RPN_MF_reshaped.npy', std_rpn_MF)
    
if is_test_data == 1:
    print('load and reshape test')

    n_remove = 4
    ind_output_heat = onp.arange(26) 
    ind_output_moist = n_remove+onp.arange(26-n_remove)
    test_yH = onp.concatenate((onp.load('data_SPCAM5_4K/all_outputs_heat.npy')[:,ind_output_heat],
                               onp.load('data_SPCAM5_4K/all_outputs_moist.npy')[:,ind_output_moist]),axis=1)
    
    test_yH = reshape_loc_onp(test_yH, dim_y)
    
    onp.save('data_SPCAM5_4K/all_outputs_reshaped.npy', test_yH)
    