#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:59:59 2023

@author: mohamedazizbhouri
"""
from matplotlib import pyplot as plt
import numpy as onp

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
plt.close('all')

lat = 96
lon = 144

is_glob_err = 0
is_MAE = 0
is_r2 = 1

dim_y = 48
dim_heat = 26
dim_moist = 22

mu_error_out = onp.concatenate((onp.zeros((1,dim_heat),dtype=onp.float32),
                                onp.zeros((1,dim_moist),dtype=onp.float32)),axis=1)
mu_error_out = mu_error_out.T[:,:,None,None]

sigma_error_out = onp.concatenate((1/1004.6*onp.ones((1,dim_heat),dtype=onp.float32),
                                    1/2.26e6*onp.ones((1,dim_moist),dtype=onp.float32)),axis=1)
    
sigma_error_out = sigma_error_out.T[:,:,None,None]

test = (onp.load('data_SPCAM5_4K/all_outputs_reshaped.npy') - mu_error_out) / sigma_error_out
test = onp.array(test,dtype=onp.float64)

n_run = 128
pred_det = (onp.load('SF_param/SF_param_det/test_pred_reshaped.npy') - mu_error_out) / sigma_error_out
pred_det = onp.array(pred_det,dtype=onp.float64)
    
pred_rpn = onp.load('SF_param/mean_RPN_SF_reshaped.npy')
pred_rpn = (pred_rpn - mu_error_out) / sigma_error_out
pred_rpn = onp.array(pred_rpn,dtype=onp.float64)

pred_rpn_MF = onp.load('MF_param/mean_RPN_MF_reshaped.npy')
pred_rpn_MF = (pred_rpn_MF - mu_error_out) / sigma_error_out
pred_rpn_MF = onp.array(pred_rpn_MF,dtype=onp.float64)

pred_rpn_LF = onp.load('MF_param/mean_RPN_LF_reshaped.npy')
pred_rpn_LF = (pred_rpn_LF - mu_error_out) / sigma_error_out
pred_rpn_LF = onp.array(pred_rpn_LF,dtype=onp.float64)

N_dt_day = 24 # we have a dt=1hour
def daily_avg(test):
    test_daily = []
    N_time_steps = test.shape[1]
    for i in range(test.shape[0]):
        test_daily.append( onp.mean( test[i,:,:,:].reshape( (N_time_steps//N_dt_day, N_dt_day, lat, lon) ), axis=1 ) )
    return onp.array(test_daily) # dim_y x N_day x lat x lon
pred_det = daily_avg(pred_det)
pred_rpn = daily_avg(pred_rpn)
pred_rpn_MF = daily_avg(pred_rpn_MF)
pred_rpn_LF = daily_avg(pred_rpn_LF)
test = daily_avg(test)

if is_glob_err == 1:
    MAE_det = onp.mean(onp.abs(pred_det-test),axis=(1,2,3)) # dim_y x nt
    MAE_rpn_SF = onp.mean(onp.abs(pred_rpn-test),axis=(1,2,3)) # dim_y x nt
    MAE_rpn_MF = onp.mean( onp.abs(pred_rpn_MF - test) ,axis=(1,2,3)) 
    MAE_rpn_LF = onp.mean( onp.abs(pred_rpn_LF - test) ,axis=(1,2,3)) 
    
    onp.save('glob_errors/MAE_det.npy',MAE_det)
    onp.save('glob_errors/MAE_rpn_SF.npy',MAE_rpn_SF)
    onp.save('glob_errors/MAE_rpn_MF.npy',MAE_rpn_MF)
    onp.save('glob_errors/MAE_rpn_LF.npy',MAE_rpn_LF)
    
    r2_det = 1 - onp.sum( (test-pred_det)**2, axis=(1,2,3) ) / onp.sum( (test-onp.mean(test,axis=(1,2,3))[:,None,None,None])**2, axis=(1,2,3) )
    r2_rpn = 1 - onp.sum( (test-pred_rpn)**2, axis=(1,2,3) ) / onp.sum( (test-onp.mean(test,axis=(1,2,3))[:,None,None,None])**2, axis=(1,2,3) )
    r2_rpn_MF = 1 - onp.sum( (test-pred_rpn_MF)**2, axis=(1,2,3) ) / onp.sum( (test-onp.mean(test,axis=(1,2,3))[:,None,None,None])**2, axis=(1,2,3) )
    r2_rpn_LF = 1 - onp.sum( (test-pred_rpn_LF)**2, axis=(1,2,3) ) / onp.sum( (test-onp.mean(test,axis=(1,2,3))[:,None,None,None])**2, axis=(1,2,3) )
    
    onp.save('glob_errors/r2_det.npy',r2_det)
    onp.save('glob_errors/r2_rpn_SF.npy',r2_rpn)
    onp.save('glob_errors/r2_rpn_MF.npy',r2_rpn_MF)
    onp.save('glob_errors/r2_rpn_LF.npy',r2_rpn_LF)
    
if is_MAE == 1:
    x_labels = ['02/2003','03/2003','04/2003','05/2003','06/2003','07/2003',
                '08/2003','09/2003','10/2003','11/2003','12/2003','01/2004','02/2004']
    
    print('Compute MAE')
    err_det = onp.mean(onp.abs(pred_det-test),axis=(2,3)) # dim_y x nt
    err_rpn = onp.mean(onp.abs(pred_rpn-test),axis=(2,3)) # dim_y x nt
    err_rpn_MF = onp.mean( onp.abs(pred_rpn_MF - test) ,axis=(2,3)) 
    err_rpn_LF = onp.mean( onp.abs(pred_rpn_LF - test) ,axis=(2,3)) 
    
    for i in range(dim_y):
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(111)
        x_labels = ['02/2003','03/2003','04/2003','05/2003','06/2003','07/2003',
                    '08/2003','09/2003','10/2003','11/2003','12/2003','01/2004','02/2004']
        ax.set_xticks(onp.linspace(0,err_det.shape[1]-1,13))
        ax.set_xticklabels(x_labels, rotation =45, fontsize=22)
        ax.yaxis.set_tick_params(labelsize=22)
        ax.plot(onp.arange(err_det.shape[1]), err_det[i,:], color='blue', linewidth=3, label = "Det. NN")
        ax.plot(onp.arange(err_det.shape[1]), err_rpn[i,:], '--', color='black', linewidth=3, label = "SF-RPN")
        ax.plot(onp.arange(err_det.shape[1]), err_rpn_LF[i,:], '--', color='orange', linewidth=3, label = "LF-RPN")
        ax.plot(onp.arange(err_det.shape[1]), err_rpn_MF[i,:], color='red', linewidth=3, label = "MF-RPN")
        ax.set_ylabel('MAE')
        ax.set_xlabel('Time')
        ax.grid()
        ax.legend(bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)

        if i > 25:
            plt.savefig('temp_plots/moist_MAE_temp_'+str(i-26+4)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            
        else:
            plt.savefig('temp_plots/heat_MAE_temp_'+str(i)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)

if is_r2 == 1:
    x_labels = ['02/2003','03/2003','04/2003','05/2003','06/2003','07/2003',
                '08/2003','09/2003','10/2003','11/2003','12/2003','01/2004','02/2004']
    
    print('Compute R^2')
    r2_det = 1 - onp.sum( (test-pred_det)**2, axis=(2,3) ) / onp.sum( (test-onp.mean(test,axis=(2,3))[:,:,None,None])**2, axis=(2,3) )
    r2_rpn_SF = 1 - onp.sum( (test-pred_rpn)**2, axis=(2,3) ) / onp.sum( (test-onp.mean(test,axis=(2,3))[:,:,None,None])**2, axis=(2,3) )
    r2_rpn_MF = 1 - onp.sum( (test-pred_rpn_MF)**2, axis=(2,3) ) / onp.sum( (test-onp.mean(test,axis=(2,3))[:,:,None,None])**2, axis=(2,3) )
    r2_rpn_LF = 1 - onp.sum( (test-pred_rpn_LF)**2, axis=(2,3) ) / onp.sum( (test-onp.mean(test,axis=(2,3))[:,:,None,None])**2, axis=(2,3) )
    
    for i in range(dim_y):
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(111)
        
        ax.set_xticks(onp.linspace(0,r2_det.shape[1]-1,13))
        ax.set_xticklabels(x_labels, rotation =45, fontsize=22)
        ax.yaxis.set_tick_params(labelsize=22)
        ax.plot(onp.arange(r2_det.shape[1]), r2_det[i,:], color='blue', linewidth=3, label = "Det. NN")
        ax.plot(onp.arange(r2_det.shape[1]), r2_rpn_SF[i,:], '--', color='black', linewidth=3, label = "SF-RPN")
        ax.plot(onp.arange(r2_det.shape[1]), r2_rpn_LF[i,:], '--', color='orange', linewidth=3, label = "LF-RPN")
        ax.plot(onp.arange(r2_det.shape[1]), r2_rpn_MF[i,:], color='red', linewidth=3, label = "MF-RPN")
        ax.set_ylabel('$R^2$')
        ax.set_xlabel('Time')
        ax.grid()
        ax.legend(bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
        
        if i > 25:
            plt.savefig('temp_plots/moist_r2_temp_'+str(i-26+4)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            
        else:
            plt.savefig('temp_plots/heat_r2_temp_'+str(i)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            
            