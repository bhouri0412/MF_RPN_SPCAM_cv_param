#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:59:59 2023

@author: mohamedazizbhouri
"""
from matplotlib import pyplot as plt
import numpy as onp

from scipy.stats import kde

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 36,
                     'lines.linewidth': 2,
                     'axes.labelsize': 36,
                     'axes.titlesize': 36,
                     'xtick.labelsize': 36,
                     'ytick.labelsize': 36,
                     'legend.fontsize': 36,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex"
                     })
plt.rcParams['agg.path.chunksize'] = 20000

lat = 96
lon = 144
nt_total = 365*24

dim_y = 48
dim_heat = 26
dim_moist = 22
mu_error_out = onp.concatenate((onp.zeros((1,dim_heat),dtype=onp.float32),
                                onp.zeros((1,dim_moist),dtype=onp.float32)),axis=1)
sigma_error_out = onp.concatenate((1/1004.6*onp.ones((1,dim_heat),dtype=onp.float32),
                                    1/2.26e6*onp.ones((1,dim_moist),dtype=onp.float32)),axis=1)
mu_error_out = onp.array(mu_error_out,dtype=onp.float64) 
sigma_error_out = onp.array(sigma_error_out,dtype=onp.float64) 

is_SF = 0
is_MF = 0 
is_LF = 1

ilist = [14, 18, 21, 36, 40, 43] # indices for tendencies at pressure levels 259, 494 and 761 hPa
ipress = [259, 494, 761, 259, 294, 761]
itend = ['heat', 'heat', 'heat', 'moist', 'moist', 'moist']
nbins = 100
epsilon = 0.125
fact_time = 4
fact_lon = 8
fact_lat = 8

onp.random.seed(1234)

import time

if is_MF == 1:
    print('loading')
    mean_rpn_MF = onp.load('MF_param/mean_RPN_MF_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    mean_rpn_MF = onp.array(mean_rpn_MF,dtype=onp.float64) 
    mean_rpn_MF = onp.reshape( mean_rpn_MF, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    mean_rpn_MF = mean_rpn_MF.T
    mean_rpn_MF = (mean_rpn_MF-mu_error_out) / sigma_error_out
    
    std_rpn_MF = onp.load('MF_param/std_RPN_MF_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    std_rpn_MF = onp.array(std_rpn_MF,dtype=onp.float64) 
    std_rpn_MF = onp.reshape( std_rpn_MF, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    std_rpn_MF = std_rpn_MF.T
    std_rpn_MF = std_rpn_MF / sigma_error_out
    
    test_yH = onp.load('data_SPCAM5_4K/all_outputs_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    test_yH = onp.array(test_yH,dtype=onp.float64) 
    test_yH = onp.reshape( test_yH, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    test_yH = test_yH.T
    test_yH = (test_yH-mu_error_out) / sigma_error_out
    
    for ii in range(len(ilist)):
        i = ilist[ii]
        print(i)
        err_MF = onp.abs(mean_rpn_MF[:,i]-test_yH[:,i])
        x = err_MF
        y = std_rpn_MF[:,i]
        Ntot = x.shape[0]
        y = y[~onp.isnan(x)]
        x = x[~onp.isnan(x)]
        x = x[~onp.isnan(y)]
        y = y[~onp.isnan(y)]
        ind = onp.argsort(x)
        x = x[ind]
        y = y[ind]
        x = x[:Ntot-int(epsilon*Ntot)]
        y = y[:Ntot-int(epsilon*Ntot)]
        ind = onp.argsort(y)
        x = x[ind]
        y = y[ind]
        x = onp.sqrt(x[:Ntot-2*int(epsilon*Ntot)])
        y = y[:Ntot-2*int(epsilon*Ntot)]
        
        tt = time.time()
        k = kde.gaussian_kde([x,y])
        xi, yi = onp.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(onp.vstack([xi.flatten(), yi.flatten()]))
        zi = zi.reshape(xi.shape)
        print('gaussian_kde', time.time()-tt )
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        plt.pcolormesh(xi, yi, zi, shading='gouraud', cmap=plt.cm.jet)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.colorbar()
        ax.set_xlabel('Error')
        ax.set_ylabel(r'$\sigma_M$')
        plt.savefig('uncertainty_density_plots/MF_uncert_dens_'+itend[ii]+'_'+str(ipress[ii])+'_hPa.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            
if is_LF == 1:
    print('loading')
    mean_rpn_MF = onp.load('MF_param/mean_RPN_LF_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    mean_rpn_MF = onp.array(mean_rpn_MF,dtype=onp.float64) 
    mean_rpn_MF = onp.reshape( mean_rpn_MF, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    mean_rpn_MF = mean_rpn_MF.T
    mean_rpn_MF = (mean_rpn_MF-mu_error_out) / sigma_error_out
    
    std_rpn_MF = onp.load('MF_param/std_RPN_LF_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    std_rpn_MF = onp.array(std_rpn_MF,dtype=onp.float64) 
    std_rpn_MF = onp.reshape( std_rpn_MF, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    std_rpn_MF = std_rpn_MF.T
    std_rpn_MF = std_rpn_MF / sigma_error_out
    
    test_yH = onp.load('data_SPCAM5_4K/all_outputs_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    test_yH = onp.array(test_yH,dtype=onp.float64) 
    test_yH = onp.reshape( test_yH, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    test_yH = test_yH.T
    test_yH = (test_yH-mu_error_out) / sigma_error_out
    
    for ii in range(len(ilist)):
        i = ilist[ii]
        print(i)
        err_MF = onp.abs(mean_rpn_MF[:,i]-test_yH[:,i])
        x = err_MF
        y = std_rpn_MF[:,i]
        Ntot = x.shape[0]
        y = y[~onp.isnan(x)]
        x = x[~onp.isnan(x)]
        x = x[~onp.isnan(y)]
        y = y[~onp.isnan(y)]
        ind = onp.argsort(x)
        x = x[ind]
        y = y[ind]
        x = x[:Ntot-int(epsilon*Ntot)]
        y = y[:Ntot-int(epsilon*Ntot)]
        ind = onp.argsort(y)
        x = x[ind]
        y = y[ind]
        x = onp.sqrt(x[:Ntot-2*int(epsilon*Ntot)])
        y = y[:Ntot-2*int(epsilon*Ntot)]
        
        tt = time.time()
        k = kde.gaussian_kde([x,y])
        xi, yi = onp.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(onp.vstack([xi.flatten(), yi.flatten()]))
        zi = zi.reshape(xi.shape)
        print('gaussian_kde', time.time()-tt )
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        plt.pcolormesh(xi, yi, zi, shading='gouraud', cmap=plt.cm.jet)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.colorbar()
        ax.set_xlabel('Error')
        ax.set_ylabel(r'$\sigma_M$')
        plt.savefig('uncertainty_density_plots/LF_uncert_dens_'+itend[ii]+'_'+str(ipress[ii])+'_hPa.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
        
if is_SF == 1:
    print('loading')
    mean_rpn = onp.load('SF_param/mean_RPN_SF_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    mean_rpn = onp.array(mean_rpn,dtype=onp.float64) 
    mean_rpn = onp.reshape( mean_rpn, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    mean_rpn = mean_rpn.T
    mean_rpn = (mean_rpn-mu_error_out) / sigma_error_out
    
    std_rpn = onp.load('SF_param/std_RPN_SF_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    std_rpn = onp.array(std_rpn,dtype=onp.float64) 
    std_rpn = onp.reshape( std_rpn, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    std_rpn = std_rpn.T
    std_rpn = std_rpn / sigma_error_out
    
    test_yH = onp.load('data_SPCAM5_4K/all_outputs_reshaped.npy')[:,::fact_time,::fact_lat,::fact_lon]
    test_yH = onp.array(test_yH,dtype=onp.float64) 
    test_yH = onp.reshape( test_yH, ( dim_y, (nt_total*lat*lon)//(fact_time*fact_lon*fact_lat) ) )
    test_yH = test_yH.T
    test_yH = (test_yH-mu_error_out) / sigma_error_out
    
    for ii in range(len(ilist)):
        i = ilist[ii]
        print(i)
        err = onp.abs(mean_rpn[:,i]-test_yH[:,i])
        x = err
        y = std_rpn[:,i]
        Ntot = x.shape[0]
        y = y[~onp.isnan(x)]
        x = x[~onp.isnan(x)]
        x = x[~onp.isnan(y)]
        y = y[~onp.isnan(y)]
        ind = onp.argsort(x)
        x = x[ind]
        y = y[ind]
        x = x[:Ntot-int(epsilon*Ntot)]
        y = y[:Ntot-int(epsilon*Ntot)]
        ind = onp.argsort(y)
        x = x[ind]
        y = y[ind]
        x = onp.sqrt(x[:Ntot-2*int(epsilon*Ntot)])
        y = y[:Ntot-2*int(epsilon*Ntot)]
                 
        tt = time.time()
        k = kde.gaussian_kde([x,y])
        xi, yi = onp.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(onp.vstack([xi.flatten(), yi.flatten()]))
        zi = zi.reshape(xi.shape)
        print('gaussian_kde', time.time()-tt )
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        plt.pcolormesh(xi, yi, zi, shading='gouraud', cmap=plt.cm.jet)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.colorbar()
        ax.set_xlabel('Error')
        ax.set_ylabel(r'$\sigma_M$')
        plt.savefig('uncertainty_density_plots/SF_uncert_dens_'+itend[ii]+'_'+str(ipress[ii])+'_hPa.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
        