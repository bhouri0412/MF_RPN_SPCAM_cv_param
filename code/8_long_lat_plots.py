#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:59:59 2023

@author: mohamedazizbhouri
"""
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16,
                     'lines.linewidth': 2,
                     'axes.labelsize': 20,
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 20,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex"
                     })

lat = 96
lon = 144
x = np.linspace(0, 360-360/144, lon)
y = np.linspace(-90, 90, lat)
X, Y = np.meshgrid(x, y)

from mpl_toolkits.basemap import Basemap

dim_y = 26+22 

MAE_det = np.load('SF_param/SF_param_det/MAE_det_long_lat.npy')
r2_det = np.load('SF_param/SF_param_det/r2_det_long_lat.npy')

MAE_SF = np.load('SF_param/MAE_SF_long_lat.npy')
r2_SF = np.load('SF_param/r2_SF_long_lat.npy')

MAE_MF = np.load('MF_param/MAE_MF_long_lat.npy')
r2_MF = np.load('MF_param/r2_MF_long_lat.npy')

MAE_LF = np.load('MF_param/MAE_LF_long_lat.npy')
r2_LF = np.load('MF_param/r2_LF_long_lat.npy')

dim_heat = 26
dim_moist = 22
mu_error_out = np.concatenate((np.zeros((1,dim_heat),dtype=np.float32),
                               np.zeros((1,dim_moist),dtype=np.float32)),axis=1)
mu_error_out = mu_error_out.T[:,:,None,None]
sigma_error_out = np.concatenate((1/1004.6*np.ones((1,dim_heat),dtype=np.float32),
                                    1/2.26e6*np.ones((1,dim_moist),dtype=np.float32)),axis=1)
sigma_error_out = sigma_error_out.T[:,:,None,None]

print('Create err plots')

is_uncert = 1
is_MAE = 1
is_R2 = 1

if is_uncert == 1: # uncertainty plots
    std_rpn_MF = np.load('MF_param/std_RPN_MF_reshaped.npy')/ sigma_error_out 
    
    std_rpn_MF = np.mean(std_rpn_MF, axis=1)
    for i in range(dim_y):
        fig = plt.figure(figsize=(10.5, 14))        
        ax = fig.add_subplot(211)
        ax.set_title("MAE")
        m = Basemap(projection='robin',lon_0=-180)
        MAE_MF_p = MAE_MF[i,:,:]
        contour_plot = m.pcolormesh(X, Y,MAE_MF_p, latlon = True, cmap='Blues_r')
        m.drawcoastlines(linewidth=2.0, color='0.25')
        m.colorbar(contour_plot)
        
        ax = fig.add_subplot(212)
        ax.set_title("Uncertainty "+r'$\sigma_M$')
        m = Basemap(projection='robin',lon_0=-180)
        std_rpn_MF_p = std_rpn_MF[i,:,:]
        contour_plot = m.pcolormesh(X, Y, std_rpn_MF_p, latlon = True, cmap='Blues_r')
        m.drawcoastlines(linewidth=2.0, color='0.25')
        m.colorbar(contour_plot)
        
        plt.subplots_adjust(hspace=-0.35)
            
        if i > 25:
            plt.savefig('long_lat_uncert_plots/MF_moist_long_lat_uncert_'+str(i-26+4)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            
        else:
            plt.savefig('long_lat_uncert_plots/MF_heat_long_lat_uncert_'+str(i)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)

    std_rpn_SF = np.load('SF_param/std_RPN_SF_reshaped.npy')/ sigma_error_out
    std_rpn_SF = np.mean(std_rpn_SF, axis=1)
    for i in range(dim_y):
        fig = plt.figure(figsize=(10.5, 14))
        ax = fig.add_subplot(211)
        ax.set_title("MAE")
        m = Basemap(projection='robin',lon_0=-180)
        MAE_SF_p = MAE_SF[i,:,:]
        contour_plot = m.pcolormesh(X, Y, MAE_SF_p, latlon = True, cmap='Blues_r')
        m.drawcoastlines(linewidth=2.0, color='0.25')
        m.colorbar(contour_plot)
        ax = fig.add_subplot(212)
        ax.set_title("Uncertainty "+r'$\sigma_M$')
        m = Basemap(projection='robin',lon_0=-180)
        std_rpn_SF_p = std_rpn_SF[i,:,:]
        contour_plot = m.pcolormesh(X, Y, std_rpn_SF_p, latlon = True, cmap='Blues_r')
        m.drawcoastlines(linewidth=2.0, color='0.25')
        m.colorbar(contour_plot)
        plt.subplots_adjust(hspace=-0.35)
        if i > 25:
            plt.savefig('long_lat_uncert_plots/SF_moist_long_lat_uncert_'+str(i-26+4)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            
        else:
            plt.savefig('long_lat_uncert_plots/SF_moist_long_lat_uncert_'+str(i)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)

    std_rpn_LF = np.load('MF_param/std_RPN_LF_reshaped.npy')/ sigma_error_out
    std_rpn_LF = np.mean(std_rpn_LF, axis=1)
    for i in range(dim_y):
        fig = plt.figure(figsize=(10.5, 14))
        ax = fig.add_subplot(211)
        ax.set_title("MAE")
        m = Basemap(projection='robin',lon_0=-180)
        MAE_LF_p = MAE_LF[i,:,:]
        contour_plot = m.pcolormesh(X, Y, MAE_LF_p, latlon = True, cmap='Blues_r')
        m.drawcoastlines(linewidth=2.0, color='0.25')
        m.colorbar(contour_plot)
        ax = fig.add_subplot(212)
        ax.set_title("Uncertainty "+r'$\sigma_M$')
        m = Basemap(projection='robin',lon_0=-180)
        std_rpn_LF_p = std_rpn_LF[i,:,:]
        contour_plot = m.pcolormesh(X, Y, std_rpn_LF_p, latlon = True, cmap='Blues_r')
        m.drawcoastlines(linewidth=2.0, color='0.25')
        m.colorbar(contour_plot)
        plt.subplots_adjust(hspace=-0.35)
        if i > 25:
            plt.savefig('long_lat_uncert_plots/LF_moist_long_lat_uncert_'+str(i-26+4)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            
        else:
            plt.savefig('long_lat_uncert_plots/LF_moist_long_lat_uncert_'+str(i)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
             
else:
    if is_MAE == 1:
        for i in range(dim_y):
            
            mmin = min( np.min(MAE_det[i,:,:]), np.min(MAE_SF[i,:,:]), np.min(MAE_MF[i,:,:]), np.min(MAE_LF[i,:,:]) )
            mmax = max( np.max(MAE_det[i,:,:]), np.max(MAE_SF[i,:,:]), np.max(MAE_MF[i,:,:]), np.max(MAE_LF[i,:,:]) )
            levels = np.linspace(mmin, mmax, 8)
            fig = plt.figure(figsize=(21, 14))
            ax = fig.add_subplot(221)
            ax.set_title("Deterministic NN")
            m = Basemap(projection='robin',lon_0=-180)
            MAE_det_p = MAE_det[i,:,:]
            contour_plot = m.pcolormesh(X, Y,MAE_det_p, latlon = True, cmap='Blues', vmin=mmin, vmax=mmax)
            m.drawcoastlines(linewidth=2.0, color='0.25')
            m.colorbar(contour_plot)
            
            ax = fig.add_subplot(222)
            ax.set_title("SF-HF-RPN")
            m = Basemap(projection='robin',lon_0=-180)
            MAE_SF_p = MAE_SF[i,:,:]
            contour_plot = m.pcolormesh(X, Y, MAE_SF_p, latlon = True, cmap='Blues', vmin=mmin, vmax=mmax)
            m.drawcoastlines(linewidth=2.0, color='0.25')
            m.colorbar(contour_plot)
            
            ax = fig.add_subplot(224)
            ax.set_title("MF-RPN")
            m = Basemap(projection='robin',lon_0=-180)
            MAE_MF_p = MAE_MF[i,:,:]
            contour_plot = m.pcolormesh(X, Y,MAE_MF_p, latlon = True, cmap='Blues', vmin=mmin, vmax=mmax)
            m.drawcoastlines(linewidth=2.0, color='0.25')
            m.colorbar(contour_plot)
            
            ax = fig.add_subplot(223)
            ax.set_title("LF-RPN")
            m = Basemap(projection='robin',lon_0=-180)
            MAE_LF_p = MAE_LF[i,:,:]
            contour_plot = m.pcolormesh(X, Y, MAE_LF_p, latlon = True, cmap='Blues', vmin=mmin, vmax=mmax)
            m.drawcoastlines(linewidth=2.0, color='0.25')
            m.colorbar(contour_plot)
            
            plt.subplots_adjust(wspace=0.15, hspace=-0.35)
            if i > 25:
                plt.savefig('long_lat_plots/moist_MAE_long_lat_'+str(i-26+4)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            else:
                plt.savefig('long_lat_plots/heat_MAE_long_lat_'+str(i)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            
    if is_R2 == 1:
        for i in range(dim_y):
            
            mmin = max(min( np.min(r2_det[i,:,:]), np.min(r2_SF[i,:,:]), np.min(r2_MF[i,:,:]), np.min(r2_LF[i,:,:])), -10 )
            levels = np.linspace(0, 1, 5)
            fig = plt.figure(figsize=(21, 14))
            ax = fig.add_subplot(221)
            ax.set_title("Deterministic NN")
            m = Basemap(projection='robin',lon_0=-180)
            r2_det_p = r2_det[i,:,:]
            contour_plot = m.pcolormesh(X, Y, r2_det_p, latlon = True, cmap='Blues', vmin=0, vmax=1)
            m.contour(X, Y, r2_det_p, [0.7], latlon = True, colors='pink', linewidths=[2])
            m.drawcoastlines(linewidth=2.0, color='0.25')
            m.colorbar(contour_plot)
            
            ax = fig.add_subplot(222)
            ax.set_title("SF-HF-RPN")
            m = Basemap(projection='robin',lon_0=-180)
            r2_SF_p = r2_SF[i,:,:]
            contour_plot = m.pcolormesh(X, Y, r2_SF_p, latlon = True, cmap='Blues', vmin=0, vmax=1)
            m.contour(X, Y, r2_SF_p, [0.7], latlon = True, colors='pink', linewidths=[2])
            m.drawcoastlines(linewidth=2.0, color='0.25')
            m.colorbar(contour_plot)
            
            ax = fig.add_subplot(224)
            ax.set_title("MF-RPN")
            m = Basemap(projection='robin',lon_0=-180)
            r2_MF_p = r2_MF[i,:,:]
            contour_plot = m.pcolormesh(X, Y, r2_MF_p, latlon = True, cmap='Blues', vmin=0, vmax=1)
            m.contour(X, Y, r2_MF_p, [0.7], latlon = True, colors='pink', linewidths=[2])
            m.drawcoastlines(linewidth=2.0, color='0.25')
            m.colorbar(contour_plot)
            
            ax = fig.add_subplot(223)
            ax.set_title("LF-RPN")
            m = Basemap(projection='robin',lon_0=-180)
            r2_LF_p = r2_LF[i,:,:]
            contour_plot = m.pcolormesh(X, Y, r2_LF_p, latlon = True, cmap='Blues', vmin=0, vmax=1)
            m.contour(X, Y, r2_LF_p, [0.7], latlon = True, colors='pink', linewidths=[2])
            m.drawcoastlines(linewidth=2.0, color='0.25')
            m.colorbar(contour_plot)
            
            plt.subplots_adjust(wspace=0.15, hspace=-0.35)
            if i > 25:
                plt.savefig('long_lat_plots/moist_r2_long_lat_'+str(i-26+4)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            else:
                plt.savefig('long_lat_plots/heat_r2_long_lat_'+str(i)+'.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
            