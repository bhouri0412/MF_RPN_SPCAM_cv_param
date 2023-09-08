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
plt.rcParams.update({'font.size': 46,
                     'lines.linewidth': 3,
                     'axes.labelsize': 46,
                     'axes.titlesize': 46,
                     'xtick.labelsize': 46,
                     'ytick.labelsize': 46,
                     'legend.fontsize': 46,
                     'axes.linewidth': 3,
                     "pgf.texsystem": "pdflatex"
                     })

lat = 96
x = np.linspace(-90, 90, lat)
y = np.array([3.5, 7.4, 14, 24, 37, 53, 70, 85, 100, 117,
               137, 160, 188, 221, 259, 305, 358, 420, 494,
               581, 673, 761, 837, 897, 937, 958])
X, Y = np.meshgrid(x, y)
Xm, Ym = np.meshgrid(x, y[4:])
  
r2_det = np.load('SF_results/r2_det_pres_lat.npy')
r2_SF = np.load('SF_results/r2_SF_pres_lat.npy')
r2_MF = np.load('MF_results/r2_MF_pres_lat.npy.npy')
r2_LF = np.load('MF_results/r2_LF_pres_lat.npy.npy')
    
dim_heat = 26

print('Create err plots')
    
mmin = 0
mmax = 1
levels = np.linspace(mmin, mmax, 8)
fig = plt.figure(figsize=(40, 15))

ax = fig.add_subplot(141)
ax.set_title("Deterministic NN")
contour_plot = ax.pcolor(X, Y, r2_det[:dim_heat,:],cmap='Blues', vmin = mmin, vmax = mmax)
ax.contour(X, Y, r2_det[:dim_heat,:], [0.7], colors='pink', linewidths=[4])
ax.contour(X, Y, r2_det[:dim_heat,:], [0.9], colors='orange', linewidths=[4])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_ylabel("Pressure (hPa)")

ax = fig.add_subplot(142)
ax.set_title("SF-HF-RPN")
contour_plot = ax.pcolor(X, Y, r2_SF[:dim_heat,:],cmap='Blues', vmin = mmin, vmax = mmax)
ax.contour(X, Y, r2_SF[:dim_heat,:], [0.7], colors='pink', linewidths=[4])
ax.contour(X, Y, r2_SF[:dim_heat,:], [0.9], colors='orange', linewidths=[4])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_yticks([]) 
ax.set_xlabel("Degrees Latitude")
ax.xaxis.set_label_coords(1.12, -0.1)

ax = fig.add_subplot(143)
ax.set_title("LF-RPN")
contour_plot = ax.pcolor(X, Y, r2_LF[:dim_heat,:],cmap='Blues', vmin = mmin, vmax = mmax)
ax.contour(X, Y, r2_LF[:dim_heat,:], [0.7], colors='pink', linewidths=[4])
ax.contour(X, Y, r2_LF[:dim_heat,:], [0.9], colors='orange', linewidths=[4])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_yticks([]) 

ax = fig.add_subplot(144)
ax.set_title("MF-RPN")
contour_plot = ax.pcolor(X, Y, r2_MF[:dim_heat,:],cmap='Blues', vmin = mmin, vmax = mmax)
ax.contour(X, Y, r2_MF[:dim_heat,:], [0.7], colors='pink', linewidths=[4])
ax.contour(X, Y, r2_MF[:dim_heat,:], [0.9], colors='orange', linewidths=[4])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_yticks([])        
p3 = ax.get_position().get_points().flatten()

cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
fig.colorbar(contour_plot, label=r'$\mathrm{R^2}$', cax=cbar_ax)
plt.suptitle("Heat tendency")
plt.savefig('r2_press_lat_heat.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)

mmin = 0
mmax = 1
levels = np.linspace(mmin, mmax, 8)
fig = plt.figure(figsize=(40, 15))

ax = fig.add_subplot(141)
ax.set_title("Deterministic NN")
contour_plot = ax.pcolor(Xm, Ym, r2_det[dim_heat:,:],cmap='Blues', vmin = mmin, vmax = mmax)
ax.contour(Xm, Ym, r2_det[dim_heat:,:], [0.7], colors='pink', linewidths=[4])
ax.contour(Xm, Ym, r2_det[dim_heat:,:], [0.9], colors='orange', linewidths=[4])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_ylabel("Pressure (hPa)")

ax = fig.add_subplot(142)
ax.set_title("SF-HF-RPN")
contour_plot = ax.pcolor(Xm, Ym, r2_SF[dim_heat:,:],cmap='Blues', vmin = mmin, vmax = mmax)
ax.contour(Xm, Ym, r2_SF[dim_heat:,:], [0.7], colors='pink', linewidths=[4])
ax.contour(Xm, Ym, r2_SF[dim_heat:,:], [0.9], colors='orange', linewidths=[4])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_yticks([]) 
ax.set_xlabel("Degrees Latitude")
ax.xaxis.set_label_coords(1.12, -0.1)

ax = fig.add_subplot(143)
ax.set_title("LF-RPN")
contour_plot = ax.pcolor(Xm, Ym, r2_LF[dim_heat:,:],cmap='Blues', vmin = mmin, vmax = mmax)
ax.contour(Xm, Ym, r2_LF[dim_heat:,:], [0.7], colors='pink', linewidths=[4])
ax.contour(Xm, Ym, r2_LF[dim_heat:,:], [0.9], colors='orange', linewidths=[4])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_yticks([])        

ax = fig.add_subplot(144)
ax.set_title("MF-RPN")
contour_plot = ax.pcolor(Xm, Ym, r2_MF[dim_heat:,:],cmap='Blues', vmin = mmin, vmax = mmax)
ax.contour(Xm, Ym, r2_MF[dim_heat:,:], [0.7], colors='pink', linewidths=[4])
ax.contour(Xm, Ym, r2_MF[dim_heat:,:], [0.9], colors='orange', linewidths=[4])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_yticks([])        
p3 = ax.get_position().get_points().flatten()

cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
fig.colorbar(contour_plot, label=r'$\mathrm{R^2}$', cax=cbar_ax)
plt.suptitle("Moisture tendency")
plt.savefig('r2_press_lat_moist.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
       