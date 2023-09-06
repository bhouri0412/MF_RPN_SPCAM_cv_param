#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:59:59 2023

@author: mohamedazizbhouri
"""
from matplotlib import pyplot as plt
import numpy as onp

#values of average pressure for different vertical levels
press = ('3.5', '7.4', '14', '24', '37', '53', '70', '85', '100', '117',
        '137', '160', '188', '221', '259', '305', '358', '420', '494',
        '581', '673', '761', '837', '897', '937', '958')

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

R2_w_neg = 1
R2_wo_neg = 0
MAE = 0
CRPS = 0

if R2_w_neg == 1: # global R^2 plots without negative values
    nn = 26
    MF = onp.load('glob_errors/r2_rpn_MF.npy')[:26]
    SF = onp.load('glob_errors/r2_rpn_SF.npy')[:26]
    LF = onp.load('glob_errors/r2_rpn_LF.npy')[:26]
    det = onp.load('glob_errors/r2_det.npy')[:26]
    MF[MF<0] = 0
    SF[SF<0] = 0
    LF[LF<0] = 0
    det[det<0] = 0
    xx = onp.arange(nn)
    fig , ax = plt.subplots(figsize=(12,8))
    ax.plot(xx, det, color='blue')
    ax.plot(xx, SF, '--', color='black')
    ax.plot(xx, LF, '--', color='orange')
    ax.plot(xx, MF, color='red')
    ax.set_xlabel('Pressure (hPa)')
    ax.set_xticks(onp.arange(nn))
    ax.set_xticklabels(press, rotation =60, fontsize = 20)
    ax.tick_params(axis='x',colors='grey')
    ax.set_ylim([0,1])
    ax.set_ylabel('$R^2$')
    plt.grid()
    ax.legend(['Det. NN', 'SF-HF-RPN', 'LF-RPN', 'MF-RPN'], bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
    plt.show()
    plt.savefig('glob_errors/R2_w_neg_heat.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
    nn = 22
    MF = onp.load('glob_errors/r2_rpn_MF.npy')[26:]
    SF = onp.load('glob_errors/r2_rpn_SF.npy')[26:]
    LF = onp.load('glob_errors/r2_rpn_LF.npy')[26:]
    det = onp.load('glob_errors/r2_det.npy')[26:]
    MF[MF<0] = 0
    SF[SF<0] = 0
    LF[LF<0] = 0
    det[det<0] = 0
    press = press[4:]
    xx = onp.arange(nn)
    fig , ax = plt.subplots(figsize=(12,8))
    ax.plot(xx, det, color='blue')
    ax.plot(xx, SF, '--', color='black')
    ax.plot(xx, LF, '--', color='orange')
    ax.plot(xx, MF, color='red')
    ax.set_xlabel('Pressure (hPa)')
    ax.set_xticks(onp.arange(nn))
    ax.set_xticklabels(press, rotation =60, fontsize = 20)
    ax.tick_params(axis='x',colors='grey')
    ax.set_ylim([0,1])
    ax.set_ylabel('$R^2$')
    plt.grid()
    ax.legend(['Det. NN', 'SF-HF-RPN', 'LF-RPN', 'MF-RPN'], bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
    plt.show()
    plt.savefig('glob_errors/R2_w_neg_moist.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
if MAE == 1: # gloab MAE plots
    nn = 26
    MF = onp.load('glob_errors/MAE_rpn_MF.npy')[:26]
    SF = onp.load('glob_errors/MAE_rpn_SF.npy')[:26]
    LF = onp.load('glob_errors/MAE_rpn_LF.npy')[:26]
    det = onp.load('glob_errors/MAE_det.npy')[:26]
    xx = onp.arange(nn)
    fig , ax = plt.subplots(figsize=(12,8))
    plt.yscale("log")
    ax.plot(xx, det, color='blue')
    ax.plot(xx, SF, '--', color='black')
    ax.plot(xx, LF, '--', color='orange')
    ax.plot(xx, MF, color='red')
    ax.set_xlabel('Pressure (hPa)')
    ax.set_xticks(onp.arange(nn))
    ax.set_xticklabels(press, rotation =60, fontsize = 20)
    ax.tick_params(axis='x',colors='grey')
    plt.ylabel('MAE')
    plt.grid()
    ax.legend(['Det. NN', 'SF-HF-RPN', 'LF-RPN', 'MF-RPN'], bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
    plt.show()
    plt.savefig('glob_errors/MAE_heat.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
    nn = 22
    MF = onp.load('glob_errors/MAE_rpn_MF.npy')[26:]
    SF = onp.load('glob_errors/MAE_rpn_SF.npy')[26:]
    LF = onp.load('glob_errors/MAE_rpn_LF.npy')[26:]
    det = onp.load('glob_errors/MAE_det.npy')[26:]
    press = press[4:]
    xx = onp.arange(nn)
    fig , ax = plt.subplots(figsize=(12,8))
    plt.yscale("log")
    ax.plot(xx, det, color='blue')
    ax.plot(xx, SF, '--', color='black')
    ax.plot(xx, LF, '--', color='orange')
    ax.plot(xx, MF, color='red')
    ax.set_xlabel('Pressure (hPa)')
    ax.set_xticks(onp.arange(nn))
    ax.set_xticklabels(press, rotation =60, fontsize = 20)
    ax.tick_params(axis='x',colors='grey')
    plt.ylabel('MAE')
    plt.grid()
    ax.legend(['Det. NN', 'SF-HF-RPN', 'LF-RPN', 'MF-RPN'], bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
    plt.show()
    plt.savefig('glob_errors/MAE_moist.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
if R2_w_neg == 1: # global R^2 plots with negative values
    nn = 26
    MF = onp.load('glob_errors/r2_rpn_MF.npy')[:26]
    SF = onp.load('glob_errors/r2_rpn_SF.npy')[:26]
    LF = onp.load('glob_errors/r2_rpn_LF.npy')[:26]
    det = onp.load('glob_errors/r2_det.npy')[:26]
    MF[MF<0] = 0
    SF[SF<0] = 0
    LF[LF<0] = 0
    det[det<0] = 0
    xx = onp.arange(nn)
    # values below are optained from r2_rpn_MF.npy, r2_rpn_SF.npy, r2_rpn_LF.npy and r2_det.npy
    det_r = ['-2.3', '-2.4', '-2.1', '-1.6', '-0.6', '', '-26', '-43', '-15', 
             '-13', '-11', '-3.5', '-3.2', '-2.4', '-1.2', '-0.4', '', '', 
             '', '', '-0.01', '-0.2', '-0.4', '-0.01', '-2.6', '', '-5']
    SF_r = ['-3.5', '-2.6', '-2.4', '-1.9', '-0.9', '', '-64', '-70', '-5.2',
            '-13', '-4.9', '-2.6', '-2.1', '-1.1', '-0.9', '-0.8', '', '', '',
            '', '', '', '', '', '-0.6', '']
    LF_r = ['-5.6', '', '', '', '', '', '', '-1.4', '-0.3', '-0.6', '', '',
            '', '', '', '', '', '', '', '', '', '', '', '', '-0.3', '-0.1']
    fig , ax = plt.subplots(figsize=(12,8))
    ax.plot(xx, det, color='blue')
    ii= 0
    for x,y in zip(xx,det):
        ax.annotate(det_r[ii],
                    (x,y),
                    textcoords="offset points",
                    xytext = (0,-60),
                    ha='center',
                    color = 'blue',
                    fontsize = 20,
                    rotation = -60)
        ii+=1
    ax.plot(xx, SF, '--', color='black')
    ii= 0
    for x,y in zip(xx,SF):
        ax.annotate(SF_r[ii],
                    (x,y),
                    textcoords="offset points",
                    xytext = (0,-120),
                    ha='center',
                    color = 'black',
                    fontsize = 20,
                    rotation = -60)
        ii+=1
    ax.plot(xx, LF, '--', color='orange')
    ii= 0
    for x,y in zip(xx,LF):
        ax.annotate(LF_r[ii],
                    (x,y),
                    textcoords="offset points",
                    xytext = (0,-180),
                    ha='center',
                    color = 'orange',
                    fontsize = 20,
                    rotation = -60)
        ii+=1
    ax.plot(xx, MF, color='red')
    ax.set_ylim([0,1])
    ax.set_xlabel('Pressure (hPa)')
    ax.xaxis.set_label_coords(0.5, -0.55)
    ax.set_xticks(onp.arange(nn))
    ax.set_xticklabels(press, rotation =60, fontsize = 20, y=-0.4)
    ax.tick_params(axis='x',colors='grey')
    ax.set_ylabel('$R^2$')
    plt.grid()
    ax.legend(['Det. NN', 'SF-HF-RPN', 'LF-RPN', 'MF-RPN'], bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
    plt.show()
    plt.savefig('glob_errors/R2_w_neg_heat.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
    nn = 22
    MF = onp.load('glob_errors/r2_rpn_MF.npy')[26:]
    SF = onp.load('glob_errors/r2_rpn_SF.npy')[26:]
    LF = onp.load('glob_errors/r2_rpn_LF.npy')[26:]
    det = onp.load('glob_errors/r2_det.npy')[26:]
    press = press[4:]
    MF[MF<0] = 0
    SF[SF<0] = 0
    LF[LF<0] = 0
    det[det<0] = 0
    xx = onp.arange(nn)
    det_r = ['-94', '-97', '-11', '-14', '-4.4', '-4.9', '-3.3', '-2.7', '-1.6',
             '-1.2', '-0.4', '-0.3', '', '', '', '', '-2.7', '-1.0', '-0.7',
             '-0.6', '-1.2', '']
    SF_r = ['-145', '-71', '-16', '-14', '9.1', '6.3', '-3.9', '-2.8', '-1.8',
            '-1.3', '-0.9', '-0.8', '-0.2', '', '', '', '-0.5', '-0.01', '-0.2',
            '-0.2', '-0.03', '']
    LF_r = ['', '-7.0', '-108', '-225', '-29', '-4.2', '', '', '',
            '', '', '', '', '', '', '', '', '', '', '-2.1', '-0.6', '-0.6']
    MF_r = ['', '', '-0.9', '-2.2', '-0.3', '', '', '', '', '', '', '', '', '',
            '', '', '', '', '', '', '', '']
    fig , ax = plt.subplots(figsize=(12,8))
    ax.plot(xx, det, color='blue')
    ii= 0
    for x,y in zip(xx,det):
        ax.annotate(det_r[ii],
                    (x,y),
                    textcoords="offset points",
                    xytext = (0,-50),
                    ha='center',
                    color = 'blue',
                    fontsize = 20,
                    rotation = -60)
        ii+=1
    ax.plot(xx, SF, '--', color='black')
    ii= 0
    for x,y in zip(xx,SF):
        ax.annotate(SF_r[ii],
                    (x,y),
                    textcoords="offset points",
                    xytext = (0,-100),
                    ha='center',
                    color = 'black',
                    fontsize = 20,
                    rotation = -60)
        ii+=1
    ax.plot(xx, LF, '--', color='orange')
    ii= 0
    for x,y in zip(xx,LF):
        ax.annotate(LF_r[ii],
                    (x,y),
                    textcoords="offset points",
                    xytext = (0,-150),
                    ha='center',
                    color = 'orange',
                    fontsize = 20,
                    rotation = -60)
        ii+=1
    ax.plot(xx, MF, color='red')
    ii= 0
    for x,y in zip(xx,MF):
        ax.annotate(MF_r[ii],
                    (x,y),
                    textcoords="offset points",
                    xytext = (0,-200),
                    ha='center',
                    color = 'red',
                    fontsize = 20,
                    rotation = -60)
        ii+=1
    ax.set_ylim([0,1])
    ax.set_xlabel('Pressure (hPa)')
    ax.xaxis.set_label_coords(0.5, -0.6)
    ax.set_xticks(onp.arange(nn))
    ax.set_xticklabels(press, rotation =60, fontsize = 20, y=-0.45)
    ax.tick_params(axis='x',colors='grey')
    ax.set_ylabel('$R^2$')
    plt.grid()
    ax.legend(['Det. NN', 'SF-HF-RPN', 'LF-RPN', 'MF-RPN'], bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
    plt.show()
    plt.savefig('glob_errors/R2_w_neg_moist.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
if CRPS == 1: # global CRPS plots
    nn = 26
    MF = onp.load('glob_errors/crps_rpn_MF.npy')[:26]
    SF = onp.load('glob_errors/crps_rpn_SF.npy')[:26]
    LF = onp.load('glob_errors/crps_rpn_LF.npy')[:26]
    xx = onp.arange(nn)
    fig , ax = plt.subplots(figsize=(12,8))
    plt.yscale("log")
    ax.plot(xx, SF, '--', color='black')
    ax.plot(xx, LF, '--', color='orange')
    ax.plot(xx, MF, color='red')
    ax.set_xlabel('Pressure (hPa)')
    ax.set_xticks(onp.arange(nn))
    ax.set_xticklabels(press, rotation =60, fontsize = 20)
    ax.tick_params(axis='x',colors='grey')
    plt.ylabel('CRPS')
    plt.grid()
    ax.legend(['Det. NN', 'SF-HF-RPN', 'LF-RPN', 'MF-RPN'], bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
    plt.show()
    plt.savefig('glob_errors/CRPS_heat.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
    nn = 22
    MF = onp.load('glob_errors/crps_rpn_MF.npy')[26:]
    SF = onp.load('glob_errors/crps_rpn_SF.npy')[26:]
    LF = onp.load('glob_errors/crps_rpn_LF.npy')[26:]
    press = press[4:]
    xx = onp.arange(nn)
    fig , ax = plt.subplots(figsize=(12,8))
    plt.yscale("log")
    ax.plot(xx, SF, '--', color='black')
    ax.plot(xx, LF, '--', color='orange')
    ax.plot(xx, MF, color='red')
    ax.set_xlabel('Pressure (hPa)')
    ax.set_xticks(onp.arange(nn))
    ax.set_xticklabels(press, rotation =60, fontsize = 20)
    ax.tick_params(axis='x',colors='grey')
    plt.ylabel('CRPS')
    plt.grid()
    ax.legend(['Det. NN', 'SF-HF-RPN', 'LF-RPN', 'MF-RPN'], bbox_to_anchor=[0.5, 1.2], loc='center', ncol=4)
    plt.show()
    plt.savefig('glob_errors/CRPS_moist.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
