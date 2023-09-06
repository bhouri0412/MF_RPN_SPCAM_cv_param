#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 01:22:45 2022

@author: mohamedazizbhouri
"""

import os
current_dirs_parent = os.path.dirname(os.getcwd())
current_dirs_parent = os.path.dirname(current_dirs_parent)
import netCDF4 as nc
import numpy as onp

def read_data(ds0, ds1, ds2):
    
    TBP = onp.transpose(ds0.variables['TBP'],axes=(1,0,2,3))
    QBP = onp.transpose(ds0.variables['QBP'],axes=(1,0,2,3))
    CLDLIQBP = onp.transpose(ds0.variables['CLDLIQBP'],axes=(1,0,2,3))
    CLDICEBP = onp.transpose(ds0.variables['CLDICEBP'],axes=(1,0,2,3))
    
    TBP = onp.reshape(TBP, (lev, time*lat*lon))
    QBP = onp.reshape(QBP, (lev, time*lat*lon))
    CLDLIQBP = onp.reshape(CLDLIQBP, (lev, time*lat*lon))
    CLDICEBP = onp.reshape(CLDICEBP, (lev, time*lat*lon))
    
    PS = onp.reshape(ds0.variables['PS'], (1, time*lat*lon))
    SOLIN = onp.reshape(ds0.variables['SOLIN'], (1, time*lat*lon))
    SHFLX = onp.reshape(ds0.variables['SHFLX'], (1, time*lat*lon))
    LHFLX = onp.reshape(ds0.variables['LHFLX'], (1, time*lat*lon))
    
    is_land = onp.reshape(ds0.variables['LANDFRAC'], (1, time*lat*lon)) > onp.reshape(ds0.variables['OCNFRAC'], (1, time*lat*lon))
    
    inputs = onp.concatenate((TBP,QBP,CLDLIQBP,CLDICEBP,PS,SOLIN,SHFLX,LHFLX,is_land)).T
    
    TBC = onp.transpose(ds1.variables['TBC'],axes=(1,0,2,3))
    TBC = onp.reshape(TBC, (lev, time*lat*lon))
    TBCTEND=(TBC-TBP)/DT
    
    QBC = onp.transpose(ds1.variables['QBC'],axes=(1,0,2,3))
    QBC = onp.reshape(QBC, (lev, time*lat*lon))
    QBCTEND=(QBC-QBP)/DT
    
    CLDLIQBC = onp.transpose(ds1.variables['CLDLIQBC'],axes=(1,0,2,3))
    CLDLIQBC = onp.reshape(CLDLIQBC, (lev, time*lat*lon))
    CLDLIQBCTEND=(CLDLIQBC-CLDLIQBP)/DT
    
    CLDICEBC = onp.transpose(ds1.variables['CLDICEBC'],axes=(1,0,2,3))
    CLDICEBC = onp.reshape(CLDICEBC, (lev, time*lat*lon))
    CLDICEBCTEND=(CLDICEBC-CLDICEBP)/DT
    
    NN2L_FLWDS = onp.reshape(ds2.variables['NN2L_FLWDS'], (1, time*lat*lon))
    
    NN2L_NETSW = onp.reshape(ds2.variables['NN2L_NETSW'], (1, time*lat*lon))
    
    NN2L_PRECC = onp.reshape(ds2.variables['NN2L_PRECC'], (1, time*lat*lon))
    
    NN2L_PRECSC = onp.reshape(ds2.variables['NN2L_PRECSC'], (1, time*lat*lon))
    
    NN2L_SOLL = onp.reshape(ds2.variables['NN2L_SOLL'], (1, time*lat*lon))
    
    NN2L_SOLLD = onp.reshape(ds2.variables['NN2L_SOLLD'], (1, time*lat*lon))
    
    NN2L_SOLS = onp.reshape(ds2.variables['NN2L_SOLS'], (1, time*lat*lon))
    
    NN2L_SOLSD = onp.reshape(ds2.variables['NN2L_SOLSD'], (1, time*lat*lon))
    
    outputs = onp.concatenate((TBCTEND,QBCTEND,CLDLIQBCTEND,CLDICEBCTEND,
                              NN2L_FLWDS,NN2L_NETSW,NN2L_PRECC,NN2L_PRECSC,
                              NN2L_SOLL,NN2L_SOLLD,NN2L_SOLS,NN2L_SOLSD)).T
    return inputs, outputs
    
is_4K = 0
is_8K = 1
if is_4K == 1:
    pp = current_dirs_parent+'/07088/tg863871/CESM2_case/CAM5_1024_NN_L26_Feb27_4K/archive/CAM5_1024_NN_L26_Feb27_4K/atm/hist/'
elif is_8K == 1:
    pp = current_dirs_parent+'/07088/tg863871/CESM2_case/CAM5_1024_NN_L26_Apr26_8K/archive/CAM5_1024_NN_L26_Apr26_8K/atm/hist/'
    
case = 4 
if is_4K == 1:
    if case == 1:
        l_year = ['2003']
        l_month = ['01','03','05','07','08','10','12']
        l_day = [1,7,13,19,25,31]
        l_N = [5,5,5,5,5,0]
    elif case == 2:
        l_year = ['2004']
        l_month = ['01']
        l_day = [1,7,13,19,25,31]
        l_N = [5,5,5,5,5,0]
    elif case == 3:
        l_year = ['2003']
        l_month = ['02','04','06','09','11']
        l_day = [1,7,13,19,25]
        l_N_other = [5,5,5,5,5]
        l_N_02 = [5,5,5,5,3]
    elif case == 4:
        l_year = ['2004']
        l_month = ['02']
        l_day = [1]
        l_N = [2]
elif is_8K == 1:
    if case == 1:
        l_year = ['2004']
        l_month = ['02']
        l_day = [1,7,13,19,25]
        l_N = [5,5,5,5,3]
    elif case == 2:
        l_year = ['2004']
        l_month = ['03']
        l_day = [1,7,13]
        l_N = [5,5,0]
    elif case == 3:
        l_year = ['2003']
        l_month = ['02','04','06','09','11']
        l_day = [1,7,13,19,25]
        l_N_other = [5,5,5,5,5]
        l_N_02 = [5,5,5,5,3]
    elif case == 4: 
        l_year = ['2004']
        l_month = ['01']
        l_day = [1,7,13,19,25,31]
        l_N = [5,5,5,5,5,0]
    elif case == 5:
        l_year = ['2003']
        l_month = ['01','03','05','07','08','10','12']
        l_day = [1,7,13,19,25,31]
        l_N = [5,5,5,5,5,0]

for k in range(len(l_year)):
    i_year = l_year[k]
    
    for i in range(len(l_month)):
        
        i_month = l_month[i]    
        
        #case 1, 2, 4, 5
        # nothing, l_N and l_day are the same for any i index
        if case == 3:
            if i_month == '02':
                l_N = l_N_02
            else:
                l_N = l_N_other
        
        for ii in range(len(l_day)):
            i_day = l_day[ii]
            N_day = l_N[ii]
            
            if i_day<10:
                i_day_str = '0'+str(i_day)
            else:
                i_day_str = str(i_day)
            
            if is_4K == 1:
                ds0 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h0.'+i_year+'-'+i_month+'-'+i_day_str+'-01800.nc')
                ds1 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h1.'+i_year+'-'+i_month+'-'+i_day_str+'-01800.nc')
                ds2 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h2.'+i_year+'-'+i_month+'-'+i_day_str+'-01800.nc')
            elif is_8K == 1:
                ds0 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h0.'+i_year+'-'+i_month+'-'+i_day_str+'-01800.nc')
                ds1 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h1.'+i_year+'-'+i_month+'-'+i_day_str+'-01800.nc')
                ds2 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h2.'+i_year+'-'+i_month+'-'+i_day_str+'-01800.nc')                
            
            DT = 30*60
            lat = ds0.dimensions['lat'].size
            time = ds0.dimensions['time'].size
            lon = ds0.dimensions['lon'].size
            lev = ds0.dimensions['lev'].size
            
            inputs, outputs = read_data(ds0, ds1, ds2)
            
            #due to zero solar insulation remove half of the points
            name_end = ['01800','05400','09000','12600','16200',
                        '19800','23400','27000','30600','34200',
                        '37800','41400','45000','48600','52200',
                        '55800','59400','63000','66600','70200',
                        '73800','77400','81000','84600']
            
            num_name = len(name_end)
            
            for i in range(num_name-1):
                if is_4K == 1:
                    ds0 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h0.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i+1]+'.nc')
                    ds1 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h1.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i+1]+'.nc')
                    ds2 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h2.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i+1]+'.nc')
                elif is_8K == 1:
                    ds0 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h0.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i+1]+'.nc')
                    ds1 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h1.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i+1]+'.nc')
                    ds2 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h2.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i+1]+'.nc')
                
                inputs_l, outputs_l = read_data(ds0, ds1, ds2)
                inputs = onp.concatenate((inputs,inputs_l))
                outputs = onp.concatenate((outputs,outputs_l))
            
            for j in range(N_day):
                print(j)
                i_day += 1
                if i_day<10:
                    i_day_str = '0'+str(i_day)
                else:
                    i_day_str = str(i_day)        
                for i in range(num_name):
                    if is_4K == 1:
                        ds0 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h0.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i]+'.nc')
                        ds1 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h1.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i]+'.nc')
                        ds2 = nc.Dataset(pp+'CAM5_1024_NN_L26_Feb27_4K.cam.h2.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i]+'.nc')
                    elif is_8K == 1:
                        ds0 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h0.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i]+'.nc')
                        ds1 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h1.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i]+'.nc')
                        ds2 = nc.Dataset(pp+'CAM5_1024_NN_L26_Apr26_8K.cam.h2.'+i_year+'-'+i_month+'-'+i_day_str+'-'+name_end[i]+'.nc')
                       
                    inputs_l, outputs_l = read_data(ds0, ds1, ds2)
                    inputs = onp.concatenate((inputs,inputs_l))
                    outputs = onp.concatenate((outputs,outputs_l))
            
            if i_day<10:
                i_day_str = '0'+str(i_day)
            else:
                i_day_str = str(i_day)
                
            if is_4K == 1:
                onp.save('data_CAM5_4K/inputs_'+i_year+'_'+i_month+'_'+i_day_str,inputs)
                onp.save('data_CAM5_4K/outputs_'+i_year+'_'+i_month+'_'+i_day_str,outputs)
            elif is_8K == 1:
                onp.save('data_CAM5_8K/inputs_'+i_year+'_'+i_month+'_'+i_day_str,inputs)
                onp.save('data_CAM5_8K/outputs_'+i_year+'_'+i_month+'_'+i_day_str,outputs)
            
            print('h0.'+i_year+'-'+i_month+'-'+i_day_str, ', train data shape: ',inputs.shape, outputs.shape) # (221184, 108) (221184, 112)
            