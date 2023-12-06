#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:30:22 2023

calculate footprints for 25m and 10m heights at goose creek tower, using Kljun 2015 FFP model (as a function fp_clim)

data files loaded: FluxData_Raw_10m.csv, FluxData_Raw_ALL.csv: data since 2019 for 25m tower, 2021 for 10m tower
tower_compare_variables.csv:table of variable names and units to match between instruments and convert
ERA5_path: folder containing a .nc file of ERA5 boundary layer height during the study period

This code processes the data for input into the FFP model, calculates Obhukov length L
Then runs FFP for each time step, and each tower height, saves in daily pickle files for further analysis

NOTE: unsolved memory issue in this code, I had to run for several months at a time depending on computer

@author: allison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
import gc
from datetime import date

import calc_footprint_FFP_climatology as fp_clim #contains Kljun FFP model
import atmos_functions as af #contains Obukhov, get_new_blh, blh_process, other functions

data_path = 'DATA_Tower'
#Download the boundary layer height for flux footprint from ERA5 data
ERA5_path = 'ERA5'

#alter this span to compute FFPs for only certain time range
months = range(1,13)
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  
years = [2019]


data_10m = pd.read_csv(data_path+ "/FluxData_Raw_10m.csv")
data_25m = pd.read_csv(data_path+"/FluxData_Raw_ALL.csv")


blh_dict = af.blh_from_nc(ERA5_path)  

#need both UTM and geographic coordinates for flux tower
tower_E = 365624.55
tower_N = 4446215.92

tower_lon = -88.5773415
tower_lat = 40.1562377

compare_var_table = pd.read_csv('tower_compare_variables.csv')
#table of variables from 25m and 10m instruments that should be comparable with eachother

df_all = pd.merge(data_25m,data_10m,on=['NewDate'],how='outer')
df_compare=pd.DataFrame()
df_compare['Date']=df_all['NewDate']

m_mass_C = 12.01+16*2
m_mass_H2O = 18.01528

plt.figure(1)
plt.subplots(5,4,figsize=(10,15))

for ct,i in enumerate(compare_var_table['25m']):

    dat_25m_var = pd.to_numeric(df_all[i],errors='coerce')
    dat_10m_var = pd.to_numeric(df_all[compare_var_table['10m'][ct]],errors='coerce')
    
    #do some operation on the 10m to merge units?
    if compare_var_table['operation10'][ct] == 'Fc':
        #Fc conversion
        #conversion to get Fc to be same units for both   
        dat_10m_var = dat_10m_var *m_mass_C/1000    
    elif compare_var_table['operation10'][ct] == 'KtoC':
        #temp conversion
        dat_10m_var = dat_10m_var - 273.15
    elif compare_var_table['operation10'][ct] == 'sqrt':
        #variance to stdev
        dat_10m_var = np.sqrt(dat_10m_var)
    elif compare_var_table['operation10'][ct] =='div1000':
        #Pa to kPa for vapor pressures
        dat_10m_var = dat_10m_var/1000
    elif compare_var_table['operation10'][ct] == 'sqrtFc':
        #take square root, convert to mg CO2 from micromoles
        dat_10m_var = np.sqrt(dat_10m_var)
        dat_10m_var = dat_10m_var *m_mass_C/1000 * 15*60            
    elif compare_var_table['operation10'][ct] == 'sqrtH2O':
        #take square root, convert to g H2O from millimoles
        dat_10m_var = np.sqrt(dat_10m_var) * 15*60 * m_mass_H2O/(10**6) 
    
    #next, do some pre-processing: set values below static min and max to np.nan
    minval = float(compare_var_table['min'][ct])
    maxval = float(compare_var_table['max'][ct])
    
    dat_25m_var = np.where((dat_25m_var < minval) | (dat_25m_var > maxval), np.nan, dat_25m_var)
    dat_10m_var = np.where((dat_10m_var < minval) | (dat_10m_var > maxval), np.nan, dat_10m_var)

    #additional pre-processing based on quality flags    
    newstr = compare_var_table['new'][ct]
    
    if (newstr == 'LE') | (newstr == 'H') | (newstr == 'Fc'):
        #test for sonic and licor samples from 25m height
        dat_25m_var = np.where(df_all['sonic_samples_Tot']<4500,np.nan, dat_25m_var)
        dat_25m_var = np.where(df_all['irga_li_samples_Tot']<4500,np.nan, dat_25m_var)
        
        #test quality flags for each flux for 10m height data
        if newstr=='LE':
            dat_10m_var=dat_10m_var
            dat_10m_var = np.where(df_all['qc_LE']==2,np.nan,dat_10m_var)
        elif newstr == 'H':
            dat_10m_var = np.where(df_all['qc_H']==2,np.nan,dat_10m_var)
        else: #Fc
            dat_10m_var = np.where(df_all['qc_co2_flux']==2,np.nan,dat_10m_var)
     
    
        #also check whether it is raining - set fluxes to np.nan for precipitation
        dat_25m_var = np.where(df_all['Precip_Tot']>1,np.nan, dat_25m_var)
        dat_10m_var = np.where(df_all['Precip_Tot']>1,np.nan, dat_10m_var)
        
    if (newstr == 'e'):      
        div = df_all['e']/df_all['es']        
        dat_10m_var = np.where(div>1,df_all['es'],dat_10m_var)

    newstr_10 = newstr+'10m'
    newstr_25 = newstr+'25m'
    
    df_compare[newstr_10]=dat_10m_var
    df_compare[newstr_25]=dat_25m_var
        
    plt.subplot(5,4,ct+1)
    plt.plot(dat_10m_var,dat_25m_var,'.')
    plt.title(newstr)
    plt.xlim([minval, maxval])
    plt.ylim([minval, maxval])
    
#for purposes of finding Obukhov L, use average H (sensible heat flux value)
#to fill some gaps

df_compare['H_avg'] =  df_compare[['H25m', 'H10m']].mean(axis=1,skipna=True)
    
df_compare['ob10m']= af.Obukhov(df_compare.Ta10m, df_compare.Pair10m, df_compare.H_avg, 
                              df_compare.rho10m, df_compare.ustar10m)    
df_compare['ob25m']= af.Obukhov(df_compare.Ta25m, df_compare.Pair25m, df_compare.H_avg, 
                              df_compare.rho25m, df_compare.ustar25m) 

plt.subplot(5,4,20)
plt.plot(df_compare['ob10m'],df_compare['ob25m'],'.')
plt.title('Ob length')
plt.xlim([-3000,3000])
plt.ylim([-3000,3000])  

df_compare['Date'] = pd.to_datetime(df_compare['Date'])

df_compare['blh'] = df_compare.Date.map(blh_dict)

#convert based on df_compare variable names

d = 2  # Estimate...
zm25m = 25 - d 
zm10m = 10 - d

#WS10m and WS25m are wind speed inputs, no zo

#input for flux tower footprints:
# Reduce df_compare resolution to 30-min (averaging)

df_compare_30min = df_compare
df_compare_30min = df_compare_30min.resample('30min',axis=0, on='Date').mean()
df_compare_30min.reset_index(inplace=True)
df_compare_30min.insert(1,'zo10m',zm10m)
df_compare_30min.insert(1,'zo25m',zm25m)

#%%
#run flux footprint climatology
rmax = 80
                           
                                
for y in years:
    
    for i,m in enumerate(months):
                
        print([y,m])
        ndays = days_in_month[i]
        
        for d in range(1,ndays+1):
        
            #print([y, m, ndays])
            date_begin = np.datetime64(date(y,m,d))
            date_end = date_begin +1                       
            
            df_clip = df_compare_30min.loc[df_compare_30min.Date >= date_begin]
            df_clip = df_clip.loc[df_compare_30min.Date < date_end]
            
            if len(df_clip)==0:
                continue
             
            FFP25m_list=[]
            FFP10m_list=[]
            dates =[]
            LE10=[]
            LE25=[]
            error_vect10 =[]
            error_vect25 =[]  
            dict25={}
            dict10={}
            
            print(df_clip['Date'].iloc[0])
            
            no_fp_count_10=0
            no_fp_count_25=0
                                    
            for ct,item in enumerate(df_clip.index):
                FFP10m=[]
                FFP25m={}
    
                try:
                #mushfika/dawn codes...
                    FFP10m = fp_clim.FFP_climatology(zm=df_clip['zo10m'][item], z0=None, 
                                          domain = [-4500, 4500, -4500, 4500],
                                          umean=df_clip['WS10m'][item], 
                                          h=df_clip['blh'][item], 
                                          ol=df_clip['ob10m'][item], 
                                          sigmav=df_clip['vstd10m'][item], 
                                          ustar=df_clip['ustar10m'][item], 
                                          wind_dir=df_clip['WD10m'][item], 
                                          rs = list(range(10, rmax+10, 10)), 
                                          #nx=600, 
                                          dx=30,
                                          fig=0, crop=1,verbosity=0)
                    
                    FFP10m_list.append(FFP10m)
                    LE10.append(df_clip['LE10m'][item])
                except: 
                    error_vect10.append(df_clip['Date'][item])   
                    FFP10m_list.append('no footprint')
                    no_fp_count_10+=1
    
    
                try:
                    FFP25m = fp_clim.FFP_climatology(zm=df_clip['zo25m'][item], z0=None, 
                                          domain = [-4500, 4500, -4500, 4500],
                                          umean=df_clip['WS25m'][item], 
                                          h=df_clip['blh'][item], 
                                          ol=df_clip['ob25m'][item], 
                                          sigmav=df_clip['vstd25m'][item], 
                                          ustar=df_clip['ustar25m'][item], 
                                          wind_dir=df_clip['WD25m'][item], 
                                          rs = list(range(10, rmax+10, 10)), 
                                          #nx=600,
                                          dx=30,
                                          fig=0, crop=1, verbosity=0)       
    
                    
                    FFP25m_list.append(FFP25m)
                    LE25.append(df_clip['LE25m'][item])
                    
    
                except: 
                    error_vect25.append(df_clip['Date'][item]) 
                    FFP25m_list.append('no footprint')
                    no_fp_count_25 +=1
            
                dates.append(df_clip['Date'][item])
             
            print(no_fp_count_10,no_fp_count_25)
            dict_25 ={"date":dates,"FFP":FFP25m_list,"errordates":error_vect25,"LE":LE25}
            dict_10 ={"date":dates,"FFP":FFP10m_list,"errordates":error_vect10,"LE":LE10}
    
            dict_all ={}
            dict_all['25m']=dict_25
            dict_all['10m']=dict_10
            dict_all['readme']='Daily flux footprints (30 min resolution) for 10m and 25m heights on GC tower, IL.  data = input data to FFP models and fluxes, 25m and 10m are dictionaries with list of dates and FFP files'
            dict_all['data']=df_clip #also save the actual input data in the dictionary
      
            towername = 'GC'
            filename = '%s_ffps_30min_%s_%s_%s.pickle' %(towername, y, m,d)
            filepath = 'GC_FFPs//'
            with gzip.open(filepath + filename, 'wb') as f: 
                pickle.dump(dict_all, f)   

            #trying to solve memory problem
            del df_clip, dict_25, dict_10, FFP10m_list, FFP25m_list
            gc.collect()
            
            
            
            
            