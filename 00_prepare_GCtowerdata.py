#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

data preparation for 10m and 25m height data from GC flux tower

pair data from different heights (excel table of common variable names and units)
omit extreme outliers from key variables (listed in excel table)
compute inputs for FFP model: obhukov length, zm, boundary layer height (from ERA5 data)

save output csv file as: 'GCFluxTower_30min_2heights.csv'

csv file to be input to FFP code (01_RunFFP_both_heights.py)


"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import atmos_functions as af #contains Obukhov, get_new_blh, blh_process, other functions


data_path = 'DATA_Tower'
#Download the boundary layer height for flux footprint from ERA5 data
ERA5_path = 'ERA5'

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
    
    #do some operation on the 10m to merge units...
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

#WS10m and WS25m are wind speed inputs

#input for flux tower footprints:
# Reduce df_compare resolution to 30-min (averaging)
df_compare_30min = df_compare
df_compare_30min = df_compare_30min.resample('30min',axis=0, on='Date').mean()
df_compare_30min.reset_index(inplace=True)
df_compare_30min.insert(1,'zo10m',zm10m)
df_compare_30min.insert(1,'zo25m',zm25m)


df_compare_30min.to_csv('GCFluxTower_30min_2heights.csv')
  
            
            