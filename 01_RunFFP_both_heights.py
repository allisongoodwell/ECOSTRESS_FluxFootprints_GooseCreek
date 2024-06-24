#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:30:22 2023

calculate footprints for 25m and 10m heights at goose creek tower, using Kljun 2015 FFP model (as a function fp_clim)
*footprints used in Goodwell, Zahan, URycki 2024 are available in Hydroshare as zipped pickle files

data files loaded: GCFluxTower_30min_2heights.csv, contains 30 min data and atmos properties to obtain flux footprints

note: pickle files saved for each day, containing dictionary with FFPs for both heights (or only 25m, depending on date)

NOTE: unsolved memory issue in this code, I had to run for several months at a time depending on computer

@author: allison
"""

import pandas as pd
import numpy as np
import pickle
import gzip
import gc
from datetime import date

import calc_footprint_FFP_climatology as fp_clim #contains Kljun FFP model

#alter this span to compute FFPs for only certain time range
months = range(1,13)
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  
years = [2019]


df_compare_30min = pd.read_csv('GCFluxTower_30min_2heights.csv')
df_compare_30min['Date']=pd.to_datetime(df_compare_30min['Date'])


#%%
#run flux footprint climatology
rmax = 80 #last contour of FFP (max 99, used 80)
                           
                                
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
             
            print('10m and 25 m no-calcs (48= no data):')
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
            
            
            
            
            