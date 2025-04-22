# -*- coding: utf-8 -*-
"""

use dailiy tower observations and crop footprint fractions to get average ET from corn and soybean

@author: goodwela
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

df = pd.read_csv('Results_performance/GC_FootprintFractions.csv') 
df['Date']=pd.to_datetime(df['Date']) 
df['month'] = pd.DatetimeIndex(df['Date']).month
df['year'] = pd.DatetimeIndex(df['Date']).year
df['hour'] = pd.DatetimeIndex(df['Date']).hour

df = df[df['hour']<16]
df = df[df['hour']>9]


df_daily = df.resample('1D',on='Date').mean()
df_daily = df_daily.reset_index()


#daily EB corrected data for both heights
df_EBR_10m = pd.read_csv('DATA_Tower/CorrectedDaily_EBR_GC10mtower.csv')
df_EBR_25m = pd.read_csv('DATA_Tower/CorrectedDaily_EBR_GC25mtower.csv')
df_EBR_10m['date']=pd.to_datetime(df_EBR_10m['date'])
df_EBR_25m['date']=pd.to_datetime(df_EBR_25m['date'])


# want a merged daily dataframe with following variables
# ETcorr for both heights, fractions for both heights, PPT, Ta, RH, WS

df_EBR_10m['ET_corr10']=df_EBR_10m['ET_corr']
df_EBR_25m['ET_corr25']=df_EBR_25m['ET_corr']

df_1 = df_EBR_10m[['date','ET_corr10']]
df_2 = df_EBR_25m[['date','ET_corr25','D5TE_VWC_15cm_Avg','VPD','D5TE_VWC_5cm_Avg','RH_tmpr_rh_mean','Rn_new','wnd_dir_compass','Precip_Tot']]
df_3 = df_daily[['Date','Area10m','Area25m','f_soy10m','f_corn10m','f_soy25m','f_corn25m','month','year']]

df_all = df_1.merge(df_2,on='date',how='inner')
df_all = df_all.merge(df_3,how='inner',left_on='date',right_on='Date')


#%%


# Function to bootstrap means within each month for multiple columns
def bootstrap_monthly_means(df, n_bootstrap=1000):
    
    ETcorn = np.zeros((2,12,n_bootstrap))
    ETsoy = np.zeros((2,12,n_bootstrap))
    ET25_avg = np.zeros((2,12,n_bootstrap))
    ET10_avg = np.zeros((2,12,n_bootstrap))
    
    ETcorn  = {m: {} for m, month in enumerate(range(1,13))}
    ETsoy= {m: {} for m, month in enumerate(range(1,13))}
    ET10_avg= {m: {} for m, month in enumerate(range(1,13))}
    ET25_avg= {m: {} for m, month in enumerate(range(1,13))}
    
    date_vect=[]
    ETcorn_vect=[]
    ETsoy_vect=[]
    ET25_vect=[]
    ET10_vect=[]
    ETcorn_std=[]
    ETsoy_std=[]
    ET25_std=[]
    ET10_std=[]
    
    for y, year in enumerate(range(2021,2023)):
        df_y = df[df['date'].dt.year == year]
        for m, month in enumerate(range(1,13)):
            monthly_data = df_y[df_y['date'].dt.month == month].dropna()
            monthdate = dt.datetime(year,month,1)
            
            ETcorn=[]
            ETsoy=[]
            ET25_avg=[]
            ET10_avg=[]
            for i in range(n_bootstrap):
                if not monthly_data.empty:
                    #sample = monthly_data.sample(n=len(monthly_data), replace=True)
                    sample = monthly_data.sample(len(monthly_data), replace=True)
     
                    #det = np.mean(sample['f_corn25m'])*np.mean(sample['f_soy10m'])-np.mean(sample['f_corn10m'])*np.mean(sample['f_soy25m'])
                    #detc = np.mean(sample['ET_corr25'])*np.mean(sample['f_soy10m'])-np.mean(sample['ET_corr10'])*np.mean(sample['f_soy25m'])
                    #dets = np.mean(sample['f_corn25m'])*np.mean(sample['ET_corr10'])-np.mean(sample['f_corn10m'])*np.mean(sample['ET_corr25'])
                    
                    #daily determinants --> daily ETcorn and ETsoy estimates, omit days with >15mm or <0mm per a crop type
                    det = sample['f_corn25m']*sample['f_soy10m']-sample['f_corn10m']*sample['f_soy25m']
                    detc = sample['ET_corr25']*sample['f_soy10m']-sample['ET_corr10']*sample['f_soy25m']
                    dets = sample['f_corn25m']*sample['ET_corr10']-sample['f_corn10m']*sample['ET_corr25']
                                        
                    ETcornvals = np.where((detc/det<15) & (detc/det>0), detc/det, np.nan)
                    ETsoyvals = np.where((dets/det<15) & (dets/det>0), dets/det, np.nan)
                                                          
                    ETcornval = np.nanmean(ETcornvals)
                    ETsoyval = np.nanmean(ETsoyvals)                        
                    
                    ETcorn.append(ETcornval)
                    ETsoy.append(ETsoyval)
                    ET25_avg.append(np.mean(sample['ET_corr25']))
                    ET10_avg.append(np.mean(sample['ET_corr10']))
            
            date_vect.append(monthdate)
            ETsoy_vect.append(np.nanmean(ETsoy))
            ETcorn_vect.append(np.nanmean(ETcorn))
            ET25_vect.append(np.nanmean(ET25_avg))
            ET10_vect.append(np.nanmean(ET10_avg))
         
            ETsoy_std.append(np.nanstd(ETsoy))
            ETcorn_std.append(np.nanstd(ETcorn))
            ET25_std.append(np.nanstd(ET25_avg))
            ET10_std.append(np.nanstd(ET10_avg))
            
    data = {'date':date_vect, 'ETsoy':ETsoy_vect, 'ETcorn':ETcorn_vect, 'ET25':ET25_vect,
            'ET10':ET10_vect}
    data_err = {'date':date_vect, 'ETsoy':ETsoy_std, 'ETcorn':ETcorn_std, 'ET25':ET25_std,
            'ET10':ET10_std}
    
    df_out = pd.DataFrame(data)
    df_err = pd.DataFrame(data_err)

    return df_out, df_err


# Perform bootstrapping
n_bootstrap = 1000
columns = ['date','ET_corr10', 'ET_corr25','f_soy10m','f_corn10m','f_soy25m','f_corn25m']

df_monthly, df_err = bootstrap_monthly_means(df_all[columns], n_bootstrap)

df_monthly = df_monthly[df_monthly['date']>dt.datetime(2021,8,31)]
df_err = df_err[df_err['date']>dt.datetime(2021,7,31)]

df_monthly['month'] = df_monthly['date'].dt.month
df_err['month'] = df_err['date'].dt.month

#don't compute crop values for months with no vegetation:
df_monthly['ETsoy']=np.where(df_monthly.month<5,np.nan,df_monthly['ETsoy'])
df_monthly['ETsoy']=np.where(df_monthly.month>9,np.nan,df_monthly['ETsoy'])
df_monthly['ETcorn']=np.where(df_monthly.month<5,np.nan,df_monthly['ETcorn'])
df_monthly['ETcorn']=np.where(df_monthly.month>9,np.nan,df_monthly['ETcorn'])

df_err['ETsoy']=np.where((df_err.month<5) | (df_err.month>9),0,df_err['ETsoy'])
df_err['ETcorn']=np.where((df_err.month<5) | (df_err.month>9),0,df_err['ETcorn'])

plt.figure(figsize=(5,2))
ax = df_monthly[['ETcorn','ETsoy','ET25','ET10']].plot(kind='bar',width=.8,ax=plt.gca(),color=['y','g','r','b'],yerr=df_err)

ax.set_xticklabels(['9/21','10/21','11/21','12/21','1/22','2/22','3/22','4/22','5/22','6/22','7/22','8/22','9/22','10/22','11/22','12/22',],rotation=45,fontsize=8)

plt.savefig("CropSpecificMonthly.svg", format="svg", bbox_inches="tight")


#%%

df_monthly.to_csv('DATA_Tower/CropSpecificMonthlyTowerET.csv')

#%%






