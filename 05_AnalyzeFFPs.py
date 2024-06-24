# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:13:12 2023

load file created in code footprint_fractions_10m_25m_GC.py

download data from Clowder and do some analysis

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

df = df[df['hour']<18]
df = df[df['hour']>6]





df_daily = df.resample('1D',on='Date').mean()
df_daily = df_daily.reset_index()




#all 15 minute data from 25m height and other variables (temp, soil moisture, etc)
df_25m_all = pd.read_csv('DATA_Tower/FluxData_SolarCorrected25m_ToQAQC.csv')
df_25m_all['Date']= pd.to_datetime(df_25m_all['Date'])

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

#load ecostress results for comparison...
df_ECOSTRESS = pd.read_csv('Results_performance/ECOSTRESS_footprintETvalues_5x5.csv')
df_ECOSTRESS['Date']=pd.to_datetime(df_ECOSTRESS['Date'])

df_ECO_all = df_ECOSTRESS
df_ECOSTRESS = df_ECOSTRESS[(df_ECOSTRESS['good']==1)]
df_ECO_allgood = df_ECOSTRESS.dropna()


df_ECOSTRESS['Date']=pd.to_datetime(df_ECOSTRESS['Date'])
df_ECOSTRESS_clip = df_ECOSTRESS[~(df_ECOSTRESS['Date'] < '2021-09-01')]
df_ECOSTRESS_clip[['AlexiCorn','AlexiSoy','JdayCorn','JdaySoy']].describe()

 
fig_path = 'Outputs/'           

        
#%%

n =3#number of days for moving average
n_min = 1

#cut out non-growing season data
df_all = df_all[df_all['month']>3]
df_all = df_all[df_all['month']<11]

df_all['f_corn10m_r']= df_all.f_corn10m.rolling(n,min_periods=n_min).mean()
df_all['f_soy10m_r']= 1 - df_all.f_corn10m.rolling(n,min_periods=n_min).mean()
df_all['f_corn25m_r']= df_all.f_corn25m.rolling(n,min_periods=n_min).mean()
df_all['f_soy25m_r']= 1 - df_all.f_corn25m.rolling(n,min_periods=n_min).mean()

df_all['ETmean']=df_all[['ET_corr10','ET_corr25']].mean(axis=1)
df_all['LEpercent_diff_heights']=np.abs((df_all['ET_corr10']-df_all['ET_corr25'])/df_all['ETmean'])*100


#Solve for ETcorn and ETsoy using determinants
df_all['Det']= df_all['f_corn25m_r']*df_all['f_soy10m_r']-df_all['f_corn10m_r']*df_all['f_soy25m_r']
df_all['Detc']=df_all['ET_corr25']*df_all['f_soy10m_r']-df_all['ET_corr10']*df_all['f_soy25m_r']
df_all['Dets']= df_all['f_corn25m_r']*df_all['ET_corr10']-df_all['f_corn10m_r']*df_all['ET_corr25']
df_all['ETcorn']=df_all['Detc']/df_all['Det']
df_all['ETsoy']=df_all['Dets']/df_all['Det']

#calculate percent difference for crops 
df_all['ETpercent_cropdiff']= np.abs((df_all['ETcorn']-df_all['ETsoy'])/df_all['ETmean'])*100

#omit days with large percent difference from tower data OR between the two tower heights (more than 100% different from mean)
Pmax = 100
df_all['ETcorn']=np.where((df_all['LEpercent_diff_heights']>Pmax) | (df_all['ETpercent_cropdiff']>Pmax), df_all['ETmean'],df_all['ETcorn'])
df_all['ETsoy']=np.where((df_all['LEpercent_diff_heights']>Pmax) | (df_all['ETpercent_cropdiff']>Pmax), df_all['ETmean'],df_all['ETsoy'])

#omit days with less than a threshold mm ET at either site based on observations
ETmin=0
df_all['ETcorn']=np.where((df_all['ET_corr25']<ETmin) | (df_all['ET_corr10']<ETmin), np.nan, df_all['ETcorn'])
df_all['ETsoy']=np.where((df_all['ET_corr25']<ETmin) | (df_all['ET_corr10']<ETmin), np.nan, df_all['ETsoy'])


#omit cases where ET is negative or greater than 10, set to tower value mean
df_all['ETcorn']=np.where((df_all['ETcorn']<ETmin) | (df_all['ETcorn']>12), df_all['ETmean'], df_all['ETcorn'])
df_all['ETsoy']=np.where((df_all['ETsoy']<ETmin) | (df_all['ETsoy']>12), df_all['ETmean'], df_all['ETsoy'])

#%%


#fill nan values
df_fill = df_all.interpolate(method='linear')


df_all['ETcropdiff']=df_fill['ETcorn']-df_fill['ETsoy']
df_all['ETheightdiff']=df_fill['ET_corr10']-df_fill['ET_corr25']
df_all['Fracdiff']=df_fill['f_soy10m']-df_fill['f_soy25m']

print(df_all[['ETcorn','ETsoy','ET_corr25','ET_corr10']].describe())


plt.figure(3,figsize = (2.5,4))

n = 30 #number of days for moving average
n_min = 1

plt.subplot(2,1,1)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ETcorn'].loc[df_fill['year']==2021].rolling(n,min_periods=2).mean(),'y',linewidth=3)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ETsoy'].loc[df_fill['year']==2021].rolling(n,min_periods=2).mean(),'g',linewidth=3)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ET_corr10'].loc[df_fill['year']==2021].rolling(n,min_periods=2).mean(),':b',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ET_corr25'].loc[df_fill['year']==2021].rolling(n,min_periods=2).mean(),':r',linewidth=2)
plt.xticks([])

#cumulative
plt.subplot(2,1,2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ETcorn'].loc[df_fill['year']==2021].cumsum(),'y',linewidth=3)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ETsoy'].loc[df_fill['year']==2021].cumsum(),'g',linewidth=3)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ET_corr10'].loc[df_fill['year']==2021].cumsum(),':b',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ET_corr25'].loc[df_fill['year']==2021].cumsum(),':r',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['Precip_Tot'].loc[df_fill['year']==2021].cumsum(),':k',linewidth=2)
plt.ylim([0,300])
plt.xticks(rotation=45)


plt.xlim([dt.datetime(2021,8,17),dt.datetime(2021,10,30)])
plt.tight_layout()

plt.savefig(fig_path+"CropET_fromTower_2021.svg", format="svg", bbox_inches="tight") 
plt.show()


plt.figure(4,figsize = (6,4))

plt.subplot(2,1,1)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ETcorn'].loc[df_fill['year']==2022].rolling(n,min_periods=2).mean(),'y',linewidth=3)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ETsoy'].loc[df_fill['year']==2022].rolling(n,min_periods=2).mean(),'g',linewidth=3)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ET_corr10'].loc[df_fill['year']==2022].rolling(n,min_periods=2).mean(),':b',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ET_corr25'].loc[df_fill['year']==2022].rolling(n,min_periods=2).mean(),':r',linewidth=2)
plt.xticks([])

plt.subplot(2,1,2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ETcorn'].loc[df_fill['year']==2022].cumsum(),'y',linewidth=3)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ETsoy'].loc[df_fill['year']==2022].cumsum(),'g',linewidth=3)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ET_corr10'].loc[df_fill['year']==2022].cumsum(),':b',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ET_corr25'].loc[df_fill['year']==2022].cumsum(),':r',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['Precip_Tot'].loc[df_fill['year']==2022].cumsum(),':k',linewidth=2)
plt.ylim([0,850])
plt.xticks(rotation=45)
plt.legend(['ET corn','ET soy','ET 10m','ET 25m','PPT'])


plt.xlim([dt.datetime(2022,4,1),dt.datetime(2022,9,30)])
plt.tight_layout()


plt.savefig(fig_path+"CropET_fromTower_2022.svg", format="svg", bbox_inches="tight") 
plt.show()



df_fill.to_csv('DATA_Tower/CropSpecificDailyTowerET.csv')

#%%






