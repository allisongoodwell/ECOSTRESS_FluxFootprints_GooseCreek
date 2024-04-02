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

df_ECOSTRESS = pd.read_csv('Results_performance/ECOSTRESS_footprintETvalues_5x5.csv')
df_ECOSTRESS['Date']=pd.to_datetime(df_ECOSTRESS['Date'])


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



df_ECO_all = df_ECOSTRESS
df_ECOSTRESS = df_ECOSTRESS[(df_ECOSTRESS['good']==1)]
df_ECO_allgood = df_ECOSTRESS.dropna()




#df = df[df['Date']> pd.datetime(2019,4,1)]
#df = df[df['Date']< pd.datetime(2022,11,1)]


#time range for both 10m and 25m overlap
df = df[~(df['Date'] < '2021-09-01')]
df = df[~(df['Date'] > '2023-11-01')]

for i,colname in enumerate(df):
    if colname != 'Date':
        df[colname] = pd.to_numeric(df[colname],errors='coerce')
   
 
fig_path = 'Outputs/'           

        
#%%


#assume that 10m tower is representative of single crop - estimate crop-specific ET of that crop
#based on 10m data

#then for other crop, solve based on 25m height fraction

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
df_all['LEpercent_diff']=np.abs((df_all['ET_corr10']-df_all['ET_corr25'])/df_all['ET_corr10'])*100

df_all['Det']= df_all['f_corn25m_r']*df_all['f_soy10m_r']-df_all['f_corn10m_r']*df_all['f_soy25m_r']
df_all['Detc']=df_all['ET_corr25']*df_all['f_soy10m_r']-df_all['ET_corr10']*df_all['f_soy25m_r']
df_all['Dets']= df_all['f_corn25m_r']*df_all['ET_corr10']-df_all['f_corn10m_r']*df_all['ET_corr25']
df_all['ETcornTest']=df_all['Detc']/df_all['Det']
df_all['ETsoyTest']=df_all['Dets']/df_all['Det']

df_all['ETpercent_cropdiff']= np.abs((df_all['ETcornTest']-df_all['ETsoyTest'])/df_all['ETsoyTest'])*100

#omit days with large percent difference from tower data
df_all['ETcornTest'] = [df_all['ETcornTest'].iloc[i] if x<100 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['LEpercent_diff'])]
df_all['ETsoyTest'] = [df_all['ETsoyTest'].iloc[i] if x<100 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['LEpercent_diff'])]

#omit days with large percent difference between crop types
df_all['ETcornTest'] = [df_all['ETcornTest'].iloc[i] if x<100 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETpercent_cropdiff'])]
df_all['ETsoyTest'] = [df_all['ETsoyTest'].iloc[i] if x<100 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETpercent_cropdiff'])]

#omit days with less than a threshold mm ET at either site based on observations
ETmin=0
df_all['ETcornTest'] = [df_all['ETcornTest'].iloc[i] if x>ETmin else np.nan for i,x in enumerate(df_all['ET_corr25'])]
df_all['ETsoyTest'] = [df_all['ETsoyTest'].iloc[i] if x>ETmin else np.nan for i,x in enumerate(df_all['ET_corr25'])]
df_all['ETcornTest'] = [df_all['ETcornTest'].iloc[i] if x>ETmin else np.nan for i,x in enumerate(df_all['ET_corr10'])]
df_all['ETsoyTest'] = [df_all['ETsoyTest'].iloc[i] if x>ETmin else np.nan for i,x in enumerate(df_all['ET_corr10'])]

#omit cases where ET is negative or greater than 10, set to tower value mean
df_all['ETcornTest']= [df_all['ETcornTest'].iloc[i] if x>0 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETcornTest'])]
df_all['ETsoyTest']= [df_all['ETsoyTest'].iloc[i] if x>0 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETsoyTest'])]
df_all['ETcornTest']= [df_all['ETcornTest'].iloc[i] if x<10 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETcornTest'])]
df_all['ETsoyTest']= [df_all['ETsoyTest'].iloc[i] if x<10 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETsoyTest'])]





plt.figure(11)
plt.plot(df_all['ETcornTest'],df_all['ETsoyTest'],'.b')
plt.xlim([0,10])
plt.ylim([0,10])




#use rolling average (smoothed) fraction to estimate when 10m represents corn or soy
df_all['ETcorn'] = [df_all['ET_corr10'].iloc[i] if x > 0.5 else np.nan for i,x in enumerate(df_all['f_corn10m_r'])]
df_all['ETsoy'] = [df_all['ET_corr10'].iloc[i] if x > 0.5 else np.nan for i,x in enumerate(df_all['f_soy10m_r'])]

#other crop ET is solved using 25m data and fractions (also rolling)
df_all['ETsoy_solve'] = (df_all['ET_corr25']-df_all['f_corn25m_r']*df_all['ETcorn'])/df_all['f_soy25m_r']
df_all['ETcorn_solve'] = (df_all['ET_corr25']-df_all['f_soy25m_r']*df_all['ETsoy'])/df_all['f_corn25m_r']

#join together: ETcrop is either = 10m data, or solved for using 25m data
df_all['ETcorn'] = [df_all['ETcorn'].iloc[i] if ~np.isnan(x) else df_all['ETcorn_solve'].iloc[i] for i,x in enumerate(df_all['ETcorn'])]
df_all['ETsoy'] = [df_all['ETsoy'].iloc[i] if ~np.isnan(x) else df_all['ETsoy_solve'].iloc[i] for i,x in enumerate(df_all['ETsoy'])]

#omit days with large percent difference from tower data
df_all['ETcorn'] = [df_all['ETcorn'].iloc[i] if x<50 else np.nan for i,x in enumerate(df_all['LEpercent_diff'])]
df_all['ETsoy'] = [df_all['ETsoy'].iloc[i] if x<50 else np.nan for i,x in enumerate(df_all['LEpercent_diff'])]

#omit days with less than a threshold mm ET at either site
ETmin=0
df_all['ETcorn'] = [df_all['ETcorn'].iloc[i] if x>ETmin else np.nan for i,x in enumerate(df_all['ET_corr25'])]
df_all['ETsoy'] = [df_all['ETsoy'].iloc[i] if x>ETmin else np.nan for i,x in enumerate(df_all['ET_corr25'])]
df_all['ETcorn'] = [df_all['ETcorn'].iloc[i] if x>ETmin else np.nan for i,x in enumerate(df_all['ET_corr10'])]
df_all['ETsoy'] = [df_all['ETsoy'].iloc[i] if x>ETmin else np.nan for i,x in enumerate(df_all['ET_corr10'])]

#omit cases where ET is negative or greater than 10, set to tower value mean
df_all['ETcorn']= [df_all['ETcorn'].iloc[i] if x>0 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETcorn'])]
df_all['ETsoy']= [df_all['ETsoy'].iloc[i] if x>0 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETsoy'])]
df_all['ETcorn']= [df_all['ETcorn'].iloc[i] if x<10 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETcorn'])]
df_all['ETsoy']= [df_all['ETsoy'].iloc[i] if x<10 else df_all['ETmean'].iloc[i] for i,x in enumerate(df_all['ETsoy'])]

#fill nan values
df_fill = df_all.interpolate(method='linear')
#df_fill = df_all.ffill()
#df_fill=df_all.fillna(0)

df_all['ETcropdiff']=df_fill['ETcornTest']-df_fill['ETsoyTest']
df_all['ETheightdiff']=df_fill['ET_corr10']-df_fill['ET_corr25']
df_all['Fracdiff']=df_fill['f_soy10m']-df_fill['f_soy25m']

print(df_all[['ETcorn','ETsoy','ET_corr25','ET_corr10']].describe())

print(df_all[['ETcornTest','ETsoyTest','ET_corr25','ET_corr10']].describe())

plt.figure(1)
plt.plot(df_all.Date,df_all['f_soy10m_r'],'b')
plt.plot(df_all.Date,df_all['f_soy25m_r'],'r')
plt.ylim([-3,3])
plt.title('Soybean fraction')


plt.figure(2)
plt.subplot(1,3,1)
df_all[['ETcornTest','ETsoyTest']].boxplot()
plt.subplot(1,3,2)
plt.axis('square')
plt.plot(df_all['ETcornTest'],df_all['ETsoyTest'],'.')
plt.xlabel('ET corn')
plt.ylabel('ET soy')
plt.axis('square')
plt.subplot(1,3,3)
plt.plot(df_all['ETcornTest'],df_all['ETmean'],'.')
plt.xlabel('ET corn')
plt.ylabel('ET mean obs')
plt.axis('square')
plt.tight_layout()



plt.figure(3,figsize = (2.5,4))

n = 1 #number of days for moving average
n_min = 1

plt.subplot(2,1,1)
#rolling average
plt.plot(df_all['Date'], df_fill['ETcornTest'].rolling(n,min_periods=n_min).mean(),'b')
plt.plot(df_all['Date'],df_fill['ETsoyTest'].rolling(n,min_periods=n_min).mean(),'r')
plt.plot(df_all['Date'], df_fill['ET_corr25'].rolling(n,min_periods=n_min).mean(),':m')
plt.plot(df_all['Date'],df_fill['ET_corr10'].rolling(n,min_periods=n_min).mean(),':g')
plt.xlim([dt.datetime(2021,8,17),dt.datetime(2021,10,30)])
plt.xticks([])

plt.subplot(2,1,2)
#cumulative
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ET_corr10'].loc[df_fill['year']==2021].cumsum(),':g',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ET_corr25'].loc[df_fill['year']==2021].cumsum(),':m',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ETcornTest'].loc[df_fill['year']==2021].cumsum(),'b')
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['ETsoyTest'].loc[df_fill['year']==2021].cumsum(),'r')
plt.plot(df_fill['Date'].loc[df_fill['year']==2021],df_fill['Precip_Tot'].loc[df_fill['year']==2021].cumsum(),':k',linewidth=2)
plt.ylim([0,300])
plt.xticks(rotation=45)



plt.xlim([dt.datetime(2021,8,17),dt.datetime(2021,10,30)])
plt.tight_layout()

plt.savefig(fig_path+"CropET_fromTower_2021.svg", format="svg", bbox_inches="tight") 
plt.show()



plt.figure(4,figsize = (6,4))

plt.subplot(2,1,1)
plt.plot(df_all['Date'], df_fill['ETcornTest'].rolling(n,min_periods=n_min).mean(),'b')
plt.plot(df_all['Date'],df_fill['ETsoyTest'].rolling(n,min_periods=n_min).mean(),'r')
plt.plot(df_all['Date'], df_fill['ET_corr25'].rolling(n,min_periods=n_min).mean(),':m')
plt.plot(df_all['Date'],df_fill['ET_corr10'].rolling(n,min_periods=n_min).mean(),':g')
plt.xlim([dt.datetime(2022,4,1),dt.datetime(2022,9,30)])


plt.xticks([])


plt.subplot(2,1,2)

plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ET_corr10'].loc[df_fill['year']==2022].cumsum(),':g',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ET_corr25'].loc[df_fill['year']==2022].cumsum(),':m',linewidth=2)
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ETcornTest'].loc[df_fill['year']==2022].cumsum(),'b')
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['ETsoyTest'].loc[df_fill['year']==2022].cumsum(),'r')
plt.plot(df_fill['Date'].loc[df_fill['year']==2022],df_fill['Precip_Tot'].loc[df_fill['year']==2022].cumsum(),':k',linewidth=2)
plt.ylim([0,850])
plt.xticks(rotation=45)
plt.legend(['ET 10m','ET 25m','ET corn','ET soy','PPT'])


plt.xlim([dt.datetime(2022,4,1),dt.datetime(2022,9,30)])
plt.tight_layout()

plt.savefig(fig_path+"CropET_fromTower_2022.svg", format="svg", bbox_inches="tight") 
plt.show()

df_ECOSTRESS['Date']=pd.to_datetime(df_ECOSTRESS['Date'])
df_ECOSTRESS_clip = df_ECOSTRESS[~(df_ECOSTRESS['Date'] < '2021-09-01')]
df_ECOSTRESS_clip[['AlexiCorn','AlexiSoy','JdayCorn','JdaySoy']].describe()



#%%
   

