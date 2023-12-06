#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:50:20 2023

This code reads in ECOSTRESS-FFP results from csv file (see code 03)
also loads fractions of crop types from csv file

analyzes model performance, makes figures

@author: allison
"""


import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import rasterio
import gzip
import performance_metrics as perf
import windrose
from matplotlib import colors
import seaborn as sns

results_path = 'Results_performance'
Maps_path = 'crops_depressions_maps/'
FFP_path = 'GC_FFPs/'
fig_path='Figure_Outputs/'

tower_lon = -88.5773415
tower_lat = 40.1562377
tower_N = 4446215.92
tower_E = 365624.55

df_ALL = pd.read_csv(results_path+'/ECOSTRESS_footprintETvalues_5x5.csv')
df_fractions = pd.read_csv(results_path + '/GC_FootprintFractions.csv')   

fname_utm = 'ECOSTRESS_USRB/ALEXI_ETdaily/ECO3ETALEXI.001_EVAPOTRANSPIRATION_ALEXI_ETdaily_doy2022109185146_aid0001_UTM.tif'
#randomly selected ECOSTRESS image (just to get the easting and northing coordinates for plotting later)


df_ALL['Date']=pd.to_datetime(df_ALL['Date'])

df_ECO_all = df_ALL
df_ECOSTRESS = df_ALL[(df_ALL['good']==1)]
df_ECO_allgood = df_ALL.dropna()

#performance file to be saved as a csv:
perf_name = results_path+'/Performance_5x5grid.csv'


for i,colname in enumerate(df_ALL):
    if colname != 'Date':
        df_ALL[colname] = pd.to_numeric(df_ALL[colname],errors='coerce')


    
#subsets of main dataframe for correlation analysis...
df_nodate = df_ALL.drop('Date',axis=1)
df_good = df_ALL[df_ALL['good']==1]





#%% plot correlation heat maps for different ECOSTRESS products relative to tower data

#different combinations...
df_alexi =    df_good[['ET10day','ET25day','ET10day_EB','ET25day_EB','AlexiET1','AlexiETg','AlexiFFP10','AlexiFFP25']]
df_jpldaily = df_good[['ET10day','ET25day','ET10day_EB','ET25day_EB','JdayET1','JdayETg','JdayFFP10','JdayFFP25']]
df_jplinst = df_good[['LE25inst','LE25inst','JinstET1','JinstETg','JinstFFP10','JinstFFP25']]
df_compare = df_good[['ET10day','ET25day','ET10day_EB','ET25day_EB','AlexiETg','JdayETg']]


plt.figure(1,figsize=(10,8), dpi= 80)  
sns.heatmap(df_alexi.corr(), xticklabels=df_alexi.corr().columns, yticklabels=df_alexi.corr().columns, cmap='RdYlGn', center=0, annot=True)

#sns.pairplot(df_alexi, kind="scatter",diag_kind = 'kde')
plt.show()

plt.figure(2,figsize=(10,8), dpi= 80)  
sns.heatmap(df_jpldaily.corr(), xticklabels=df_jpldaily.corr().columns, yticklabels=df_jpldaily.corr().columns, cmap='RdYlGn', center=0, annot=True)

#sns.pairplot(df_jpldaily, kind="scatter",diag_kind = 'kde')
plt.show()

plt.figure(3,figsize=(10,8), dpi= 80)  
sns.heatmap(df_compare.corr(), xticklabels=df_compare.corr().columns, yticklabels=df_compare.corr().columns, cmap='RdYlGn', center=0, annot=True)

#sns.pairplot(df_compare, kind="scatter",diag_kind = 'kde')
plt.show()

plt.figure(4,figsize=(10,3), dpi= 80) 

#%% plot and save figure showing crop-specific differences, RMSE, and ET_ffp for daily products

#df_good['ET_tower']  = df_good[['ET25day_EB', 'ET10day_EB']].mean(axis=1,skipna=True)

df_good['ET_tower']= df_good['ET25day_EB']
#df_good['ET_tower']= df_good['ET25day'] (use for non-energy balance corrected case)

#df_good['LE_tower'] = df_good[['LE10inst','LE25inst']].mean(axis=1,skipna=True)
df_good['LE_tower'] = df_good['LE25inst']

df_good['dep_avg'] = df_good[['fdep_25','fdep_10']].mean(axis=1,skipna=True)

df_crops = df_good[['ET_tower','LE_tower','JdayCorn','JdaySoy','JinstCorn','JinstSoy','AlexiCorn','AlexiSoy','AlexiFFP25','JdayFFP25','JinstFFP25','dep_avg']]
df_crops = df_crops.dropna()

df_dep = df_good[['ET_tower','LE_tower','JdayDep','JdayNoDep','JinstDep','JinstNoDep','AlexiDep','AlexiNoDep','AlexiFFP25','JdayFFP25','JinstFFP25','dep_avg']]
df_dep = df_dep.dropna()

cvectET = np.asfarray(df_crops['ET_tower'])
cvectLE = np.asfarray(df_crops['LE_tower'])

Alexi_err = np.sqrt((df_crops['AlexiFFP25']-df_crops['ET_tower'])**2)
JPL_err = np.sqrt((df_crops['JdayFFP25']-df_crops['ET_tower'])**2)
JPLinst_err = np.sqrt((df_crops['JinstFFP25']-df_crops['LE_tower'])**2)
Alexi_err_dep = np.sqrt((df_dep['AlexiFFP25']-df_dep['ET_tower'])**2)
JPL_err_dep = np.sqrt((df_dep['JdayFFP25']-df_dep['ET_tower'])**2)
JPLinst_err_dep = np.sqrt((df_dep['JinstFFP25']-df_dep['LE_tower'])**2)

cropdiff_Alexi = (df_crops['AlexiCorn']-df_crops['AlexiSoy'])#/df_crops['AlexiCorn']
cropdiff_JPL = (df_crops['JdayCorn']-df_crops['JdaySoy'])#/df_crops['JdayCorn']

depdiff_Alexi = (df_dep['AlexiDep']-df_dep['AlexiNoDep'])#/df_crops['AlexiCorn']
depdiff_JPL = (df_dep['JdayDep']-df_dep['JdayNoDep'])#/df_crops['JdayCorn']


plt.figure(5,figsize=(8,3), dpi= 80) 
plt.subplot(1,2,1)
plt.scatter(np.asfarray((df_crops['AlexiCorn']-df_crops['AlexiFFP25'])/df_crops['AlexiFFP25']),
            np.asfarray((df_crops['AlexiSoy']-df_crops['AlexiFFP25'])/df_crops['AlexiFFP25']),c=cropdiff_Alexi, s=Alexi_err*50,vmin=-1,vmax=1,cmap='seismic',edgecolors='black')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.plot([-1,1],[0,0],'k')
plt.plot([0,0],[-1,1],'k')
plt.xlabel('$\dfrac{ET_{corn}-ET_{ffc}}{ET_{ffc}}$')
plt.ylabel('$\dfrac{ET_{soy}-ET_{ffc}}{ET_{ffc}}$')
plt.title('ALEXI',fontsize = 10)
plt.colorbar()

plt.subplot(1,2,2)
plt.scatter(np.asfarray((df_crops['JdayCorn']-df_crops['JdayFFP25'])/df_crops['JdayFFP25']),
            np.asfarray((df_crops['JdaySoy']-df_crops['JdayFFP25'])/df_crops['JdayFFP25']),c=cropdiff_JPL, s=JPL_err*50,vmin=-1,vmax=1,cmap='seismic',edgecolors='black')
plt.colorbar()
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.plot([-1,1],[0,0],'k')
plt.plot([0,0],[-1,1],'k')
plt.xlabel('$\dfrac{ET_{corn}-ET_{ffc}}{ET_{ffc}}$')
plt.title('PT-JPL',fontsize = 10)

plt.tight_layout()
plt.savefig("ALEXI_JPL_cropscatter_5x5.pdf", format="pdf", bbox_inches="tight")
plt.show()


#%% performance metrics: KGE, NSE, RMSE, save as a csv file

obsET = df_good['ET_tower'] #for daily products (mm)
obsLE = df_good['LE_tower'] #for PT-JPL inst product (W/m2)

sim_list = [df_good['JdayFFP25'],df_good['JdayET1'],df_good['JdayETg'],
            df_good['AlexiFFP25'],df_good['AlexiET1'],df_good['AlexiETg'],
            df_good['JinstFFP25'],df_good['JinstET1'],df_good['JinstETg']]

df_perf = pd.DataFrame(index=['KGE','corr','alpha','beta','NSE','RMSE'])

for ct,sim in enumerate(sim_list):
  
    if ct<6:
        df_mini = pd.DataFrame(data = [obsET,sim]).T
    else:
        df_mini = pd.DataFrame(data = [obsLE,sim]).T
    
    sim_name = df_mini.columns[1]
    
    df_mini.columns = ['obs','sim']
    
    df_mini = df_mini.dropna()
    
    obs_nonans = df_mini['obs']
    sim_nonans = df_mini['sim']
    
    KGE, correlation, alpha, beta = perf.KGE(obs_nonans,sim_nonans)
    NSE = perf.NSE(obs_nonans,sim_nonans)
    RMSE = perf.RMSE(obs_nonans,sim_nonans)
    
    df_perf[sim_name]=[KGE, correlation, alpha,beta,NSE,RMSE]


df_perf.to_csv(perf_name,float_format='%.2f')

#%% crop maps, look at "very best" images with all values intact

df_everythinggood = df_good.dropna()
#only 9 images with "everything"!
good_dates = df_everythinggood['Date']


#use the UTM version of the ECOSTRESS rasters...
with rasterio.open(fname_utm) as src:
    raster = src.read(1)
    height = raster.shape[0]
    width = raster.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    raster_Easting = np.array(xs)[0,:]
    raster_Northing = np.array(ys)[:,0]
    #print(src.crs)

#create a raster with same dimensions as ECOSTRESS raster, with crop type identified
#load the crop raster pickle file - indexed by year
with open(Maps_path + 'cropmapsIL.pickle', 'rb') as pickle_file:
    crop_frames = pickle.load(pickle_file)
    [crop_rasters, UTM_Easting_crop, UTM_Northing_crop, years] = crop_frames
    
        
#want to determine fractions for the footprint according to the raster...
y2d_near_ixs_crop = [(np.abs(UTM_Northing_crop - y)).argmin() for y in raster_Northing] # indices of crop x UTMs closest to FFC x UTMs
x2d_near_ixs_crop = [(np.abs(UTM_Easting_crop - x)).argmin() for x in raster_Easting] # indices of crop y UTMs closest to FFC y UTMs

CropID_map = []
crop_yrs = years;
for yr_ct,yr in enumerate(crop_yrs):
    cropraster = crop_rasters[yr_ct]
# Iterate over the FFC grid defined by x2d_utms, y2d_utms and extract Datacube ET at those locations:
    crop_arr = np.empty(raster.shape)    
    
    for i,y in enumerate(y2d_near_ixs_crop):
        for j,x in enumerate(x2d_near_ixs_crop):
            crop_pixel = cropraster[y,x]
            crop_arr[i,j] = crop_pixel  

    CropID_map.append(crop_arr)
       


cropcmap = colors.ListedColormap(['white', 'yellow','green']) #1 = corn (yellow), 2= soy (green)
bounds=[0,1,2,3]
norm = colors.BoundaryNorm(bounds, cropcmap.N)


for date_val in good_dates:
    print(date_val)
    
    yr = date_val.year
    for i,y in enumerate(crop_yrs):
        if y==yr:
            cropraster = CropID_map[i] #pick crop map for that year
            
            
    f_name_ffc = 'GC_ffps_30min_'+str(date_val.year)+'_'+str(date_val.month)+'_'+str(date_val.day)+'.pickle'
    #print(f_name_ffc)
    


    file_path = FFP_path + '/'+ f_name_ffc
    
    with gzip.open(file_path, "rb") as f:
        ffc_models_loaded = pickle.load(f)

    FFP25m_list = ffc_models_loaded['25m']['FFP']
    FFP10m_list = ffc_models_loaded['10m']['FFP']               
    inputdata = ffc_models_loaded['data']
    
    hr = date_val.hour
    minute = date_val.minute
    if minute > 45:
        hr = hr+1
        t_index = hr*2
    elif minute <15:
        t_index = hr*2
    else:
        t_index = hr*2+1
       
    FFP25m = FFP25m_list[t_index]
    FFP10m = FFP10m_list[t_index]
    
    #check if pixel near tower is a nan, then skip if it is...
    y_ind_tower = (np.abs(np.asarray(raster_Northing) - tower_N)).argmin()
    x_ind_tower = (np.abs(np.asarray(raster_Easting) - tower_E)).argmin()
    
    raster_extent = [raster_Easting.min(), raster_Easting.max(), raster_Northing.min(), raster_Northing.max()]
    extent_tight = [tower_E - 1500, tower_E+1500, tower_N-1500, tower_N+1500]


    (fig, ax) = plt.subplots(1,1,figsize=(14, 8))
    c0 = ax.imshow(cropraster,extent = raster_extent, cmap=cropcmap, norm=norm)
    ax.axis(xmin = extent_tight[0], xmax=extent_tight[1], ymin=extent_tight[2], ymax=extent_tight[3])
    ax.scatter(tower_E, tower_N,color='r')
    ax.set_title(date_val)
    #fig.colorbar(c0, ax=ax)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])

    #now, also plot the footprint
    try:
        Easting_25m = FFP25m['x_2d']+tower_E
        Northing_25m = FFP25m['y_2d']+tower_N
        c1 = ax.contour(Easting_25m, Northing_25m, FFP25m['fclim_2d'], FFP25m['fr'][::-1], colors = 'red', linewidths=3, alpha = 1)
    
    except:
        if i ==0:
            print('no 25m footprint')
    try:
        Easting_10m = FFP10m['x_2d']+tower_E
        Northing_10m = FFP10m['y_2d']+tower_N
        c2 = ax.contour(Easting_10m, Northing_10m, FFP10m['fclim_2d'], FFP10m['fr'][::-1], colors = 'blue', linewidths=3, alpha = 1)
    except:
        if i==0:
            print('no 10m footprint')
            
    plt.tight_layout()
    plt.savefig(fig_path + 'Fig_CropMap_'+str(date_val)+'.svg', format="svg", bbox_inches="tight")  
    plt.show()


#%% performance for different sized grids (load in csv file that combines performance metrics from multiple runs of code 03_LoadECOSTRESS)


df_perf_all = pd.read_csv(results_path+'/Performance_Allgrids.csv')
df_perf_all = df_perf_all.set_index('Unnamed: 0')
RMSE = df_perf_all.loc['RMSE']

J_FFP = RMSE.loc['JdayFFP25']
A_FFP = RMSE.loc['AlexiFFP25']

J_ratios = np.asarray(RMSE[1:5]/J_FFP)
A_ratios = np.asarray(RMSE[6:10]/A_FFP)

plt.figure(figsize=(3.25,2.5))
plt.plot(J_ratios,'-o')
plt.plot(A_ratios,'-*')
plt.hlines(1,0,3,color='k')
plt.title('RMSE(NxN) / RMSE(FFP)',fontsize=10)
plt.xlabel('NxN grid area')
plt.xticks([0,1,2,3],labels=['1x1','5x5','25x25','50x50'])
plt.legend(['PT-JPL','ALEXI'])
plt.tight_layout()
plt.savefig(fig_path + 'Fig_RMSEratio.pdf', format="pdf", bbox_inches="tight")  
plt.show()




#%%
df_fractions['Date']=pd.to_datetime(df_fractions['Date']) 

df_fractions = df_fractions[df_fractions['Date']> pd.datetime(2019,4,1)]
df_fractions = df_fractions[df_fractions['Date']< pd.datetime(2022,11,1)]

for i,colname in enumerate(df_fractions):
    if colname != 'Date':
        df_fractions[colname] = pd.to_numeric(df_fractions[colname],errors='coerce')
   
 
        
#%% plot crop fractions over time - also plot dates of ECOSTRESS overpasses

n = 48*10
    
(fig, ax)=plt.subplots(1,1,figsize=(10, 3))
c1=ax.plot(df_fractions.Date,df_fractions.f_corn25m.rolling(n,min_periods=48).mean(),'g',label='25m')
c2=ax.plot(df_fractions.Date,df_fractions.f_corn10m.rolling(n,min_periods=48).mean(),'b',label='10m')

ax.plot(df_ECO_all['Date'],np.zeros((len(df_ECO_all['Date']),1)), '.',color='gray')
ax.plot(df_ECOSTRESS['Date'],.025+np.zeros((len(df_ECOSTRESS['Date']),1)), '.k')
ax.plot(df_ECO_allgood['Date'],.05+np.zeros((len(df_ECO_allgood['Date']),1)), '.r')


plt.title('Corn fraction (10-day moving avg)')
for y in range(2017,2023):
    ax.fill_between(df_fractions.Date, .98, .01, 
                where=df_fractions.Date.dt.month.le(3) | df_fractions.Date.dt.month.ge(11),
                color='gray', alpha=0.9, transform=ax.get_xaxis_transform())
    
ax.set_xlim(xmin = pd.datetime(2019,4,1), xmax=pd.datetime(2022,11,1))

ax.legend(['25m','10m','ECO','ECO good','ECO+towers'])


plt.savefig(fig_path+"All_time_crop_fractions.pdf", format="pdf", bbox_inches="tight") 


#%%



df_fractions['hour']=pd.DatetimeIndex(df_fractions['Date']).hour
df_fractions['year']=pd.DatetimeIndex(df_fractions['Date']).year
df_fractions['month']=pd.DatetimeIndex(df_fractions['Date']).month

dfday = df_fractions.loc[(df_fractions.hour >7) & (df_fractions.hour<20) & (df_fractions.year==2021) & (df_fractions.month >3) & (df_fractions.month<11)]


plt.plot(dfday.f_corn10m,dfday.f_corn25m,'.')
plt.xlabel('10m corn fraction')
plt.ylabel('25m corn fraction')

LE_diff = np.asfarray(dfday.LE10m/dfday.LE25m)
frac_diff = np.asfarray(dfday.f_corn10m/dfday.f_corn25m)
dep_diff = np.asfarray(dfday.f_dep10m/dfday.f_dep25m)
A_diff = np.asfarray(dfday.Area10m/dfday.Area25m)

WD = np.asfarray(dfday.WD25m)

#get rid of some values that are far off

frac_diff[frac_diff>100]=np.nan #10m has 100x more corn
frac_diff[frac_diff<1/100]=np.nan #25m has 100x more corn
dep_diff[dep_diff>100]=np.nan
dep_diff[dep_diff<1/100]=np.nan

frac_diff[np.isnan(dep_diff)]=np.nan
dep_diff[np.isnan(frac_diff)]=np.nan

LE_diff[np.isnan(frac_diff)]=np.nan

# Create Fig and gridspec
fig = plt.figure(figsize=(16, 10), dpi= 80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

# Scatterplot on main ax
ax_main.scatter(LE_diff, frac_diff, c=WD, cmap = 'hsv',alpha=.9, edgecolors='black', linewidths=.5)
ax_main.axhline(y = 1, color = 'r', linestyle = '-')
ax_main.axvline(x = 1, color = 'r', linestyle = '-')
ax_main.axis(xmin = 0, xmax=2, ymin=0, ymax=3)


# Add a graph in each part
sns.boxplot(y=frac_diff, ax=ax_right, orient="v")
sns.boxplot(x=LE_diff, ax=ax_bottom, orient="h")

ax_bottom.axis(xmin = 0, xmax=2)
ax_right.axis(ymin=0,ymax=3)

# Decorations ------------------
# Remove x axis name for the boxplot
ax_bottom.set(xlabel='')
ax_right.set(ylabel='')

# Main Title, Xlabel and YLabel
ax_main.set(title='LE vs corn fractions', xlabel='LE10/LE25', ylabel='f10/f25')

# Set font size of different components
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)


fig = plt.figure(figsize =(5,3))
plt.scatter(LE_diff, frac_diff, c=WD, cmap = 'hsv')
plt.colorbar()
plt.hlines(1, xmin=0, xmax=2,color='k')
plt.vlines(1, ymin=0, ymax=3,color='k')
plt.xlim([0,2])
plt.ylim([0,3])
#ax_main.axhline(y = 1, color = 'r', linestyle = '-')
#ax_main.axvline(x = 1, color = 'r', linestyle = '-')
#ax_main.axis(xmin = 0, xmax=2, ymin=0, ymax=3)
plt.xlabel('LE10/LE25')
plt.ylabel('fc10/fc25')
plt.title('Colored by wind direction')


    
plt.savefig(fig_path+"LE_vs_cornfracs.svg", format="svg", bbox_inches="tight") 

plt.show()




# Create Fig and gridspec
fig = plt.figure(figsize=(16, 16), dpi= 80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

# Scatterplot on main ax
ax_main.scatter(LE_diff, A_diff, alpha=.9, edgecolors='black', linewidths=.5)
ax_main.axhline(y = 1, color = 'r', linestyle = '-')
ax_main.axvline(x = 1, color = 'r', linestyle = '-')
ax_main.axis(xmin = 0, xmax=2, ymin=0, ymax=5)


# Add a graph in each part
sns.boxplot(y=A_diff, ax=ax_right, orient="v")
sns.boxplot(x=frac_diff, ax=ax_bottom, orient="h")

ax_bottom.axis(xmin = 0, xmax=2)
ax_right.axis(ymin=0,ymax=5)

# Decorations ------------------
# Remove x axis name for the boxplot
ax_bottom.set(xlabel='')
ax_right.set(ylabel='')

# Main Title, Xlabel and YLabel
ax_main.set(title='LE vs area fractions', xlabel='LE10/25', ylabel='A10/A25')

# Set font size of different components
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)
plt.show()


print(np.nanmedian(LE_diff))
print(np.nanmedian(frac_diff))
print(np.nanmedian(dep_diff))
print(np.nanmedian(A_diff))

#%% plot fetch area roses, also histograms for LE and crop fractions

from matplotlib import cm


df_fractions['Hour']= df_fractions['Date'].dt.hour
df_fractions['Year'] = df_fractions['Date'].dt.year
df_fractions['Month'] = df_fractions['Date'].dt.month

dfrose = df_fractions
dfrose = dfrose[(dfrose['Hour']>7) & (dfrose['Hour']<20)] #daytime only

dfrose = dfrose[(dfrose['Month']>3) & (dfrose['Month']<11)] #growing season (approx) only

dfrose['Area10m']= dfrose['Area10m']/(1000**2) #square km
dfrose['Area25m']= dfrose['Area25m']/(1000**2) #square km

dfrose['LEdiff'] = dfrose['LE10m']/dfrose['LE25m']
dfrose['fcdiff'] = dfrose['f_corn10m']/dfrose['f_corn25m']

critval10 = dfrose.Area10m.mean() + 4*dfrose.Area10m.std()
critval25 = dfrose.Area25m.mean() + 4*dfrose.Area25m.std()
dfrose = dfrose[(dfrose['Area10m']<=critval10)]
dfrose = dfrose[(dfrose['Area25m']<=critval25)]

dfrose = dfrose.dropna(subset = ['LE10m', 'LE25m', 'f_corn10m','f_corn25m'])

dfrose_orig=dfrose


for y in range(2021,2023):
    dfrose = dfrose_orig[(dfrose_orig['Year']==y)] #1-year only

    wd10 = dfrose.WD10m
    a10 = dfrose.Area10m
    wd25 = dfrose.WD25m
    a25 = dfrose.Area25m
    le10 = dfrose.LE10m
    le25 = dfrose.LE25m
    lediff = dfrose.LEdiff
    fcdiff = dfrose.fcdiff
    
    abins = [0, 0.1, .2, .5, 1, 2, 3]
    
    ax1 = windrose.WindroseAxes.from_ax()
    ax1.bar(wd10, a10, opening=1, normed=False, bins=abins, edgecolor="white")
    ax1.set_legend(fontsize=20,loc='upper right')
    ax1.set_title('10m height FFP Area')
    
    plt.savefig(fig_path+str(y)+"AreaRose_10.svg", format="svg", bbox_inches="tight") 
    
    
    ax2 = windrose.WindroseAxes.from_ax()
    ax2.bar(wd25, a25, opening=1,normed=False, bins=abins, edgecolor="white")
    ax2.set_title('25m height FFP Area')
    ax2.set_legend(fontsize=20,loc='upper right')
    
    plt.savefig(fig_path+str(y)+"AreaRose_25.svg", format="svg", bbox_inches="tight") 
    
    
    lebins = [0, 50, 100, 200, 300]
    
    ax3 = windrose.WindroseAxes.from_ax()
    ax3.bar(wd10, le10, opening=1, normed=False, bins=lebins, edgecolor="white")
    ax3.set_title('10m height LE')
    ax3.set_legend(fontsize=20,loc='upper right')
    
    plt.savefig(fig_path+str(y)+"LERose_10.svg", format="svg", bbox_inches="tight") 
    
    
    ax4 = windrose.WindroseAxes.from_ax()
    ax4.bar(wd25, le25, opening=1,normed=False, bins = lebins, edgecolor="white")
    ax4.set_title('25m height LE')
    ax4.set_legend(fontsize=20,loc='upper right')
    
    plt.savefig(fig_path+str(y)+"LERose_25.svg", format="svg", bbox_inches="tight") 
    
    
    
    dbins = [0, .5, .9, 1.1,2]
    ax5 = windrose.WindroseAxes.from_ax()
    ax5.bar(wd25, lediff, opening=1,normed=False,bins=dbins, edgecolor="black",cmap = cm.bwr)
    ax5.set_title('LE10/LE25')
    ax5.set_legend(fontsize=20,loc='upper right')
    
    plt.savefig(fig_path+str(y)+"LEfracRose.svg", format="svg", bbox_inches="tight") 
    
    

    ax5 = windrose.WindroseAxes.from_ax()
    ax5.bar(wd25,fcdiff, opening=1,normed=False,bins=dbins, edgecolor="black",cmap = cm.bwr)
    ax5.set_title('f10/f25')
    ax5.set_legend(fontsize=20,loc='upper right')
    
    plt.savefig(fig_path+str(y)+"CropfracRose.svg", format="svg", bbox_inches="tight") 

