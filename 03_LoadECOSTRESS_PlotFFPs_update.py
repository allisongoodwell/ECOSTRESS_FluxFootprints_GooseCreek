#!/usr/bin/env python
# coding: utf-8

"""
Load ECOSTRESS data obtained from AppEARS application, make plots with FFP analysis of tower data
data for CINet flux tower region downloaded for: ALEXI daily product, PT-JPL daily and inst products
PT-JPL (ET in W/m2) contains an uncertainty product, ALEXI (ET in mm/day) contains a quality flag product

csv metadata files have file names, overpass times in UTC, and counts

input files: 
ECOSTRESS files
FFP pickle files
Energy-balance corrected ET data from 25m and 10m heights (used flux-qaqc pipeline to obtain from raw data)
footprint fractions csv file (from O2_Cropfractions_both_heights.py): 'GC_FootprintFractions.csv'
crop types and depressions maps

#output: csv file with ET estimates from each satellite product, tower, crop fractions for each image
#also do gridded averages
 

@author: Allison Goodwell
"""



# In[12]:


import gzip
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import rasterio
import datetime as dt

import warnings
warnings.filterwarnings('ignore')


#size of grid to consider (nn*2-1 is the length of cells in the large grid)
grid_sizes = [3, 13, 23]


tower_lon = -88.5773415
tower_lat = 40.1562377
tower_N = 4446215.92
tower_E = 365624.55

do_UTM_reproject = 0 #set to 1 to run the code to get the UTM versions of the ECOSTRESS data

data_path = 'ECOSTRESS_USRB'
FFP_path = 'GC_FFPs'
EB_corrected_path = 'DATA_Tower'
Maps_path = 'crops_depressions_maps/'
fig_path='Figure_Outputs/'
results_path = 'Results_performance'

save_name= results_path+'/ECOSTRESS_footprintETvalues.csv'

#FFP_path = '/Users/allisongoodwell/Dropbox/UCDenver/Github/CINet_Flux_Tower/footprint_codes/GC_FFPs'

PTJPL_filenames = [data_path + '/metadata/USRB2019/ECO3ETPTJPL-001-Statistics.csv',
                   data_path + '/metadata/USRB2021/ECO3ETPTJPL-001-Statistics.csv',
                   data_path + '/metadata/USRB2022/ECO3ETPTJPL-001-Statistics.csv']

ALEXI_filenames = [data_path + '/metadata/USRB2019/ECO3ETALEXI-001-Statistics.csv',
                   data_path + '/metadata/USRB2021/ECO3ETALEXI-001-Statistics.csv',
                   data_path + '/metadata/USRB2022/ECO3ETALEXI-001-Statistics.csv']

  
df_25m_EB = pd.read_csv(EB_corrected_path + '/CorrectedDaily_EBR_GC25mtower.csv')    
df_10m_EB = pd.read_csv(EB_corrected_path + '/CorrectedDaily_EBR_GC10mtower.csv') 
df_25m_EB['date']=pd.to_datetime(df_25m_EB['date'])
df_10m_EB['date']=pd.to_datetime(df_10m_EB['date']) 


df_fractions = pd.read_csv(results_path + '/GC_FootprintFractions.csv')  
df_fractions['Date']=pd.to_datetime(df_fractions['Date'])


df_cropET = pd.read_csv(EB_corrected_path + '/CropSpecificDailyTowerET.csv')
df_cropET['Date'] = pd.to_datetime(df_cropET['Date'])

  
PTJPL_stats = pd.DataFrame()
for i in range(3):
    PTJPL_stats = pd.concat([PTJPL_stats, pd.read_csv(PTJPL_filenames[i])],ignore_index='True')
    
ALEXI_stats = pd.DataFrame()
for i in range(3):
    ALEXI_stats = pd.concat([ALEXI_stats, pd.read_csv(ALEXI_filenames[i])],ignore_index='True')
        
#print(PTJPL_stats.columns)

ALEXI_stats['Centraldate']=pd.to_datetime(ALEXI_stats['Date'])
ALEXI_stats['Centraldate']=ALEXI_stats['Centraldate'].dt.tz_convert('US/Central')
ALEXI_stats['hour']= ALEXI_stats['Centraldate'].dt.hour
ALEXI_stats['year']= ALEXI_stats['Centraldate'].dt.year
ALEXI_stats['month'] = ALEXI_stats['Centraldate'].dt.month

PTJPL_stats['Centraldate']=pd.to_datetime(PTJPL_stats['Date'])
PTJPL_stats['Centraldate']=PTJPL_stats['Centraldate'].dt.tz_convert('US/Central')
PTJPL_stats['hour']= PTJPL_stats['Centraldate'].dt.hour
PTJPL_stats['year']= PTJPL_stats['Centraldate'].dt.year
PTJPL_stats['month'] = PTJPL_stats['Centraldate'].dt.month


stats_list = [PTJPL_stats, PTJPL_stats, ALEXI_stats]
filenames_list = [PTJPL_stats['File Name'], PTJPL_stats['File Name'],ALEXI_stats['File Name']]
type_string = ['ETinst','ETdaily','ETdaily']
paths = [data_path+'/JPL_ETinst/',data_path+'/JPL_ETdaily/',data_path+'/ALEXI_ETdaily/']

Raster_List_JPLInst={}
Raster_List_ALEXI={}
Raster_List_JPLDaily={}
Raster_List_ALEXIUncert={}
Raster_List_JPLUncert={}

print('loading ecostress images...')

conv_fact = 2450*1000/(60*60*24) #divide LE by this to get ET (W/m2 to mm per day)

#load ECOSTRESS images, put into list
for type_ct, file_names in enumerate(filenames_list):

    rasterlist=[]
    for ct,f in enumerate(file_names):

        #check for daytime overpasses or very cloudy periods, or very few pixel counts in whole image
        if stats_list[type_ct]['hour'][ct]>20 or stats_list[type_ct]['hour'][ct]<5 or stats_list[type_ct]['Count'][ct]<10000:
            continue        
            
        #edit file_name: should be ECO3ETPTJPL.001 instead of ECO3ETPTJPL_001, (ugh...)
        f=f.replace('_', '.',1)
 
        if type_ct ==0: # for JPL - also look at the uncertainty gridded product
            f_uncert = f
            f_uncert = f_uncert.replace('ETinst','ETinstUncertainty')
            path_u = data_path + '/ECOSTRESS_USRB/JPL_ETinst_uncertainty/'
        elif type_ct == 2: #for ALEXI - look at quality flag gridded product
            f_uncert = f
            f_uncert = f_uncert.replace('ETdaily','QualityFlag')
            path_u = data_path + '/ECOSTRESS_USRB/ALEXI_ETdaily_qualityflag/'


        if type_string[type_ct] in f:

            fname_utm = paths[type_ct] + f + '_UTM.tif'
            #print(fname_utm)
            
            if type_ct != 1:
                fname_u_utm = path_u + f_uncert + '_UTM.tif'
            #print(fname_u_utm)
            
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
                raster[raster==9999]=np.nan
                raster[raster>1000]=np.nan
                raster[raster<0]=0
                
            if type_ct !=1:
                try:
                    with rasterio.open(fname_u_utm) as src:
                        raster_uncert = src.read(1)
                        height = raster.shape[0]
                        width = raster.shape[1]
                        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                except:
                    raster_uncert=np.nan

            #check that tower location is in the range of the UTM coords 
            if tower_N<raster_Northing.min() or tower_N>raster_Northing.max():
                continue
            elif tower_E<raster_Easting.min() or tower_E>raster_Easting.max():
                continue
          
            if type_ct ==0:
                Raster_List_JPLInst[stats_list[type_ct]['Centraldate'][ct]] = [raster, raster_Easting, raster_Northing]
                
                
                raster_uncert = raster_uncert/conv_fact
                Raster_List_JPLUncert[stats_list[type_ct]['Centraldate'][ct]] = raster_uncert
                
            elif type_ct==1:
                #for JPL daily, convert to mm ET for ALEXI comparability
                raster = raster/conv_fact
                Raster_List_JPLDaily[stats_list[type_ct]['Centraldate'][ct]] =[raster, raster_Easting, raster_Northing]
            else:
                Raster_List_ALEXI[stats_list[type_ct]['Centraldate'][ct]] = [raster, raster_Easting, raster_Northing]
                Raster_List_ALEXIUncert[stats_list[type_ct]['Centraldate'][ct]] = raster_uncert
 
    
#only use dates where I find all 3 ECOSTRESS products
for date_val in Raster_List_JPLInst.copy():
    #look for JPLDaily and ALEXI rasters with same date key...
    if date_val not in Raster_List_JPLDaily or date_val not in Raster_List_ALEXI:
        del Raster_List_JPLInst[date_val]


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



# In[24]:

LE10_list=[]
LE25_list=[]
LE10_day_list=[]
LE25_day_list=[]
Grid_list=[]
Pixel_list=[]
FFP_list=[]
hour_list=[]
date_list=[]
ET10_mm=[]
ET25_mm=[]
ET10_mm_EB=[]
ET25_mm_EB=[]
ET_FFP_list=[]
Corn_list=[]
Soy_list=[]
Uncert_list=[]
ET_FFPpix_list=[]

Frac25_corn =[]
Frac25_soy =[]
Frac10_corn =[]
Frac10_soy =[]

ETcorn=[]
ETsoy=[]

#now want to plot for each date with footprints...
img_array = []

#counters for good vs bad images (based on pixels with ET computed)
good_image =[]

#loop through all images, find ET with different methods, compare to tower and FFP
for date_val in Raster_List_JPLInst:
    
    rasterJPLInst = Raster_List_JPLInst[date_val][0]
    rasterJPLUncert = Raster_List_JPLUncert[date_val]
    #date_val = rasterlist[1]
    print(date_val)
    
    raster_Easting = Raster_List_JPLInst[date_val][1]
    raster_Northing = Raster_List_JPLInst[date_val][2]
    
    rasterJPLdaily = Raster_List_JPLDaily[date_val][0]
    rasterAlexi = Raster_List_ALEXI[date_val][0]
    rasterAlexiUncert = Raster_List_ALEXIUncert[date_val]
    
    y_ind_tower = (np.abs(np.asarray(raster_Northing) - tower_N)).argmin()
    x_ind_tower = (np.abs(np.asarray(raster_Easting) - tower_E)).argmin()
    
    #extents for plotting
    raster_extent = [raster_Easting.min(), raster_Easting.max(), raster_Northing.min(), raster_Northing.max()]
    extent_tight = [tower_E - 1500, tower_E+1500, tower_N-1500, tower_N+1500]
    
    yr = date_val.year
    for i,y in enumerate(crop_yrs):
        if y==yr:
            cropraster = CropID_map[i] #pick crop map for that year

    #find FFC matching date_val (within the hour - maybe more than 1)
    #pickle files saved by month and year
    f_name_ffc = 'GC_ffps_30min_'+str(date_val.year)+'_'+str(date_val.month)+'_'+str(date_val.day)+'.pickle'
    #print(f_name_ffc)
       

    file_path = FFP_path + '/'+ f_name_ffc
    
    with gzip.open(file_path, "rb") as f:
        ffc_models_loaded = pickle.load(f)

    FFP25m_list = ffc_models_loaded['25m']['FFP']
    FFP10m_list = ffc_models_loaded['10m']['FFP']               
    inputdata = ffc_models_loaded['data']
    
    inputdata['Hour']=inputdata['Date'].dt.hour
    inputdata['Year']=inputdata['Date'].dt.year
    inputdata['Month']=inputdata['Date'].dt.month
    
    #get time of day index (half-hourly increments) based on hour and minute of overpass
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
     
    raster_pixels=[]
    grid_list=[]
    ET_ffp=[]
    CornET=[]
    SoyET=[]
    Uncert=[]
    ET_ffp_pixels=[]
    
    
    for i,(raster, raster_u) in enumerate([(rasterJPLInst, rasterJPLUncert),(rasterJPLdaily, rasterJPLUncert),(rasterAlexi, rasterAlexiUncert)]):
    #for i,(raster, raster_u) in enumerate([(rasterJPLInst, rasterJPLUncert),(rasterJPLdaily, rasterJPLUncert),(Alexi_JPL_raster, rasterAlexiUncert)]):
        
        raster[raster==9999]=np.nan
        raster[raster<-50]=np.nan
        raster[raster>1000]=np.nan
        
        raster_pixels.append(raster[y_ind_tower,x_ind_tower])
        
        
        raster_grids=[]
        for k,nn in enumerate(grid_sizes):
        
            raster_neighborhood = raster[y_ind_tower-nn+1:y_ind_tower+nn,x_ind_tower-nn+1:x_ind_tower+nn]            
            crop_neighborhood = cropraster[y_ind_tower-nn+1:y_ind_tower+nn,x_ind_tower-nn+1:x_ind_tower+nn]
    
            #use the 25x25 case to get crop ET averages, uncertainty estimation, flag for good images
            if nn==13: 
                
                corn_vals = raster_neighborhood[crop_neighborhood == 1]
                soy_vals = raster_neighborhood[crop_neighborhood == 2]
                
                if np.shape(raster_u) != ():
                    raster_u_neighborhood = raster_u[y_ind_tower-nn+1:y_ind_tower+nn,x_ind_tower-nn+1:x_ind_tower+nn]            
                    raster_u_avg = np.mean(raster_u_neighborhood)        
                else:
                    raster_u_avg = np.nan
                
                CornET.append(corn_vals)
                SoyET.append(soy_vals)
                Uncert.append(raster_u_avg)
               
            #flag for nan values or highly negative values in ET rasters 
            flag = np.sum(np.isnan(raster_neighborhood)) #check for nan values near tower
            
            if flag > ((nn*2-1)**2)*.75: #if 75% or more nan values in the grid
                if i==0:
                    print('raster has nan or bad values near tower, grid size: '+str(nn))
                grid_mean = np.nan
                raster_grids.append(grid_mean)
            else:
                grid_mean = np.nanmean(raster_neighborhood.flatten())
                raster_grids.append(grid_mean)
        
        grid_list.append(raster_grids)        
        
        #find FFP-weighted ET value
        for FFP in [FFP25m, FFP10m]:
            
            try:
                dx = FFP['x_2d'][0][1] - FFP['x_2d'][0][0]
                dy = FFP['y_2d'].T[0][1] - FFP['y_2d'].T[0][0]
        
                # Convert FFC footprint coordinates (tower = (0 m, 0 m)) to UTMs
                x_2d_utm = FFP['x_2d'] + tower_E
                y_2d_utm = FFP['y_2d'] + tower_N
                # indices of datacube x UTMs closest to FFC x UTMs
                x2d_near_ixs = [(np.abs(raster_Easting - x)).argmin() for x in x_2d_utm[0]]
                # indices of datacube y UTMs closest to FFC y UTMs
                y2d_near_ixs = [(np.abs(raster_Northing - y)).argmin() for y in y_2d_utm.T[0]]
    
                dc_nans = []
                flux_arr = np.empty(FFP['fclim_2d'].shape)
                #print('for i in %d ys' %len(y2d_near_ixs))
                for i in range(len(y2d_near_ixs)):
                    y = y2d_near_ixs[i]
                    # print(y)
                    #print('for j in %d xs' %len(x2d_near_ixs))
                    for j in range(len(x2d_near_ixs)):
                        x = x2d_near_ixs[j]
                        # print(cur_et[y][x])
                        flux = raster[y][x]  # flux for the grid cell
                        flux_arr[i][j] = flux
    
                scaler = (FFP['fclim_2d'] * dx * dy).sum()
                FFC_adj = FFP['fclim_2d'] / scaler
    
                # daily ET over entire footprint area
                et_f_arr = flux_arr * FFC_adj * dx * dy
                et_f = np.nansum(et_f_arr)
                mask = FFC_adj != 0
                cur_min, cur_max = np.nanmin(flux_arr[mask]), np.nanmax(flux_arr[mask])
                cur_area = len(et_f_arr.flatten())*dx*dy
                
                ET_ffp.append(et_f)
                ET_ffp_pixels.append(flux_arr.flatten())
                
            except:
                ET_ffp.append(np.nan)  
                ET_ffp_pixels.append(np.nan)
        
    
    ET_FFP_list.append(ET_ffp) #list of 6-item lists --> JPLInst_25m, JPLInst_10m...Alexi10m
    LE10_list.append(inputdata['LE10m'].iloc[t_index])
    LE25_list.append(inputdata['LE25m'].iloc[t_index])
    Pixel_list.append(raster_pixels)
    Grid_list.append(grid_list)
    FFP_list.append(0)
    hour_list.append(t_index/2)
    date_list.append(date_val)
    
    Corn_list.append(CornET)
    Soy_list.append(SoyET)
    Uncert_list.append(Uncert)
    
    ET_FFPpix_list.append(ET_ffp_pixels)   
    
    #find matching daily ET values from energy balance corrected files
    select_25mEB_row = df_25m_EB.loc[df_25m_EB['date']==dt.datetime(date_val.year,date_val.month,date_val.day)]
    select_10mEB_row = df_10m_EB.loc[df_10m_EB['date']==dt.datetime(date_val.year,date_val.month,date_val.day)]
    
    if select_10mEB_row.empty:
        ET10_mm_EB.append(np.nan)
        ET10_mm.append(np.nan)
    else:
        ET10_mm_EB.append(select_10mEB_row.iloc[0]['ET_corr'])
        ET10_mm.append(select_10mEB_row.iloc[0]['ET'])
    if select_25mEB_row.empty:
        ET25_mm_EB.append(np.nan)
        ET25_mm.append(np.nan)
    else:
        ET25_mm_EB.append(select_25mEB_row.iloc[0]['ET_corr'])
        ET25_mm.append(select_25mEB_row.iloc[0]['ET'])
      
    
    #select matching footprint crop and depression fractions for overpass time
    hr = date_val.hour
    minute = date_val.minute
    if minute > 45:
        hr = hr+1
        min_pick = 0
    elif minute <15:
        min_pick=0
    else:
        min_pick=30

    op_time = dt.datetime(date_val.year,date_val.month,date_val.day,hr,min_pick)
    select_fraction_rows = df_fractions.loc[df_fractions['Date']==op_time]
    
    Frac10_corn.append(select_fraction_rows.iloc[0]['f_corn10m'])
    Frac10_soy.append(select_fraction_rows.iloc[0]['f_soy10m'])
    Frac25_corn.append(select_fraction_rows.iloc[0]['f_corn25m'])
    Frac25_soy.append(select_fraction_rows.iloc[0]['f_soy25m'])  
    
    if flag >  ((13*2-1)**2)*.75: #don't make plots for bad dates with over 75% nan values
        good_image.append(0)
        continue
    else:
        good_image.append(1)
    
    #plot
    fig, ([ax1, ax2],[ax3, ax4]) = plt.subplots(2,2,figsize=(5.5,5))
    
    hours = np.arange(0,24,.5)  
    ax4.plot(hours,np.asfarray(inputdata['LE10m']),'b')
    ax4.plot(hours,np.asfarray(inputdata['LE25m']),'r')
    ax4.vlines(x=t_index/2,ymin=0,ymax=600,color='k',linestyle=":")
    ax4.axis(ymin=0,ymax=600)
    ax4.set_xticks(ticks = range(0,24,4))
    ax4.set_xticklabels(labels=['0','4','8','12','16','20'])
    #only plot inst comparison for JPLinst product
    ax4.plot(t_index/2+.1, raster_pixels[0], '.k',markersize=10)
    ax4.plot(t_index/2-0.05, ET_ffp[0],'*r',markersize=10)
    ax4.plot(t_index/2-0.1, ET_ffp[1],'*b',markersize=10)
    ax4.legend(['10m LE','25m LE','overpass','ET_1x1','ET_ffp25','ET_ffp10'],loc = 'upper left')
    ax4.set_title('Diurnal LE and JPL inst')

    
    uncertmin = np.mean(ET_ffp_pixels[2])-Uncert[1]/2
    uncertmax = uncertmin + Uncert[1]
    ax3.hlines(y=[uncertmin, uncertmax],xmin=1, xmax=4,color='k',linestyle=':')
    ax3.hlines(y = ET25_mm_EB[-1],xmin = 1, xmax=4,color='red',linestyle='--')
    ax3.hlines(y = ET10_mm_EB[-1],xmin=1, xmax=4,color='blue',linestyle='--')
    
    flierprops = dict(marker='.', markerfacecolor='k', markersize=6,
                  markeredgecolor='none')
    
    ax3.boxplot([CornET[1], SoyET[1], CornET[2],SoyET[2]],flierprops=flierprops)

    ax3.set_xticklabels(['PT_c','PT_s','AX_c','AX_s'])
    ax3.set_title('Crop-specific daily ET')
    ax3.axis(ymin=0,ymax=12)
    
    ax3.legend(['PT uncertainty range','ET 25m','ET 10m'])
    
    LE_vect = np.asarray(inputdata['LE25m'])
    if np.isnan(LE_vect[t_index]):
        LE_vect[t_index]=100
    

    
    for i,(ax,raster,ETvals) in enumerate([(ax1, rasterJPLdaily, CornET[1]),(ax2, rasterAlexi,CornET[2])]):
        #c0 = ax.imshow(raster,extent = raster_extent,cmap = 'viridis_r',vmin=minval[i],vmax=maxval[i])
        minval = np.mean(ETvals)-2
        maxval = np.mean(ETvals)+2
        c0 = ax.imshow(raster,extent = raster_extent,cmap = 'viridis_r',vmin=minval,vmax=maxval)
        ax.axis(xmin = extent_tight[0], xmax=extent_tight[1], ymin=extent_tight[2], ymax=extent_tight[3])
        ax.scatter(tower_E, tower_N,color='r')
        fig.colorbar(c0, ax=ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
        #now, also plot the footprint
        try:
            Easting_25m = FFP25m['x_2d']+tower_E
            Northing_25m = FFP25m['y_2d']+tower_N
            c1 = ax.contour(Easting_25m, Northing_25m, FFP25m['fclim_2d'], FFP25m['fr'][:5:-1], colors = 'red', linewidths=2, alpha = 1)
        
        except:
            if i ==0:
                print('no 25m footprint')
        try:
            Easting_10m = FFP10m['x_2d']+tower_E
            Northing_10m = FFP10m['y_2d']+tower_N
            c2 = ax.contour(Easting_10m, Northing_10m, FFP10m['fclim_2d'], FFP10m['fr'][:5:-1], colors = 'blue', linewidths=2, alpha = 1)
        except:
            if i==0:
                print('no 10m footprint')
      

    ax1.set_title('PTJPL '+ str(date_val))
    ax2.set_title('Alexi '+ str(date_val))

   
    plt.tight_layout()
    plt.savefig(fig_path + 'Fig_'+str(date_val)+'.svg', format="svg", bbox_inches="tight") 
    plt.show()


    

#%%
J_inst_pixel = [i[0] for i in Pixel_list]
J_daily_pixel = [i[1] for i in Pixel_list]
A_daily_pixel = [i[2] for i in Pixel_list]

J_inst_5x5 =   [i[0][0] for i in Grid_list]
J_inst_25x25 = [i[0][1] for i in Grid_list]
J_inst_50x50 = [i[0][2] for i in Grid_list]
J_daily_5x5 =  [i[1][0] for i in Grid_list]
J_daily_25x25 = [i[1][1] for i in Grid_list]
J_daily_50x50 = [i[1][2] for i in Grid_list]
A_daily_5x5 = [i[2][0] for i in Grid_list]
A_daily_25x25= [i[2][1] for i in Grid_list]
A_daily_50x50 = [i[2][2] for i in Grid_list]

J_inst_ffp25 = [i[0] for i in ET_FFP_list]
J_inst_ffp10 = [i[1] for i in ET_FFP_list]
J_daily_ffp25 = [i[2] for i in ET_FFP_list]
J_daily_ffp10 = [i[3] for i in ET_FFP_list]
A_daily_ffp25 = [i[4] for i in ET_FFP_list]
A_daily_ffp10 = [i[5] for i in ET_FFP_list]

J_inst_ffp25_pix = [i[0] for i in ET_FFPpix_list]
J_inst_ffp10_pix = [i[1] for i in ET_FFPpix_list]
J_daily_ffp25_pix = [i[2] for i in ET_FFPpix_list]
J_daily_ffp10_pix = [i[3] for i in ET_FFPpix_list]
A_daily_ffp25_pix = [i[4] for i in ET_FFPpix_list]
A_daily_ffp10_pix = [i[5] for i in ET_FFPpix_list]

J_inst_corn = [np.nanmean(i[0]) for i in Corn_list]
J_daily_corn = [np.nanmean(i[1]) for i in Corn_list]
A_daily_corn = [np.nanmean(i[2]) for i in Corn_list]
J_inst_soy = [np.nanmean(i[0]) for i in Soy_list]
J_daily_soy = [np.nanmean(i[1]) for i in Soy_list]
A_daily_soy = [np.nanmean(i[2]) for i in Soy_list]



data=np.asarray([date_list,hour_list,good_image,ET10_mm,ET25_mm,LE10_list, LE25_list,
                 ET10_mm_EB, ET25_mm_EB,
                 Frac10_corn, Frac25_corn, Frac10_soy, Frac25_soy, 
                 A_daily_pixel,A_daily_5x5, A_daily_25x25, A_daily_50x50, A_daily_ffp25, A_daily_ffp10,
                 J_daily_pixel,J_daily_5x5, J_daily_25x25, J_daily_50x50, J_daily_ffp25, J_daily_ffp10,
                 J_inst_pixel,J_inst_5x5, J_inst_25x25, J_inst_50x50, J_inst_ffp25, J_inst_ffp10,
                 J_inst_corn, J_inst_soy, J_daily_corn, J_daily_soy, A_daily_corn, A_daily_soy]).T

df_ALL = pd.DataFrame(data=data,
                  columns = ['Date','hour','good','ET10day','ET25day','LE10inst','LE25inst',
                             'ET10day_EB','ET25day_EB',
                             'fcorn_10','fcorn_25','fsoy_10','fsoy_25',
                             'AlexiET1x1','AlexiET5x5','AlexiET25x25','AlexiET50x50','AlexiFFP25','AlexiFFP10',
                             'JdayET1x1','JdayET5x5','JdayET25x25','JdayET50x50','JdayFFP25','JdayFFP10',
                             'JinstET1x1','JinstET5x5','JinstET25x25','JinstET50x50','JinstFFP25','JinstFFP10',
                             'JinstCorn','JinstSoy','JdayCorn','JdaySoy','AlexiCorn','AlexiSoy'])

for i,colname in enumerate(df_ALL):
    if colname != 'Date':
        df_ALL[colname] = pd.to_numeric(df_ALL[colname],errors='coerce')
 
df_ALL['AlexiCropDiff']=df_ALL['AlexiCorn']-df_ALL['AlexiSoy']
df_ALL['JdayCropDiff']=df_ALL['JdayCorn']-df_ALL['JdaySoy']    
 
    
df_ALL.to_csv(save_name) 



#%%



from scipy.stats import pearsonr

def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return pvalues

df_little = df_ALL[['AlexiCropDiff','JdayCropDiff']]

print(df_little.describe())

print(df_little.corr())



