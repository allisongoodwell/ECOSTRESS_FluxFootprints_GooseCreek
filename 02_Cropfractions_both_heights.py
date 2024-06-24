# -*- coding: utf-8 -*-
"""

This code loads in FFP results (daily pickle files)
and computes contributing fractions of crop type (corn, soybean)

files needed: daily pickle files from FFP results
crop maps from USDA CDL

half-hourly results output to csv file: 'GC_FootprintFractions.csv'

"""

import gzip
import pickle
import pandas as pd
import numpy as np
import rasterio

years = range(2019,2023)
months = range(1,13)
towername = 'GC'
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] 

#need both UTM and geographic coordinates for flux tower
tower_E = 365624.55
tower_N = 4446215.92

tower_lon = -88.5773415
tower_lat = 40.1562377

tower_coords = [tower_E, tower_N, tower_lon, tower_lat]

Maps_path = 'crops_depressions_maps/'
FFP_path = 'GC_FFPs/'


#function that inputs crop map and FFC data to get fractions and area of FFPs
def cropfractions(crop_frames, FFC_list, date_list, tower_coords, dx):
    
    df_out = pd.DataFrame(columns=['Date','Area', 'f_corn', 'f_soy'])  
    [tower_E, tower_N, tower_lon, tower_lat] = tower_coords
    [rasters, UTM_E, UTM_N, years_crop] = crop_frames #upack list of rasters, coords, years
     
    for ct,FFP in enumerate(FFC_list):
        
        ffc_date = date_list[ct]
        #print(ffc_date.year)
          
        #pick appropriate crop map raster based on years
        for i,y in enumerate(years_crop):
            if ffc_date.year==y:
                #print(ffc_date.year)
                raster = rasters[i]
        
        #if np.mod(ct,48)==0: print(ffc_date)

        if FFP == 'no footprint':   #move to next footprint       
            df_out.loc[ffc_date,:] = [ffc_date,np.nan, np.nan, np.nan]
            continue
        
        x_2d, y_2d, fs, cs = FFP['x_2d'], FFP['y_2d'], FFP['fclim_2d'], FFP['fr']    

        #convert footprint fractions (mutiply by grid size, divide by total to sum to 1)
        ffpfracs = fs*dx**2            
        ffpfracs = ffpfracs/np.nansum(ffpfracs) 

        EastingFFP = tower_E + x_2d
        NorthingFFP = tower_N + y_2d

        #want to determine fractions for the footprint according to the raster...
        y2d_near_ixs_crop = [(np.abs(UTM_N - y)).argmin() for y in NorthingFFP[:,0]] # indices of crop x UTMs closest to FFC x UTMs
        x2d_near_ixs_crop = [(np.abs(UTM_E - x)).argmin() for x in EastingFFP[0,:]] # indices of crop y UTMs closest to FFC y UTMs

        # Iterate over the FFC grid defined by x2d_utms, y2d_utms and extract Datacube ET at those locations:
        crop_arr = np.empty(FFP['fclim_2d'].shape)    

        for i,y in enumerate(y2d_near_ixs_crop):
            for j,x in enumerate(x2d_near_ixs_crop):
                crop_pixel = raster[y,x]
                crop_arr[i,j] = crop_pixel  

        #next, find area of footprint and crop and depression fractions   
        fs_masked = fs
        fs_masked[np.where(fs_masked ==0)] = np.nan
        A = np.count_nonzero(~np.isnan(fs_masked))*dx**2

        f_soy = np.nansum([x for i,x in enumerate(ffpfracs.flatten()) if crop_arr.flatten()[i] ==2])
        f_corn = np.nansum([x for i,x in enumerate(ffpfracs.flatten()) if crop_arr.flatten()[i] ==1])

        df_out.loc[ffc_date,:] = [ffc_date, A, f_corn, f_soy]
              
    return df_out


#%% crop data loading
already_pickled = 1

if already_pickled ==1:
    #load the crop raster pickle file - indexed by year
    with open(Maps_path + 'cropmapsIL.pickle', 'rb') as pickle_file:
        crop_frames = pickle.load(pickle_file)
        [crop_rasters, UTM_Easting_crop, UTM_Northing_crop, years_crop] = crop_frames
else: #load individual .tif files from USDA CDL
    crop_rasters = []
    years_crop = range(2008,2023)
    for y in years_crop:
        file_name = Maps_path + 'IL//CDL_'+ str(y) +'_clip_20230503161913_1691951662.tif'
        #file_name = './crops/NE/CDL_'+ str(y) +'_clip_20230503164152_936002847.tif'
        
        with rasterio.open(file_name) as src:
            crop_raster=src.read(1) 

            crop_raster = crop_raster.astype('float')
            crop_raster[crop_raster>5]=0       #make into a binary raster: values of 1 == corn, 2= soybean (formerly 5), 0 = no crop
            crop_raster[crop_raster<1]=np.nan
            crop_raster[crop_raster==5]=2
            crop_raster[crop_raster>2]=0
            
            #these are the same for all the clipped files
            height = crop_raster.shape[0]
            width = crop_raster.shape[1]
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            UTM_Easting_crop = np.array(xs)[0,:]
            UTM_Northing_crop = np.array(ys)[:,0]
        
        crop_rasters.append(crop_raster) #append to get list based on years
    
    pickle.dump([crop_rasters, UTM_Easting_crop, UTM_Northing_crop,years_crop], open(Maps_path + "cropmapsIL.pickle", "wb" ))

    crop_frames = [crop_rasters, UTM_Easting_crop, UTM_Northing_crop, years_crop]   

#%%

crop25_list=[]
crop10_list=[]
input_list=[]

for y in years:
    for ct_m,m in enumerate(months):
        
        ndays = days_in_month[ct_m]
        for d in range(1,ndays+1):
        
        
            #load pickle file
            filename =  '%s_ffps_30min_%s_%s_%s.pickle' %(towername, y, m,d)
            print(filename)
            #extract all of their tags, pick the right file and open it

            with gzip.open(FFP_path+filename, "rb") as f:
                ffc_models_loaded = pickle.load(f)

            FFP25m = ffc_models_loaded['25m']['FFP']
            FFP10m = ffc_models_loaded['10m']['FFP']
            date_list =  ffc_models_loaded['25m']['date'] #date vector is the same for both heights                
            inputdata = ffc_models_loaded['data']
               
            dfcrop_25 = cropfractions(crop_frames, FFP25m, date_list, tower_coords, 30)
            dfcrop_10 = cropfractions(crop_frames, FFP10m, date_list, tower_coords, 30)


            crop25_list.append(dfcrop_25)
            crop10_list.append(dfcrop_10)

            input_list.append(inputdata)


#%% concatenate list of dataframes
dfcrop_all25 = pd.concat(crop25_list)
dfcrop_all10 = pd.concat(crop10_list)
inputs_all = pd.concat(input_list)

#merge into one larger dataframe
df_all = dfcrop_all10.join(dfcrop_all25, how='left', lsuffix='10m', rsuffix='25m')

df_all['Date']=pd.to_datetime(df_all['Date10m'])

df_all = pd.merge(inputs_all,df_all,on='Date')

for i,colname in enumerate(df_all):
    if colname != 'Date':
        df_all[colname] = pd.to_numeric(df_all[colname],errors='coerce')
  
df_all = df_all.drop(['Date10m','Date25m'],axis=1)
    
#save as csv
df_all.to_csv('GC_FootprintFractions.csv')

#%%
stats = df_all.describe()

print(stats[['Area10m','Area25m','f_corn10m','f_corn25m']])


