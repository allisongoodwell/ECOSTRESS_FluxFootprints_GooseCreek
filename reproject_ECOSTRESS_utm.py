#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:45:09 2023

reproject all ECOSTRESS files to UTM projections

@author: allison

"""

import glob
import rasterio
import utm
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np


Janus=0

if Janus ==1:
    data_path = 'D:\\Users\\Previous_students\\Allison\\Github\\ECOSTRESS_IT\\DATA\\'

else:
    data_path = '/Users/allison/Dropbox/UCDenver/Github/ECOSTRESS_IT/DATA/'


#get projection (UTM) that I want to re-project ECOSTRESS data to...
with rasterio.open(data_path+'/Crops_Depressions_Landcover/depressions_UTM.tif') as src:
    dst_crs = src.crs


usrb_folder = 'ECOSTRESS_USRB/'
folder_names = ['ALEXI_ETdaily','JPL_ETdaily', 'JPL_ETinst','ALEXI_ETdaily_qualityflag','JPL_ETinst_uncertainty']


for folder in folder_names: 
    print(folder)
    file_list = glob.glob(data_path + usrb_folder + folder + "/[!UTM]*.tif")
    omit_list = glob.glob(data_path + usrb_folder + folder + "/*UTM.tif")
    
    file_list = list(set(file_list) - set(omit_list))
    
    for fname in file_list:
        
        fname_utm = fname.replace('.tif', '_UTM.tif',1)
        
            
        with rasterio.open(fname) as src:
            raster = src.read(1)
            height = raster.shape[0]
            width = raster.shape[1]
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            raster_Lon = np.array(xs)[0,:]
            raster_Lat = np.array(ys)[:,0]
            #print(src.crs)
        
            transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
        
            with rasterio.open(fname_utm, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)




