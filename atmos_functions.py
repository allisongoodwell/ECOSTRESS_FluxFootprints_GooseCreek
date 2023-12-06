#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:51:11 2023

atmospheric related functions towards calculations for flux footprint

@author: allison
"""

import numpy as np
import cdsapi
import pandas as pd
import netCDF4 as nc
import os
import pickle


def Obukhov(Ta, p, Hc, rho, ustar): 
    # Ta = air temp [deg C]
    # p = atmpospheric pressure [kPa]
    # Hc = sensible heat flux [W m-2]
    # rho = air density [kg m-3]
    # ustar = friction velocity [m s-1]
    P0 = 100 # reference pressure [kPa]
    Tp = (Ta + 273.15) * (P0/p)**(0.286) # potential temperature [K]
    cp = 1005 # specific heat of air for constant pressure [J K-1 kg-1]
    k = 0.4 # von Karmann
    g = -9.8 # gravitational acceleration [m s-2]
    w_theta = Hc/(rho * cp) # kinematic vertical turbulent sensible heat flux [m K s-1]
    L = (ustar**3 * Tp) / (k * g * w_theta)
    
    return L  


def get_new_blh():    
    
    
    #c = cdsapi.Client()
    c = cdsapi.Client(timeout=600,quiet=False,debug=True)

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'boundary_layer_height',
            'year': [
                '2016','2017','2018','2019','2020',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                40.2, -88.6, 40.19,
                -88.5,
            ],
        },
        'download.nc')
    
    return "log into ERA5 to retrieve new blh"

#function to retrieve blh from most recent file in ERA5 folder
def blh_from_nc(path):
    
    def newest(path):
        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        return max(paths, key=os.path.getctime)


    latest_file = newest(path) #obtain the most recent file in the ERA5 folder (.nc file)
    #print(latest_file)
    
    ds = nc.Dataset(latest_file)
    
    #for dim in ds.dimensions.values():
    #    print(dim)
    #for var in ds.variables.values():
    #    print(var)
        
        #print(ds.variables.keys())
    blh = ds['blh']
    blh_vect = blh[:,0,0,0]
    #plt.plot(time_vect,blh_vect)
    #plt.title('boundary layer height')
    
    time_convert = nc.num2date(ds['time'][:], ds['time'].units,ds['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    
    blh_df = pd.DataFrame(blh_vect,columns=['blh'])
    blh_df['Date'] = pd.to_datetime(time_convert)
    blh_df.set_index('Date')
    
    blh_gaps = blh_df[['Date','blh']].set_index('Date').resample('15min').asfreq()
    blh_filled = blh_gaps.interpolate('time')


    blh_filled['Date'] = blh_filled.index
    blh_dict = {row.Date:row.blh for row in blh_filled.itertuples()}
    
    with open('blh.pickle', 'wb') as f: 
            pickle.dump(blh_dict, f)   
    
    return blh_dict