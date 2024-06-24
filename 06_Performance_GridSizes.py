#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:50:20 2023

analyze performance of JPL and ALEXI ECOSTRESS products vs grid size (1x1, 5x5, 25x25, 50x50)

@author: allison
"""


import pandas as pd
import matplotlib.pyplot as plt


results_path = 'Results_performance'
Maps_path = 'crops_depressions_maps/'
FFP_path = 'GC_FFPs/'
fig_path='Figure_Outputs/'

tower_lon = -88.5773415
tower_lat = 40.1562377
tower_N = 4446215.92
tower_E = 365624.55


df_perf5 = pd.read_csv(results_path+'/Performance_5x5grid.csv')
df_perf25 = pd.read_csv(results_path+'/Performance_25x25grid.csv')
df_perf49 = pd.read_csv(results_path+'/Performance_49x49grid.csv')

df_perf5 = df_perf5.set_index('Unnamed: 0')
df_perf25 = df_perf25.set_index('Unnamed: 0')
df_perf49 = df_perf49.set_index('Unnamed: 0')

metric_opts = ['RMSE', 'alpha','beta']

fig, axs =  plt.subplots(3,1,figsize=(3.25,4))

for i,metric in enumerate(metric_opts):
    RMSE_ALEXI_ratio=[]
    RMSE_JPL_ratio=[]
    RMSE_ALEXI_ratio.append(df_perf5['AlexiET1'][metric]/df_perf5['AlexiFFP25'][metric])
    RMSE_JPL_ratio.append(df_perf5['JdayET1'][metric]/df_perf5['JdayFFP25'][metric])
    
    for df in [df_perf5,df_perf25,df_perf49]:
        RMSE_ALEXI_ratio.append(df['AlexiETg'][metric]/df['AlexiFFP25'][metric])
        RMSE_JPL_ratio.append(df['JdayETg'][metric]/df['JdayFFP25'][metric])

    axs[i].plot(RMSE_JPL_ratio,'-o')
    axs[i].plot(RMSE_ALEXI_ratio,'-*')
    axs[i].hlines(1,0,3,color='k')
    axs[i].set_ylim([.7,1.3])

    
    
    #axs[i].set_title('RMSE(NxN) / RMSE(FFP)',fontsize=10)
    if i ==2:
        axs[i].set_xlabel('NxN grid area')
        axs[i].set_xticks([0,1,2,3],labels=['1x1','5x5','25x25','50x50'])
    else:
        axs[i].set_xticklabels([])
    
    if i ==0:
        axs[i].legend(['PT-JPL','ALEXI'])
    
    
plt.tight_layout()
plt.savefig(fig_path + 'Fig_Error_ratios.svg', format="svg", bbox_inches="tight")  
plt.show()

