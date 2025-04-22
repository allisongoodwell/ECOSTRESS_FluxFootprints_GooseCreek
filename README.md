# ECOSTRESS_FluxFootprints_GooseCreek
 repository associated with "Spatial heterogeneity of agricultural evapotranspiration as quantified by satellite and flux tower sources" by Goodwell, Zahan, , Cao, URycki
(under revision for Agricultural and Forest Meteorology, April 2025)

Python codes needed to produce plots in paper.

Flux tower data (25m and 10m heights) and flux footprint model outputs can be found on HydroShare at:  https://www.hydroshare.org/resource/0ef3eda3534f44a6bbd65786d57222ea/

From Hydroshare repository above, download and unzip the folders "GC_FFPs" (pre-made flux footprint contours in daily pickle files, which can also generated from O1_RunFFP_bothheights.py) and ECOSTRESS_URSB (satellite ET images and uncertainty products).  In this Github repository, I include several example files of these datasets but not enough to reproduce all results.


Python codes and descriptions:

00_prepare_GCtowerdata.py: pre-processing of tower data for input into FFP (Kljun 2015) model.  Outputs GCFluxTower_30min_2heights.csv data file (located in DATA_Tower folder).  
Most inputs for this pre-processing are located in the DATA_Tower folder, except FluxData_Raw_ALL.csv which is a large file, available in the Hydroshare repository listed above (file is named GC_FluxData_RAW_25m_042216_050224.csv) 

01_RunFFP_both_heights.py: Running the FFP model for 30 minute time steps for both tower heights.  Results are saved as dictionaries in .pickle files.  See Hydroshare repository for these files that have been pre-made (it can take a while to run this code).  

02_Cropfractions_both_heights.py: Uses the pickle FFP files and USDA Crop Data Layer (CDL) maps (crop_maps folder) to compute fractions of corn and soybean for each footprint.  Outputs file GC_FootprintFractions.csv (in DATA_Tower folder).


03_LoadECOSTRESS_PlotFFPs.py: Plots FFP contours and ECOSTRESS images, box plots and diurnal plots for individual overpasses.


03b_LoadECOSTRESS_PlotDiurnalJPL: Similar code, but produces combined figure showing errors for many overpasses.  Outputs file "ECOSTRESS_footprintETvalues.csv) (in Results_performance folder)


04_AnalyzeECOSTRESS_FFPs: Outputs figures showing model performance, and file Performance_gridsizes.csv (in Results_performance folder)


05_Analyze_FFPs_TowerCropContributions.py: Does monthly average estimate of corn and soybean ET based on tower data.  Produces file CropSpecificMonthlyTowerET.csv (in DATA_Tower folder)





