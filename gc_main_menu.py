#!/usr/bin/env python
# -*- coding: utf-8 -*-
#gc_main_menu.py
#author: Jeff Nivitanont, OU

##/--DATE--/
year = 2007
month = 12
day = 21 #Default is 21 for equinoxes and solstices. Change only if 
         #   appropriate MODIS file available.

##/--REQUIRED FILE POINTERS--/
nc_alb = f'../MODIS/modis_wsa_0.5deg_2007_seasonal.h5'
    #Default file contains only solstices and equinoxes of 2007

##/--OPTIONAL FILE POINTERS--/
directory = None #Set save directory. If None, outputs will be saved in './output/'
scanBlockGeoms = None # This should point to candidate scan blocks (fine, medium, coarse resolution)
timewindow = None  #Supply a pre-calculated timewindow. If None, timewindow is calculated.
af_window = None  #Supply a pre-calculated airmass window. If None, grids are calculated. 
                  #    If af_window is provided without timewindow, will be recalculated.
areaOfInterest = None #'mostpopcities_whem.pkl'
reservedScanBlocks = None #'testreservation.pkl'
cloudProbabilityMap = None #'testprecip.pkl'

##/--COMPUTING OPTIONS--/
enableParallel = True #Recommend turning on. A single 2.5Ghz CPU, ~7-10mins to calculate airmass grid.
forceCalcs = False #force calculation of the airmass grid
trimUniverseSet = True  #Default: True. Scan blocks should roughly cover area of interest.
                        #    This aids in aliasing issues. 

##/--SCANNING OPTIONS--/
scanExtraTime = True #use extra daylight in scan?
scanStartAirmassThreshold = 3.0
scanEndAirmassThreshold = 5.0
scanStartRefCoords = (0, -50) #Macapa, BR (0,-50)
scanEndRefCoords = (19.5, -99.25) #Mexico City, MX (19.5,-99.25)
weightDistPenalty = 1.0
weightOverlapPenalty = 1.0
distanceThreshold = 8  #How many 5-min scan blocks in the E-W direction
                       #    to consider per algorithm step?
universeCoverageTol = 0.005 #allowable limit of uncovered area

##/--PLOTTING OPTIONS--/
plotCoverset = True
createMov = True #create a movie of the scan?
compareToBaseline = True #plot diagnostic histograms of xco2_uncert, signal-to-noise ratio,
                         #    solar zenith angle, and satellite zenith angle.