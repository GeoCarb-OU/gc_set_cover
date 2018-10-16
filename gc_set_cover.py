#gc_set_cover.py // Functions used for set cover problem applied to GeoCARB mission
#updated 2018/04/24
#@author: Jeff Nivitanont, University of Oklahoma

import numpy
import datetime as dt
import time
import calendar
from numba import jit
from shapely.ops import transform
from functools import partial
import pyproj
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from pylab import *
#import winsound
import pandas as pd
import geopandas as gpd
from shapely.validation import explain_validity

geo_lon = -85.    #satellite lon
geo_lat = 0.      #satellite lat
geo_ht = 42.336e6 #fixed satellite height
surf_ht = 6.370e6 #uniform surface height
sol_ht = 149.6e9 #fixed solar height
r_earth = 6.371e6 # radius of Earth
north_solst = 354 # date of southern summer solstice
declin = 23.44
earth_sun = 149.6e9 #distance to sun
atrain_alt = 705e3 
dtor = pi/180.   #convert degs to rads

#mesh grid coords lat/lon
x = linspace(-130, -30, 201)
y = linspace(50, -50, 201)
xv, yv = meshgrid(x, y)

#for converting lat/lon to Geo
geo = ccrs.Geostationary(central_longitude=-85.0, satellite_height=42.336e6)
geo_proj4 = geo.proj4_init
latlon = {'init' :'epsg:4326'}

def latlon_to_geo(geom):
    '''
    This function takes a Shapely Geometry projected in lat/lon and returns a Shapely Geometry projected in Geostationary.
    
    NOTE: geom.crs should be initialized before using this function.
    
    Params:
    - geom: Shapely Geometry projected in lat/lon.
    
    Return: A Shapely Geometry projected in Geostationary.
    
    '''
    #partially applied function from pyproj for converting from lat/lon to geo
    project = partial( pyproj.transform, pyproj.Proj(latlon), pyproj.Proj(geo_proj4) )
    return transform(project, geom)

def geo_to_latlon(geom):
    '''
    This function takes a Shapely Geometry projected in Geostationary and returns a Shapely Geometry projected in lat/lon.
    
    NOTE: geom.crs should be initialized before using this function.
    
    Params:
    - geom: Shapely Geometry projected in Geostationary.
    
    Return: A Shapely Geometry projected in lat/lon.
    
    '''
    #partially applied function from pyproj for converting from lat/lon to geo
    project2 = partial( pyproj.transform, pyproj.Proj(geo_proj4), pyproj.Proj(latlon))
    return transform(project2, geom)


@vectorize #this is a NumPy decorator
def airmass(lat, lon, time):
    '''
    This function gives the airmass at a lat,lon point of time.
    
    Params:
    - lat: latitude in degrees.
    - lon: longitude in degrees.
    - time: time, DateTime object.
    
    Return:
    - airmass factor, float.
    '''
    start_day = dt.datetime(time.year,1,1,0,0,0)
    view_lat = lat #viewer on ground
    view_lon = lon
    day_dec = (time-start_day).total_seconds()/86400.  #decimal julian date for scalar_subsol func
    subsol_lat,subsol_lon = scalar_subsol(day_dec) #sub-solar lat/lon
    
    sol = array([array([sol_ht,subsol_lat,subsol_lon])])
    ex = array([array([surf_ht,view_lat,view_lon])])
    sol_zenith = zenith_angle_cosine_batch(ex,sol) #(viewer, target)

    sat = array([array([geo_ht,0.,geo_lon])])
    sat_zenith = zenith_angle_cosine_batch(ex,sat)
    
    if( sol_zenith >= 90. or sat_zenith >= 90.):
        return 9999.
    else:
        return 1./cos(dtor*sol_zenith) + 1./cos(dtor*sat_zenith)


        
#from earth_calcs.py
@jit
def scalar_earth_angle( lat1, lon1, lat2, lon2):
    """ angle in degrees on great circle between two points """
    theta1 = lat1 *dtor
    phi1 = lon1 *dtor
    theta2 = lat2 * dtor
    phi2 = lon2 * dtor
    p1 = numpy.vstack((cos(theta1)*cos(phi1),cos(theta1)*sin(phi1),sin( theta1))).T
    p2 = numpy.vstack((cos(theta2)*cos(phi2), cos( theta2)* sin( phi2), sin( theta2))).T
    dsq = ((p1-p2)**2).sum(-1)
    return numpy.arccos((2 -dsq)/2.)/dtor

#from earth_calcs.py
@jit
def scalar_subsol( day):
    "subsolar lat-lon given decimal day-of-year "
    lat = -declin * numpy.cos(2*pi* numpy.mod(365 + day - north_solst,  365.)/365.)
    lon = numpy.mod( 180. -360.*(day -numpy.floor(day)), 360.)
    return lat, lon

#from earth_calcs.py
@jit
def zenith_angle_cosine_batch( viewer,target):
    """ gives the zenith angle of a target  (r theta, phi) from the viewer (r, theta phi), theta, phi and result are in degrees"""
    centre_angle = scalar_earth_angle( viewer[:,1],viewer[:,2], target[:,1], target[:,2]) # angle between the two locations at centre of earth
    dist = (viewer[:,0]**2 + target[:,0]**2 -2.*target[:,0]*viewer[:,0]*numpy.cos( centre_angle*dtor))**0.5 # cosine rule
    cos_zenith = -0.5*(dist**2+viewer[:,0]**2-target[:,0]**2)/(dist*viewer[:,0])  # the minus makes it a zenith angle
    return numpy.arccos(cos_zenith)/dtor

@jit #using jit and numpy.vectorized airmass func
def calc_afmesh_5min(month, day, hour, minute, length):
    '''calculates an array of lat, lon points in a time interval and returns a 3-D array, [time, lat, lon], 16-hour span'''
    time_intv = list([dt.datetime(2018, month, day, hour, minute)])
    for i in arange(1,length):
        time_intv.append(time_intv[i-1]+dt.timedelta(0,300))
    day_mesh = zeros([len(time_intv), len(yv), len(xv) ])
    for i in range(len(time_intv)):
        day_mesh[i,:,:] = airmass(lat=yv, lon=xv, time = time_intv[i])
    day_mesh[day_mesh < 2.0] = 20.
    return day_mesh

@jit #using jit and numpy.vectorized airmass func
def calc_afmesh_window(year, month, day, start_th = 3., end_th = 4.):
    '''
    Calculates optimal start and finish time for a scan. Using Macapa, Brazil and Mexico City, Mexico as points of reference.
    
    Params:
    - year: Year (int).
    - month: Month (int).
    - day: Day (int).
    - start_th: Specified Airmass Factor threshold with which to start the scan.
    - end_th: Specified Airmass Factor threshold with which to end the scan.
    
    Return:
    - day_mesh: Numpy array of lat, lon points - [time, lat, lon].
    - time_intv: list of datetimes for scan.
    '''

    time_intv = list([dt.datetime(year, month, day, 9)])
    for i in arange(1,288):
        time_intv.append(time_intv[i-1]+dt.timedelta(0,300))
    startpt_af = airmass(lat = 0, lon = -50, time = time_intv)
    endpt_af = airmass(lat = 19.5, lon = -99.25, time = time_intv)
    start_ind = 0
    while( startpt_af[start_ind] < start_th):
        start_ind += 1
    end_ind = 287
    while( endpt_af[end_ind] > end_th):
        end_ind -= 1
    window_len = end_ind - start_ind
    day_mesh = full([window_len, 201, 201 ], 20.)
    for i in range(window_len):
        day_mesh[i,:,:] = airmass(lat=yv, lon=xv, time = time_intv[i + start_ind])
    day_mesh[day_mesh < 2.0] = 9999.
    return day_mesh, time_intv[start_ind:end_ind]


def calc_set_err(coverset, albedo, mesh_pts):
    '''
    This function applies calc_xco2_err to each block and returns a Numpy array.
    
    Params:
    - coverset: a covering set from the Greedy Algorithm, GeoPandas.GeoDataFrame.
    - albedo: 360x720 albedo map, MODIS product wsa-band6/7.
    - mesh_pts: lat/lon mesh grid.
    
    Return:
    - error_df: a Pandas.DataFrame
    '''
    mesh = mesh_pts
    error_df = calc_xco2_err(albedo = albedo, scan_block = coverset.iloc[0], mesh_pts = mesh)
    for i in np.arange(1,len(coverset)):
        block_df = calc_xco2_err(albedo = albedo, scan_block = coverset.iloc[i], mesh_pts = mesh)
        try:
            error_df = error_df.append(block_df, ignore_index = True)
        except ValueError:
            pass
    return error_df

@jit
def calc_xco2_err(albedo, scan_block, mesh_pts, aod=0.3,threshold=0):
    '''
    
    This function takes in a blockset and calculates sucess based on indicated threshold for Signal-Noise Ratio (SNR).
    
    Params:
    - albedo: 360x720 albedo map, MODIS product wsa-band6/7.
    - scan_block: Pandas.DataFrame slice, Pandas.Series
    - mesh_pts: lat/lon mesh grid.
    - aod: Aerosol Optical Depth, default = 0.3.
    - threshold: SNR threshold, default = 0.
    
    Return:
    - array of lat/lon grid points where SNR passes threshold. [lon, lat, error, snr]
    
    '''
    block_geom = scan_block['geometry']
    time = scan_block['time']
    fsun = 2073 #nW cm sr^(-1)  cm^(-2)
    n0sqd = (0.1296)**2
    n1 = 0.00175
    start_day = dt.datetime(time.year,1,1,0,0,0)
    af_intx = block_geom.intersection(mesh_pts)
    af_intx = geo_to_latlon(af_intx)
    intxpts = np.array(af_intx) #[x,y]
    if(len(intxpts.shape) != 2):                    #shapely intersection issue 
            return None
    pts_arr = [((intxpts[:,1]+90)*2.).round().astype(int), ((intxpts[:,0]+180)*2.).round().astype(int)] #lat(row), lon(col)
    albedo_arr = albedo[pts_arr]
    #albedo_arr = np.nan_to_num(albedo_arr)

    view_lat = intxpts[:,1].round(1) #viewer on ground
    view_lon = intxpts[:,0].round(1)
    day_dec = (time-start_day).total_seconds()/86400.  #decimal julian date for scalar_subsol func
    subsol_lat,subsol_lon = scalar_subsol(day_dec) #sub-solar lat/lon

    sol = np.array([[sol_ht, subsol_lat,subsol_lon]])
    ex = np.array([repeat(surf_ht, len(view_lat)), view_lat, view_lon]).T
    sol_zenith = zenith_angle_cosine_batch(ex, sol) #(viewer, target)

    sat = np.array([[geo_ht, geo_lat ,geo_lon]])
    sat_zenith = zenith_angle_cosine_batch( ex, sat)
    af = 1./cos(sol_zenith*dtor) + 1./cos(sat_zenith*dtor)
    af[sat_zenith >= 90.] = 9999.
    af[sol_zenith >= 90.] = 9999.
    
    snr = zeros(sol_zenith.size)
    snr_th = zeros(sol_zenith.size)
    #Signal
    S = fsun*albedo_arr*cos(sol_zenith*dtor)*exp(-af*aod)

    #Noise
    N = sqrt(n1*S+n0sqd)
    snr[S>0] = S[S>0]/N[S>0]
    #xco2 uncert
    sig_xco2 = maximum(1./(1./14.+0.0039*snr),threshold*ones(snr.shape))

    #n0_thsqd = 0.05900041 #(0.2429)**2
    #n1_th = 0.002828
    #N_th = sqrt(n1_th*S+n0_thsqd)
    #snr_th[S>0] = S[S>0]/N_th[S>0]
    #sig_xco2_th = maximum(1./(1./14.+0.0039*snr_th),threshold*ones(snr_th.shape))
    block_df = pd.DataFrame({'block_id' : np.repeat(scan_block['block_id'], len(snr)), 'time' : np.repeat(time, len(snr)), 'lat' : view_lat, 'lon' : view_lon, 'error': sig_xco2, 'SNR' : snr})

    return block_df
    
@jit
def create_mask_err(error_df):
    '''
    This function creates an array of minimum XCO2 error. Areas not intersecting land are masked with np.nan for plotting.
    
    Params:
    - error_df: error Pandas.DataFrame output from `calc_set_error`.
    
    Return:
    - E: masked array of minimum errors.
    '''
    sorted_df = error_df.iloc[error_df['error'].argsort()].copy() #sorted in ascending order
    e_unqs = sorted_df.drop_duplicates(subset=['lat', 'lon']) #choose minimum error
    E = np.full(xv.shape, np.nan )
    e_pts_arr = [((50-e_unqs['lat'])*2.).round().astype(int), ((e_unqs['lon']+130)*2.).round().astype(int)]
                #lat(row), lon(col)
    E[e_pts_arr] = e_unqs['error'].copy()
    return E


@jit
def calc_block_cost(block, universe, mesh_pts, mesh_airmass, last, covered, albedo, weights = [1.,1.]):
    '''
    Cost function for Greedy Algorithm
    
    (default): cost = avg(exp(af)/albedo)*(1. + (overlap + dist)/block.intersection(universe).area)
    
    Params:
    - block: candidate scan block.
    - universe: Shapely Geometry to be covered by block.
    - mesh_pts: mesh coordinate points; Shapely.MultiPoint object.
    - mesh_airmass: Airmass Factor scores at mesh coordinate points.
    - last: last selected block.
    - covered: union of selected blocks.
    - albedo: 360x720 albedo map, MODIS product wsa-band6/7.
    - weights: an iterable object of weights, i.e. tuples, lists, arrays, etc. Must contain 2 items. (w_dist, w_overlap). Default is [1., 1.]
    '''
    if(block.intersects(mesh_pts)==False or block.intersection(universe).area < 1.):# or block.intersection(last).area/block.area > .05):#if block doesn't have mesh points or if block intersects last block
        return float(inf)
    else:
        af_intx = block.intersection(mesh_pts)
        af_intx = geo_to_latlon(af_intx)
        intxpts = array(af_intx)
        if(len(intxpts.shape)!=2):                    #shapely intersection issue 
            return float(inf)
        else:
            pts_arr = [((50-intxpts[:,1])*2.).round().astype(int), ((intxpts[:,0]+130)*2.).round().astype(int)]
            af_arr = mesh_airmass[pts_arr]
            #avg_af = af_arr.mean()
            alb_pts_arr = [((intxpts[:,1]+90)*2.).round().astype(int), ((intxpts[:,0]+180)*2.).round().astype(int)]
            alb_arr = albedo[alb_pts_arr]
            #avg_alb = alb_arr.mean()
            avg_exp = np.nanmedian(np.exp(af_arr)/alb_arr)
            overlap = block.intersection(covered).area
            if(overlap < block.area*.04):
                overlap = 0.
            dist = block.distance(last)
            cost = avg_exp*(1. + (weights[0]*(dist**2) + weights[1]*overlap )/block.intersection(universe).area)
            #lower bound = 0
            return cost
                    
                    
##Greedy Algorithm for Set Cover
@jit
def greedy_gc_cost(blockset,
                   universe_set,
                   albedo,
                   mesh_pts,
                   mesh_airmass,
                   t = 0,
                   tol = .005,
                   setmax = 144,
                   weights = [1., 1.] ):
    '''
    This Greedy Algorithm selects scan blocks by the cost function, which utilizes the mesh inputs.
    block.
    
    Params:
    - blockset: A set of candidate scan blocks in GeoPandas.GeoDataFrame format.
    - universe_set: A Shapely Geometry object for the area required to be covered.
    - albedo: 360x720 albedo map, MODIS product wsa-band6/7.
    - mesh_pts: A Shapely.MultiPoint object that contains the mesh pts in lat/lon to be passed to the cost function.
    - mesh_airmass: The Airmass Factor scores used in the cost function calculated at 5-min intervals during the day.
    - t: time to start algorithm (t*5-minutes) from beginning of time window.
    - tol: Tolerance for uncovered land.
    - setmax: max covering set size.
    - weights: an iterable object of weights, i.e. tuples, lists, arrays, etc. Must contain 2 items. (w_dist, w_overlap). Default is [1., 1.]

    Return:
    - cover_set: covering set for universe.
    '''
    
    if(all(blockset.is_valid)==False):
        print('Error: invalid geometries in blocks.')
        return
    universe = universe_set
    mesh_unv = mesh_pts.intersection(universe)
    scan_blocks = blockset[blockset.intersects(universe)].reset_index(drop=True)
    init_area = universe.area
    if(universe_set.difference(blockset.unary_union).area > tol*init_area):
        print('Error: Blocks do not cover universe.')
        return
    cover_set = gpd.GeoDataFrame()
    cover_set.crs = geo_proj4
    covered = sgeom.Point()
    lastgeom = sgeom.Point()
    block_cost = []
    sel_ind = None
    ii=0 #flow control index, prevents infinite looping
    #print('Greedy Algorithm commenced' )
    while( (universe.area > init_area*tol)  and (ii < setmax) ):
        #print(ii,explain_validity(universe_set),explain_validity(covered)) #checks validity of geoms for debugging
        if(universe.area > init_area*.05):   #if majority of area is uncovered
            #partially apply args to calc_block_cost function
            cost_mesh_f = partial(calc_block_cost,
                                  universe = universe,
                                  albedo = albedo,
                                  mesh_pts = mesh_unv,
                                  mesh_airmass = mesh_airmass[t, :, :],
                                  last=lastgeom,
                                  covered = covered,
                                  weights = weights)
            dist_idx = np.array(scan_blocks.centroid.distance(lastgeom.centroid)< 1200000.) #boolean array
            block_cost = np.full(len(scan_blocks), float(inf))
            block_cost[dist_idx] = list(map( cost_mesh_f, scan_blocks[dist_idx]['geometry'])) #list object
            sel_ind = np.argmin(block_cost)
        else:
#         elif(universe.area > init_area*.03):   #if there is more area than a normal scan block size
            #partially apply args to calc_block_cost function
            cost_mesh_f = partial(calc_block_cost,
                                  universe = universe,
                                  albedo = albedo,
                                  mesh_pts = mesh_unv,
                                  mesh_airmass = mesh_airmass[t, :, :],
                                  last=lastgeom,
                                  covered = covered,
                                  weights = weights)
            dist_idx = np.array(scan_blocks.intersects(universe)) #boolean array
            block_cost = np.full(len(scan_blocks), float(inf))
            block_cost[dist_idx] = list(map( cost_mesh_f, scan_blocks[dist_idx]['geometry'])) #list object
            sel_ind = np.argmin(block_cost)
#         else: #find coverage uniform weights
#             block_cost = scan_blocks.intersection(universe).area #Pandas.Series
#             sel_ind = block_cost.idxmax()
        cover_set = cover_set.append(scan_blocks.iloc[sel_ind], ignore_index = True)
        covered = cover_set.unary_union.buffer(0)
        universe = universe.difference(covered).buffer(0)
        mesh_unv = mesh_unv.intersection(universe)
        lastgeom = scan_blocks.geometry[sel_ind]
        scan_blocks = scan_blocks.drop(sel_ind).reset_index(drop=True)
        ii+=1
        t+=1
        #print (t, end=' ')
    #print('\nFinished.',len(cover_set),'scanning blocks chosen.')
    #print('Uncovered area  = '+str( round(100*universe.area/init_area, 2) )+'%')
    return cover_set


@jit
def greedy_gc_unif(blockset, universe_set, tol= 0.0):
    '''
    This Greedy Algorithm gives uniform weight to all areas of land. Most basic weight function.
    
    Params:
    - blockset: A set of candidate scan blocks in GeoPandas.GeoDataFrame format.
    - universe_set: A Shapely Geometry object for the area required to be covered.
    - tol: allowable remaining uncovered area.
    '''
    if(universe_set.difference(blockset.unary_union).area > 0.0):
        print('Error: Blocks do not cover universe.')
        return
    if(all(blockset.is_valid)==False):
        print('Error: invalid geometries in blocks.')
        return
    scan_blocks = blockset
    universe = universe_set
    init_area = universe.area
    cover_set = gpd.GeoDataFrame()
    cover_set.crs = geo_proj4
    covered = sgeom.Point()
    lastgeom = sgeom.Point()
    block_weight = []
    max_ind = []
    ii=0 #flow control index, prevents infinite looping
    #t=0 #time from start in hours
    #print('Greedy Algorithm commenced')
    while( (universe.area > init_area*tol)  and (ii<168) ):
        #if(ii%12 ==0):
        #    t+=1
        #partially apply args to calc_block_weight function
        dist = scan_blocks.centroid.distance(lastgeom.centroid)
        block_weight = scan_blocks.intersection(universe).area - dist
        max_ind = block_weight.idxmax()
        lastgeom = scan_blocks.geometry[max_ind]
        cover_set = cover_set.append(scan_blocks.iloc[max_ind])
        covered = cover_set.unary_union
        #print(ii,explain_validity(universe_set),explain_validity(covered)) #checks validity of geoms for debugging
        universe = universe.difference(covered).buffer(0)
        scan_blocks = scan_blocks.drop(max_ind).reset_index(drop=True)
        ii+=1
        #print (ii, end=' ')
    #print('Finished.',len(cover_set),'scanning blocks chosen.')
    #print('Uncovered Area = '+str(round(100*universe.area/init_area,2))+'%')
    cover_set = cover_set.reset_index(drop=True)
    return cover_set