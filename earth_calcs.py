import numpy
from numba import jit
import pdb
from numpy import pi, sin, cos

r_earth = 6.371e6 # radius of Earth
north_solst = 354 # date of southern summer solstice
declin = 23.44
earth_sun = 149.6e9
atrain_alt = 705e3 
dtor = pi/180. # conversion from degrees to radians

def scalar_earth_area( minlat, maxlat, minlon, maxlon):
    "area of earth in the defined box"
    diff_lon = numpy.unwrap((minlon,maxlon), discont=360.+1e-6)
    return 2.*pi*r_earth**2 *(numpy.sin( dtor*maxlat) -numpy.sin(dtor*minlat))*(diff_lon[1]-diff_lon[0])/360.

def earth_area( minlat, maxlat, minlon, maxlon):
    "returns grid of areas with shape=(minlon, minlat) and earth area"
    result = numpy.zeros((numpy.array(minlon).size, numpy.array(minlat).size))
    diffsinlat = 2.*pi*r_earth**2 *(numpy.sin( dtor*maxlat) -numpy.sin(dtor*minlat))
    diff_lon = numpy.unwrap((minlon,maxlon), discont=360.+1e-6)
    for i in xrange(numpy.array(minlon).size): result[ i, :] = diffsinlat *(diff_lon[1,i]-diff_lon[0,i])/360.
    return result

def earth_area_grid(lats,lons):
    """grids earth area hopefully"""
    result=numpy.zeros( (lats.size, lons.size))
    minlats = lats -0.5*(lats[1]-lats[0])
    maxlats= lats +0.5*(lats[1]-lats[0])
    minlons = lons -0.5*(lons[1]-lons[0])
    maxlons=lons +0.5*(lons[1]-lons[0])
    for i in xrange(lats.size):
        result[i,:] = scalar_earth_area(minlats[i],maxlats[i],minlons[0],maxlons[0])
    return result

def scalar_earth_angle_old( lat1, lon1, lat2, lon2):
    """ finds the angle at the centre of the earth between two points """
    d_lon = numpy.unwrap((lon1,lon2),360.)*dtor
    cordsq = 2*(numpy.cos(dtor*lat1))**2 *(1. -numpy.cos( d_lon[1] -d_lon[0])) # direct distance from (lat1,dlon) to (lat1,0) by cos rule
    cos_cord = (2. -cordsq)/2. # cos at earth centre subtended by (lat1, 0)-(lat1,d_lon) by cos rule
    # that's also the distance
    # along the line from CE to
    # (lat1,0) where the perp drops, call
    # it P
    cord2sq = 1. + cos_cord**2. - 2.*cos_cord * numpy.cos( dtor*(lat1 - lat2)) # should be distance in plane from P to (lat2,0)
    distsq = cord2sq + 1 - cos_cord**2
    #return numpy.arccos((2. -distsq)/2.)/dtor
    return numpy.arccos(numpy.sin(lat1*dtor)*numpy.sin(lat2*dtor)+numpy.cos(lat1*dtor)*numpy.cos(lat2*dtor)*numpy.cos(d_lon[1]-d_lon[0]))/dtor

@jit
def scalar_earth_angle( lat1, lon1, lat2, lon2):
    """ angle on great circle between two points """
    theta1 = lat1 *dtor
    phi1 = lon1 *dtor
    theta2 = lat2 * dtor
    phi2 = lon2 * dtor
    p1 = numpy.vstack((cos(theta1)*cos(phi1),cos(theta1)*sin(phi1),sin( theta1))).T#numpy.array([ cos( theta1)*cos(phi1), cos( theta1)* sin( phi1), sin( theta1)])
    p2 = numpy.vstack((cos(theta2)*cos(phi2), cos( theta2)* sin( phi2), sin( theta2))).T
    dsq = ((p1-p2)**2).sum(-1)
    return numpy.arccos((2 -dsq)/2.)/dtor

earth_angle = numpy.vectorize( scalar_earth_angle)

def scalar_earth_distance( lat1, lon1, lat2, lon2):
    """ finds the distance on the surface of a spherical earth between two points"""
    return dtor* r_earth*earth_angle( lat1, lon1, lat2, lon2)

earth_distance = numpy.vectorize( scalar_earth_distance)

@jit
def scalar_subsol( day):
    "subsolar lat-lon given decimal day-of-year "
    lat = -declin * numpy.cos(2*pi* numpy.mod(365 + day - north_solst,  365.)/365.)
    lon = numpy.mod( 180. -360.*(day -numpy.floor(day)), 360.)
    return lat, lon

subsol = numpy.vectorize( scalar_subsol)

def sphere_rotate( axis, angle, coord):
    "return coordinate obtained by rotating coord through angle about axis (defined by lat,lon of edge"
 # obtained from http://www.uwgb.edu/dutchs/mathalgo/sphere0.htm   
    c1 = numpy.cos( axis[0]*pi/180.) * numpy.cos( axis[1]*pi/180.)
    c2 = numpy.cos( axis[0]*pi/180.) * numpy.sin( axis[1]*pi/180.)
    c3 = numpy.sin( axis[0]*pi/180.)
    x = numpy.cos( coord[0]*pi/180.) * numpy.cos( coord[1]*pi/180.)
    y = numpy.cos( coord[0]*pi/180.) * numpy.sin( coord[1]*pi/180.)
    z = numpy.sin( coord[0]*pi/180.)

    cosa = numpy.cos( angle*pi/180.)
    sina = numpy.sin( angle*pi/180.)
    xx = x *cosa + (1 - cosa)*(c1*c1*x + c1*c2*y + c1*c3*z) + (c2*z - c3*y)*sina
    yy = y *cosa + (1 - cosa)*(c2*c1*x + c2*c2*y + c2*c3*z) + (c3*x - c1*z)*sina
    zz = z *cosa + (1 - cosa)*(c3*c1*x + c3*c2*y + c3*c3*z) + (c1*y - c2*x)*sina
#    print 'checking rotation',xx**2+yy**2+zz**2
# back to lat and lon
    lat = numpy.arcsin(zz)*180./pi
    lon = numpy.arctan2(yy,xx)*180./pi
    return lat,lon

def greatcircle_axis(p1, p2):
    " finds axis of great circle defined by p1, p2"
    x1 = numpy.cos( p1[0]*dtor) * numpy.cos( p1[1]*dtor)
    y1 = numpy.cos( p1[0]*dtor) * numpy.sin( p1[1]*dtor)
    z1 = numpy.sin( p1[0]*dtor)
    x2 = numpy.cos( p2[0]*dtor) * numpy.cos( p2[1]*dtor)
    y2 = numpy.cos( p2[0]*dtor) * numpy.sin( p2[1]*dtor)
    z2 = numpy.sin( p2[0]*dtor)
# now use cross-product to find vector perpendicular to both
    x= y1*z2 - y2*z1
    y = -(z2*x1 -x2*z1)
    z = x1*y2 -x2*y1
#    print 'checking dot-products',(x*x1 +y*y1 +z*z1),(x*x2+y*y2+z*z2)
    lat = numpy.arcsin(z/numpy.sqrt(x**2 +y**2 +z**2 ))/dtor
    lon = numpy.arctan2(y,x)/dtor
    return lat,lon

# helper function for sine of angle in triangle 
def sinang( a, b, theta): return numpy.sin(theta) *b/numpy.sqrt(a**2 +b**2 -2.*a*b*numpy.cos( theta))

def scalar_glint( sat, sun):
    " finds glint lat,  lon and sza for two points, if sza > pi/2 it's not possible"
# working as follows: triangle in principal plane, 
# angle at centre is alpha, consider a rotation theta from the subsat point
# we need the angle theta such that
# ray from sun-surface makes same angle with centre-surface as ray sat-surface
# so 2 triangles, sat-centre-glint sun-centre-glint     
# for satellite is sin(x)/(r_earth+atrain_alt) = sin(theta)/sqrt((r_earth+atrain_alt)^2 +r_earth^2 -2r_erath*(r_earth +atrain_alt)*cos(theta)
# same expression for Sun, now make them equal
    delta = earth_angle(sat[0],sat[1], sun[0], sun[1]) *pi/180.

    if ( delta > pi/2. and numpy.cos( delta) < r_earth/(r_earth +atrain_alt)): return 0.,0.,180. # in shadow, no glint
# now iterative solution
    theta = delta/ 2.
    inc = delta /4.
    for i in range(20):
#        import pdb; pdb.set_trace()
        diff = delta - theta
        #expr = sinang( r_earth+atrain_alt, r_earth, theta) -sinang( r_earth, r_earth+earth_sun, diff)
        expr = sinang( r_earth, r_earth+atrain_alt, theta) -sinang( r_earth, r_earth+earth_sun, diff)
        if abs(expr) < 1e-3: break
        theta -= inc * numpy.sign(expr) 

        inc /= 2.
    # theta is the angle from the subsat point to the glint spot in the principal plane
    # to find that point perform rotation in principal plane
    rot_axis = greatcircle_axis( sat, sun)
    result =  sphere_rotate( rot_axis, theta*180./pi, sat)
# rotation must be towards the sun, check this and correct sign if necesary
    if earth_distance( sun[0], sun[1], result[0], result[1]) > earth_distance( sun[0], sun[1], sat[0], sat[1]):\
            result = sphere_rotate( rot_axis, -theta*180./pi, sat)
    #print theta*180./pi
    sza = numpy.arcsin( sinang( r_earth, r_earth+atrain_alt, theta))*180./pi
    return result[0],result[1],sza

def scalar_half_glint( sat, sun):
    " finds glint lat,  lon and sza for two points, if sza > pi/2 it's not possible"
# working as follows: triangle in principal plane, 
# angle at centre is alpha, consider a rotation theta from the subsat point
# we need the angle theta such that
# ray from sun-surface makes same angle with centre-surface as ray sat-surface
# so 2 triangles, sat-centre-glint sun-centre-glint     
# for satellite is sin(x)/(r_earth+atrain_alt) = sin(theta)/sqrt((r_earth+atrain_alt)^2 +r_earth^2 -2r_erath*(r_earth +atrain_alt)*cos(theta)
# same expression for Sun, now make them equal
    delta = earth_angle(sat[0],sat[1], sun[0], sun[1]) *pi/180.

    if ( delta > pi/2. and numpy.cos( delta) < r_earth/(r_earth +atrain_alt)): return 0.,0.,180. # in shadow, no glint
# now iterative solution
    theta = delta/ 2.
    inc = delta /4.
    for i in range(50):
#        import pdb; pdb.set_trace()
        diff = delta - theta
        #expr = sinang( r_earth+atrain_alt, r_earth, theta) -sinang( r_earth, r_earth+earth_sun, diff)
        expr = sinang( r_earth, r_earth+atrain_alt, theta) -sinang( r_earth, r_earth+earth_sun, diff)
        if abs(expr) < 1e-3: break
        theta -= inc * numpy.sign(expr) 

        inc /= 2.
    # theta is the angle from the subsat point to the glint spot in the principal plane
    # to find that point perform rotation in principal plane
    theta /= 2 #halfway to glint, to avoid saturation of detectors

    rot_axis = greatcircle_axis( sat, sun)
    result =  sphere_rotate( rot_axis, theta*180./pi, sat)
# rotation must be towards the sun, check this and correct sign if necesary
    if earth_distance( sun[0], sun[1], result[0], result[1]) > earth_distance( sun[0], sun[1], sat[0], sat[1]):\
            result = sphere_rotate( rot_axis, -theta*180./pi, sat)
    #print theta*180./pi
    sza = numpy.arcsin( sinang( r_earth, r_earth+atrain_alt, theta))*180./pi
    return result[0],result[1],sza

def zenith_angle( viewer, target):
    """ gives the zenith angle of a target  (r theta, phi) from the viewer (r, theta phi), theta, phi and result are in degrees"""
    centre_angle = scalar_earth_angle( viewer[1],viewer[2], target[1], target[2]) # angle between the two locations at centre of earth
    dist = (viewer[0]**2 + target[0]**2 -2.*target[0]*viewer[0]*numpy.cos( centre_angle*dtor))**0.5 # cosine rule
    sin_zenith = target[0]*numpy.sin( centre_angle*dtor)/dist
    return numpy.arcsin( sin_zenith)/dtor

def zenith_angle_cosine( viewer,target):
    """ gives the zenith angle of a target  (r theta, phi) from the viewer (r, theta phi), theta, phi and result are in degrees"""
    #pdb.set_trace()
    centre_angle = scalar_earth_angle( viewer[1],viewer[2], target[1], target[2]) # angle between the two locations at centre of earth
    dist = (viewer[0]**2 + target[0]**2 -2.*target[0]*viewer[0]*numpy.cos( centre_angle*dtor))**0.5 # cosine rule
    cos_zenith = -0.5*(dist**2+viewer[0]**2-target[0]**2)/(dist*viewer[0])
    return numpy.arccos(cos_zenith)/dtor

@jit
def zenith_angle_cosine_batch( viewer,target):
    """ gives the zenith angle of a target  (r theta, phi) from the viewer (r, theta phi), theta, phi and result are in degrees"""
    #pdb.set_trace()
    centre_angle = scalar_earth_angle( viewer[:,1],viewer[:,2], target[:,1], target[:,2]) # angle between the two locations at centre of earth
    dist = (viewer[:,0]**2 + target[:,0]**2 -2.*target[:,0]*viewer[:,0]*numpy.cos( centre_angle*dtor))**0.5 # cosine rule
    cos_zenith = -0.5*(dist**2+viewer[:,0]**2-target[:,0]**2)/(dist*viewer[:,0])  # the minus makes it a zenith angle
    return numpy.arccos(cos_zenith)/dtor

def bearing( viewer, target):
    """ gives bearing at earth's surface from viewer to target, zero is due north """
    # first normalize to earth's surface
    latlon_viewer = viewer[1:]
    latlon_target = target[1:]
    perp = greatcircle_axis( latlon_viewer, latlon_target)
    lat,lon = greatcircle_axis( latlon_viewer, perp)
    return lat

def azimuth(viewer,target):
    """ gives the azimuth between the sub-satellite (or sub-solar) point the viewer"""
    view_lat = viewer.T[1] * dtor
    view_lon = viewer.T[2] * dtor
    tg_lat = target.T[1] * dtor
    tg_lon = target.T[2] * dtor
    gam = numpy.arccos(numpy.cos(view_lat)*numpy.cos(tg_lon-view_lon))
    alp = numpy.min((numpy.tan(abs(view_lat))/numpy.tan(gam),1))
    bet = numpy.arccos(alp)
    az = (view_lat < 0)*((tg_lon >= view_lon)*bet + (tg_lon < view_lon)*(2*numpy.pi-bet)) + \
                       (view_lat >= 0)*((tg_lon >= view_lon)*(numpy.pi-bet) + (tg_lon < view_lon)*(numpy.pi+bet))
    return az/dtor


