from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import numpy as np
import glob, pdb, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import subprocess as sbp
from matplotlib import animation

block_names = []
scan_file = open('scan_order_031018.txt').readlines()
for l in scan_file:
    block_names.append(l.split()[1])
fid = Dataset('obrien_blocks/small_scan_blocks.nc4')
blocks = dict.fromkeys(block_names)
for b in block_names:
    blocks[b] = {}
    blocks[b]['lat'] = fid[b]['latitude_centre'][:]
    blocks[b]['lon'] = fid[b]['longitude_centre'][:]

m = Basemap(projection='geos',lon_0=-85,resolution='c')
def animate(i):
    clf()
    x,y = m(blocks[block_names[i]]['lon'],blocks[block_names[i]]['lat'])
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    m.plot(x[:,0],y[:,0],'m')
    m.plot(x[:,-1],y[:,-1],'m')
    m.plot(x[0,:],y[0,:],'m')
    m.plot(x[-1,:],y[-1,:],'m')

fig = figure()
anim = animation.FuncAnimation(fig,animate,frames=len(block_names))
anim.save('scan_order.mov')    
