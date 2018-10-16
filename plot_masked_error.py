err_mask = create_mask_err(error)
base_err_mask = create_mask_err(base_error)
plt.style.use('fivethirtyeight')


fig =  plt.figure(figsize=(5,3), dpi=300)
ax1 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                facecolor='gray' , edgecolor='black', linewidth=0.5,
               zorder=0)
ax1.gridlines(zorder=1, linewidth=0.5)
ax1.axis('scaled')
ax1.set_extent([-130, -30, -50, 50], crs=ccrs.PlateCarree())
#built-in plotting function for geopandas objects
#blockset.plot(ax=ax1, zorder=5, alpha=0.1, color='red', edgecolor='black')
plt.title('Algorithm, ' + timewindow[0].strftime('%d %b %Y') , fontsize=8 )
colmesh = ax1.pcolormesh(xv, yv, err_mask,  cmap = 'gist_ncar',
                         alpha = 0.5, zorder=2, shading='flat',
                         vmin=0.3, vmax=2, transform=ccrs.PlateCarree())
# cbar = plt.colorbar(colmesh, shrink=0.7)
# cbar.ax.tick_params(labelsize=10) 
# cbar.ax.set_ylabel('Error (in ppm)')
#plt.savefig(directory + timewindow[0].strftime('%Y%m%d') + '_spatial_dist_chosen')

#------------------------------------------


# plt.style.use('fivethirtyeight')
# fig =  plt.figure(figsize=(10,10), dpi=200)
ax2 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
ax2.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                facecolor='gray' , edgecolor='black', linewidth=0.5,
               zorder=0)
ax2.gridlines(zorder=1, linewidth=0.5)
ax2.axis('scaled')
ax2.set_extent([-130, -30, -50, 50], crs=ccrs.PlateCarree())
#built-in plotting function for geopandas objects
#blockset.plot(ax=ax1, zorder=5, alpha=0.1, color='red', edgecolor='black')
plt.title('Baseline, ' + timewindow[0].strftime('%d %b %Y'), fontsize=8)
colmesh2 = ax2.pcolormesh(xv, yv, base_err_mask,  cmap = 'gist_ncar',
                         alpha = 0.5, zorder=2, shading='flat',
                         vmin=0.3, vmax=2, transform=ccrs.PlateCarree())

cbar = plt.colorbar(colmesh, ax=[ax1, ax2], shrink=0.7)
cbar.ax.tick_params(labelsize=6) 
cbar.ax.set_ylabel('Error (in ppm)', fontsize=8)
plt.savefig(directory + timewindow[0].strftime('%Y%m%d') + '_spatial_dist_plate.pdf')
plt.show()

#==========================================
#==========================================

fig =  plt.figure(figsize=(5,3), dpi=300)
ax1 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                facecolor='gray' , edgecolor='black', linewidth=0.5,
               zorder=0)
ax1.gridlines(zorder=1, linewidth=0.5)
ax1.axis('scaled')
ax1.set_extent([-90, -30, -30, 15], crs=ccrs.PlateCarree())

plt.title('Algorithm, ' + timewindow[0].strftime('%d %b %Y') , fontsize=8 )
colmesh = ax1.pcolormesh(xv, yv, err_mask,  cmap = 'gist_ncar',
                         alpha = 0.5, zorder=2, shading='flat',
                         vmin=0.3, vmax=2, transform=ccrs.PlateCarree())

ax2 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
ax2.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                facecolor='gray' , edgecolor='black', linewidth=0.5,
               zorder=0)
ax2.gridlines(zorder=1, linewidth=0.5)
ax2.axis('scaled')
ax2.set_extent([-90, -30, -30, 15], crs=ccrs.PlateCarree())
plt.title('Baseline, ' + timewindow[0].strftime('%Y-%m-%d'), fontsize=8)
colmesh2 = ax2.pcolormesh(xv, yv, base_err_mask,  cmap = 'gist_ncar',
                         alpha = 0.5, zorder=2, shading='flat',
                         vmin=0.3, vmax=2, transform=ccrs.PlateCarree())

cbar = plt.colorbar(colmesh, ax=[ax1, ax2], shrink=0.7)
cbar.ax.tick_params(labelsize=6) 
cbar.ax.set_ylabel('Error (in ppm)', fontsize=8)
plt.savefig(directory + timewindow[0].strftime('%Y%m%d') + '_spatial_dist_amazon.pdf')
plt.show()


# ratio_mask = err_mask/base_err_mask

# plt.style.use('fivethirtyeight')
# fig =  plt.figure(figsize=(10,10), dpi=200)
# ax1 = axes(projection=geo)
# ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
#                 facecolor='gray' , edgecolor='black', linewidth=0.5,
#                zorder=0)
# ax1.gridlines(zorder=1)
# ax1.axis('scaled')
# #built-in plotting function for geopandas objects
# #blockset.plot(ax=ax1, zorder=5, alpha=0.1, color='red', edgecolor='black')
# #plt.title('Algorithm/Baseline Error Ratio, ' + timewindow[0].strftime('%Y-%m-%d'))
# colmesh = ax1.pcolormesh(xv, yv, ratio_mask,  cmap = 'seismic',
#                          alpha = 0.5, zorder=2, shading='gouraud',
#                          vmin=0, vmax=2, transform=ccrs.PlateCarree())
# cbar = plt.colorbar(colmesh, shrink=0.7)
# cbar.ax.tick_params(labelsize=10) 
# cbar.ax.set_ylabel('Error (in ppm)')
# plt.savefig(directory + timewindow[0].strftime('%Y%m%d') + '_spatial_dist_ratio')