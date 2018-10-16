from matplotlib import animation

def animate2(i, cover_set, mesh, color = 'red', time = 0):
    global cont, time_text, timewindow, counter
    counter.set_text(str(time+i+1) + ' Blocks')
    time_text.set_text( timewindow[time+i])
    ax1.add_patch( PolygonPatch(polygon=cover_set.geometry[i],color=color, alpha=0.2, zorder=4 ))
    for c in cont.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    cont = ax1.contourf(xv, yv, mesh[time+i,:,:], linspace(2,5,7), zorder=2, transform=ccrs.PlateCarree(), cmap='terrain', alpha=0.4)
        
mywriter = animation.FFMpegWriter(fps=5)

# #base covering set
# style.use('fivethirtyeight')
# fig, ax1 = subplots(figsize=(10,10), dpi=200)
# ax1 = axes(projection=geo)
# ax1.axis('scaled')
# ax1.gridlines(zorder=3)
# ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
#                 facecolor='white', edgecolor='black', linewidth=0.3, zorder=0)
# cont = ax1.contourf(xv, yv, af_window[20,:,:], linspace(2, 5, 7), zorder=2, transform=ccrs.PlateCarree(), cmap='terrain', alpha=0.4)
# cbar = colorbar(cont, shrink=0.5 )
# cbar.ax.set_ylabel('Airmass Factor', fontsize=8)
# time_text = ax1.text(0.8, 1., '', transform=ax1.transAxes)
# counter = ax1.text(0.1, 1., '', transform=ax1.transAxes)
# anim = animation.FuncAnimation(fig,partial(animate2, cover_set = baseline, mesh=af_window),frames=len(baseline))
# anim.save(directory + 'base_scan_order_'+timewindow[0].strftime('%Y%m%d')+'.mov', writer = mywriter)

#selected covering set
style.use('fivethirtyeight')
fig, ax1 = subplots(figsize=(10,10), dpi=200)
ax1 = axes(projection=geo)
ax1.axis('scaled')
ax1.gridlines(zorder=3)
ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                facecolor='white', edgecolor='black', linewidth=0.3, zorder=0)
cont = ax1.contourf(xv, yv, af_window[20,:,:], linspace(2, 5, 7), zorder=2, transform=ccrs.PlateCarree(), cmap='terrain', alpha=0.4)
cbar = colorbar(cont, shrink=0.5 )
cbar.ax.set_ylabel('Airmass Factor', fontsize=8)
time_text = ax1.text(0.8, 1., '', transform=ax1.transAxes)
counter = ax1.text(0.1, 1., '', transform=ax1.transAxes)
anim = animation.FuncAnimation(fig,partial(animate2, cover_set = coverset, mesh=af_window),frames=len(coverset))
anim.save(directory + 'scan_order4_'+timewindow[0].strftime('%Y%m%d')+'.mov', writer = mywriter)
#extra blocks
#anim = animation.FuncAnimation(fig, partial(animate2, cover_set = cover_extra, mesh=af_window, color='purple', time = len(coverset)), #frames=len(cover_extra))
#anim.save(directory + 'scan_extra_'+ timewindow[0].strftime('%Y%m%d')+'.mov', writer = mywriter)