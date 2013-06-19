import library.readinsight3 as readinsight3
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
import numpy as np
import scipy.ndimage
import matplotlib.cm as cm

def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return [array[idx],idx]

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

shape = np.array([191,183])
px_size = 133                 # pixel size in the source images, in nm
render_px = 20                # pixel size in the final image, in nm
shape = px_size*shape         # shape conversion to um

i3_data_in = readinsight3.loadI3GoodOnly(sys.argv[1])
x_locs = i3_data_in['xc']*px_size                 # px -> um conversion
y_locs = i3_data_in['yc']*px_size
# campoi = i3_data_in['i']

H, xedges, yedges = np.histogram2d(0.001*y_locs, 0.001*x_locs, bins=np.ceil(shape/render_px))

# Image saving 
scipy.misc.imsave('out.png', H)


punto1 = find_nearest(xedges,4.5)[1],find_nearest(yedges,10)[1]
punto2 = find_nearest(xedges,5.2)[1],find_nearest(yedges,8.6)[1]
print punto1, punto2
print H[punto1], H[punto2]
length = int(np.hypot(punto2[0]-punto1[0], punto2[1]-punto1[1]))
length = 1000000
print length
lin_x, lin_y = np.linspace(punto1[0], punto2[0], length), np.linspace(punto1[1], punto2[1], length)
profile = H[lin_x.astype(np.int), lin_y.astype(np.int)]
# profile = scipy.ndimage.map_coordinates(H, np.vstack((lin_x,lin_y)))

extent = [yedges[0], yedges[-1],  xedges[0], xedges[-1]]

# dpi = 180
# margin = 0.05 # (5% of the width/height of the figure...)
# figsize = (1 + margin) * np.ceil(shape/render_px) / dpi
# figsize = (1 + margin)*H.shape[1]/dpi, (1 + margin)*H.shape[0]/dpi
print  H.shape[1], H.shape[0]
fig = plt.figure(figsize=(8.0, 10.0))
# fig = plt.figure(figsize=figsize, dpi=dpi)
gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])

ax1 = plt.subplot(gs[0])
img = ax1.imshow(H, extent=extent, cmap='gray', vmax = 10, interpolation='none')
plt.xlabel('$\mu$m')
plt.ylabel('$\mu$m')
print fig.get_size_inches()
fig.savefig('ax2_figure.png',  bbox_inches='tight')
fig.colorbar(img, ax=ax1)
# axes[0].plot([punto1[0], punto2[0]], [punto1[1], punto2[1]], 'ro-')
ax1.plot([4.5, 5.2], [10, 8.6], 'ro-')
ax1.axis('image')

ax2 = plt.subplot(gs[1])
ax2.plot(profile)
plt.show()
