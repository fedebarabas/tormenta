import library.readinsight3 as readinsight3
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

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

# plt.plot(0.001*x_locs, 0.001*y_locs, 'b', marker='o', markersize=0.5, linestyle='None')
# # plt.scatter(x_locs, y_locs,s=sigma)
# plt.xlim((0,0.001*shape[0]))
# plt.ylim((0,0.001*shape[1]))
# plt.xlabel('$\mu$m')
# plt.ylabel('$\mu$m')
# # plt.show()
# # p rint(campoi)

# imagen = np.zeros([np.ceil(shape[0]/render_px), np.ceil(shape[1]/render_px)])     # I need to add px_size to each definition due to indexing
# # pimagen = np.zeros([(shape[0]+px_size)/render_px, (shape[1]+px_size)/render_px])     # I need to add px_size to each definition due to indexing
# # rint 'imagen_px ', shape[0]/render_px, shape[1]/render_px
# for i in np.arange(len(x_locs)):
#     print x_locs[i], np.trunc(x_locs[i]/render_px) 
#     imagen[np.trunc(x_locs[i]/render_px),np.trunc(y_locs[i]/render_px)] += 1    # I need to divide by render_px to place the point in the final image
# 
# plt.imshow(imagen,cmap='gray',interpolation='none')
# plt.colorbar()
# plt.show()

H, xedges, yedges = np.histogram2d(0.001*y_locs, 0.001*x_locs, bins=np.ceil(shape/render_px))

punto1 = find_nearest(xedges,3.1)[1],find_nearest(yedges,12.6)[1]
punto2 = find_nearest(xedges,4.6)[1],find_nearest(yedges,9.8)[1]
print punto1, punto2
print H[punto1], H[punto2]
length = int(np.hypot(punto2[0]-punto1[0], punto2[1]-punto1[1]))
print length
lin_x, lin_y = np.linspace(punto1[0], punto2[0], length), np.linspace(punto1[1], punto2[1], length)
profile = H[lin_x.astype(np.int), lin_y.astype(np.int)]

extent = [yedges[0], yedges[-1],  xedges[0], xedges[-1]]

# fig, axes = plt.subplots(1,2)
# axes[0].imshow(H, extent=extent, cmap='gray', vmax = 10, interpolation='none')
# # axes[0].plot([punto1[0], punto2[0]], [punto1[1], punto2[1]], 'ro-')
# axes[0].plot([3.1, 4.6], [12.6, 9.8], 'ro-')
# axes[0].axis('image')
# axes[1].plot(profile)

plt.imshow(H, extent=extent, cmap='gray', vmax = 10, interpolation='none')
# axes[0].plot([punto1[0], punto2[0]], [punto1[1], punto2[1]], 'ro-')
# axes[0].plot([3.1, 4.6], [12.6, 9.8], 'ro-')
plt.show()
