# -*- coding: utf-8 -*-
"""

This software is to perform some calibration and pre-processing operations on the light field microscopy (LFM) data, including:

- Detect the rotation angle of a raw LFM image.
- Detect lenslet pitch.
- Crop a region of interest (ROI) from each LFM frame.
- Detect the centers of the microlens array using an out-of-focus LFM image, instead of a white image.
- Extract micro-images to construct 4D LFM data.
- Convert 4D LFM data to multi-view / sub-aperture images.
- Perform matrix factorization to separate the foreground and background.
- Convert the foreground to a clean LF image.
- Construct Epi-Polar Images (EPIs)


Original data is from: 
[path]/Chronos_Test/chronos_190301/MLA_1x1_f_2-8_50ms_490nm_1
Now the data is copied to:
[path]/LFMdata/LFimg.tif


References:
----------------------------
[1] P. Song, H. Verinaz Jadan, C. L. Howe, P. Quicke, A. J. Foust and P. L. Dragotti, "3D Localization for Light-Field Microscopy via Convolutional Sparse Coding on Epipolar Images," in IEEE Transactions on Computational Imaging, doi: 10.1109/TCI.2020.2997301.
[2] P. Song, H. Verinaz Jadan, P. Quicke, C. L. Howe, A. J. Foust, and P. L. Dragotti, "Location Estimation for Light Field Microscopy based on Convolutional Sparse Coding," in Imaging and Applied Optics 2019 (COSI, IS, MATH, pcAOP), OSA Technical Digest (Optical Society of America, 2019), paper MM2D.2

    
Usage:
----------------------------

Run 'main_LFM_preprocess.py' to explore each calibration and pre-processing operation.

If you want to use your own images, please put them in folder "LFMdata", update the 'WhiteImg.tif' with an appropriate out-of-focus light-field image, and update the data path and some parameters accordingly in the code file.


Contact:
----------------------------
Please report problems to

Pingfan Song

Electronic and Electrical Engineering,

Imperial College London

p.song@imperial.ac.uk


# ----------------

# ROI 1: (x,y) = (xxx,xxx) from frame 1 to frame 55, i.e. depth from -18 to 36 um, i.e. depth 0 at frame 19;


21/07/2021 Construct i-k and j-l EPIs simultaneously.
20/07/2021 Use 2D peak detection instead of thresholds to find neurons. 
20/07/2021 Even though adding back the trends' slopes to detrended data can preserve contrast, it can not reduce the background. So we multiply the detrended data by the slopes. 
16/07/2021 add back the trends' slopes to detrended data to preserve contrast and obtain detrended frames with better quality. 
16/07/2021 add pre-filtering before detrending to give a better detrend performance. Otherwise, the detrended frames have a very poor quality.
15/07/2021 add post-filtering after detrending to get a good std image.
13/07/2021 detrend the functional LFM to remove trends caused by photobleaching.
7/07/2021 decode functional light-field microscopy cell data
22/09/2020 Add an option that manually specifies the coordinates of the neurons, i.e. coords_list, in each frame.
01/07/2020 Construct EPIs for detected all of spatial positions.
06/03/2019 made a thorough clean to tidy and simplify the code. 
05/03/2019 average each row and column of centers, use averaged coordinates.
generate EPI arrays. Show a set of EPI in an image.

@author: p.song@imperial.ac.uk

"""


#%% ---------------------------------------------- 

import numpy as np
import f.lightfield_functions as lf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import tifffile
import copy
import scipy.io
import scipy.ndimage.filters as filters
from scipy.ndimage import gaussian_filter
from scipy import signal

import skimage.transform
import skimage.filters
import f.general_functions as gf

#import matlab.engine
from numpy.linalg import svd
#from PIL import Image
import imageio
import os
import h5py
import pandas as pd
from skimage.feature import peak_local_max
from skimage import measure

#%% ---------------------------------------------- 
# Detrend the functional LFM data (image series). Perform pixel-wise detrend as the trends are different for different pixels.


#% read tif image task
#% FileDir = './s1a1_LF_1P_1x1_400mA_100msExp_func_600frames_1/s1a1_LF_1P_1x1_400mA_crop537_x500y850.tif'
FileDir = './s1a1_LF_1P_1x1_400mA_100msExp_func_600frames_1/s1a1_LF_1P_1x1_400mA_func_600frames_1044pix.tif'

#% FileDir = './s1a1_WF_1P_1x1_400mA_100msExp_func_600frames_1/s1a1_WF_1P_1x1_400mA_crop537_x500y850.tif'

imgs = tifffile.imread(FileDir)
frame_start = 10
frame_end = 510 #imgs.shape[0];
imgs = imgs[frame_start:frame_end,:,:]


##===========================
# Perform per-pixel detrending and pre/post 3D Gaussian filtering at the whole field of view.

_, height, width = imgs.shape
imgs_prefilter = np.zeros_like(imgs, dtype='int16')
imgs_detrend = np.zeros_like(imgs, dtype='int16')
imgs_postfilter = np.zeros_like(imgs, dtype='int16')
trends = np.zeros_like(imgs, dtype='int16')
trends_std = np.zeros( (height, width), dtype='int16')
trends_mean = np.zeros( (height, width), dtype='int16')
std_raw = np.zeros( (height, width), dtype='int16')
std_prefilter = np.zeros( (height, width), dtype='int16')
std_detrend = np.zeros( (height, width), dtype='int16')
std_postfilter = np.zeros( (height, width), dtype='int16')

#Iterative pre-filtering, detrending, post-filtering over local region to save memory.
# set border larger than 0 to eliminate border effect caused by filtering.
stride, border = 300, 20
for row in range(border,height,stride):
	for col in range(border,width,stride):

#		# crop a local region as example
#		stride, border = 100, 0
#		row, col = 550+border, 200+border

		imgs_local = imgs[:, row-border:row+stride+border, col-border:col+stride+border]
		
		# pre-filtering using 3D Gaussian filtering
		imgs_prefilter_local = skimage.filters.gaussian(imgs_local, sigma=[1.5,1.5,1.5], preserve_range=True).astype('int16')
		
		# pixel-wise detrending
		imgs_detrend_local = scipy.signal.detrend(imgs_prefilter_local, axis=0, type='linear').astype('int16')
		
		# store trends
		trends_local = imgs_prefilter_local - imgs_detrend_local
		trends_std_local = trends_local.std(axis=0)
		trends_mean_local = trends_local.mean(axis=0) # the mean of trends is used as the white image for calibration (rotation and center detection).
		
		# flip up-and-down due to negative dye.
		imgs_detrend_local = -imgs_detrend_local 

		# post-filtering using 3D Gaussian filtering
		imgs_postfilter_local = skimage.filters.gaussian(imgs_detrend_local, sigma=[2,2,2], preserve_range=True).astype('int16')
	
		# Enhance the contrast by multiplying the std of the trends, since this operation reduces the intensity of less active region while increases the intensity of more active region, leading to amplified contrast.
		imgs_postfilter_local = imgs_postfilter_local*(2*trends_std_local/trends_std_local.max()) # multiplying with std to amplify contrast.
		#imgs_postfilter_local = imgs_postfilter_local+ 10*(trends_mean_local-trends_mean_local.min())/(trends_mean_local.max()-trends_mean_local.min()) # preserve the intensity by adding back the min-max normalized mean. No effect on the final std.
	
		# compute STD of pre-filtered, detrened, and post-filtered frames
		std_raw_local = np.std(imgs_local, axis=0)
		std_prefilter_local =  np.std(imgs_prefilter_local, axis=0)
		std_detrend_local =  np.std(imgs_detrend_local, axis=0)		
		std_postfilter_local = np.std(imgs_postfilter_local, axis=0)
		
		# store pre-filtering, detrending, post-filtering local region into the whole region.
		imgs_prefilter[:, row:row+stride, col:col+stride] = imgs_prefilter_local[:, border:stride+border, border:stride+border]
		
		imgs_detrend[:, row:row+stride, col:col+stride] = imgs_detrend_local[:, border:stride+border, border:stride+border]
		
		imgs_postfilter[:, row:row+stride, col:col+stride] = imgs_postfilter_local[:, border:stride+border, border:stride+border]

		trends[:, row:row+stride, col:col+stride] = trends_local[:, border:stride+border, border:stride+border]		
		
		trends_std[row:row+stride, col:col+stride] = trends_std_local[border:stride+border, border:stride+border]
		
		trends_mean[row:row+stride, col:col+stride] = trends_mean_local[border:stride+border, border:stride+border]
		
		std_raw[row:row+stride, col:col+stride] = std_raw_local[border:stride+border, border:stride+border]
		
		std_prefilter[row:row+stride, col:col+stride] = std_prefilter_local[border:stride+border, border:stride+border]
		
		std_detrend[row:row+stride, col:col+stride] = std_detrend_local[border:stride+border, border:stride+border]
		
		std_postfilter[row:row+stride, col:col+stride] = std_postfilter_local[border:stride+border, border:stride+border]
		

		plt.figure()
		plt.imshow(std_postfilter, vmin=0, vmax=30)
#		plt.savefig('STD', transparent = True, bbox_inches='tight',pad_inches = 0,dpi=300)
		plt.show()
		

## find the sequence with the smallest slope as the baseline.
#ind = np.unravel_index(trends_std.argmin(), trends_std.shape)
#baseline = trends_local[:,ind[0],ind[1]]

#%%
# show results at a local region 
stride, border = 100, 0
row, col = 550, 200
imgs_local = imgs[:, row:row+stride, col:col+stride]
imgs_prefilter_local = imgs_prefilter[:, row:row+stride, col:col+stride]
imgs_detrend_local = imgs_detrend[:, row:row+stride, col:col+stride]
imgs_postfilter_local = imgs_postfilter[:, row:row+stride, col:col+stride]
trends_local = trends[:, row:row+stride, col:col+stride]

std_raw_local = std_raw[row:row+stride, col:col+stride]
std_prefilter_local = std_prefilter[row:row+stride, col:col+stride]
std_detrend_local = std_detrend[row:row+stride, col:col+stride]
std_postfilter_local = std_postfilter[row:row+stride, col:col+stride]


# show trends at different pixels
fig = plt.figure(figsize=(9,6))
fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
plt.subplot(2,2,1)
plt.plot(imgs_local[:,50,[25, 50, 75]])
plt.legend(['pixel (50,25)', 'pixel (50,50)', 'pixel (50,75)'], loc='lower left') #loc='lower left'
plt.title('raw signals')

plt.subplot(2,2,2)
plt.plot(imgs_prefilter_local[:,50,[25, 50, 75]])
plt.plot(trends_local[:,50,[25, 50, 75]])
plt.title('trends of pre-filtered signals ')

plt.subplot(2,2,3)
plt.plot(imgs_detrend_local[:,50,[25, 50, 75]])
plt.title('detrended')

plt.subplot(2,2,4)
plt.plot(imgs_postfilter_local[:,50,[25, 50, 75]])
plt.title('post-filtered')

#FigName = 'seq_detrend_with_only_post-filtering.png'
#FigName = 'seq_detrend_with_pre-post-filtering.png'
FigName = 'seq_detrend_with_pre-post-filtering_HighContrast.png'
plt.savefig(FigName, transparent = True, bbox_inches='tight',pad_inches = 0,dpi=200)
plt.show()


# show raw, detrended, and detrended+smoothed results.
fig=plt.figure(figsize=(9,5))
fig.subplots_adjust(hspace = 0.3, wspace=0.3)

plt.subplot(2,4,1)
plt.imshow(imgs_local[110,:,:], cmap='inferno')
plt.title('raw frame')
plt.subplot(2,4,2)
plt.imshow(imgs_prefilter_local[110,:,:], cmap='inferno')
plt.title('pre-filtered frame')
plt.subplot(2,4,3)
plt.imshow(imgs_detrend_local[110,:,:], cmap='inferno')
plt.title('detrended frame')
plt.subplot(2,4,4)
plt.imshow(imgs_postfilter_local[110,:,:], cmap='inferno')
plt.title('post-filtered frame')

plt.subplot(2,4,5)
#plt.imshow(imgs_local.std(axis=0), cmap='inferno')
plt.imshow(std_raw_local, cmap='inferno')
plt.title('std of raw data')
plt.subplot(2,4,6)
#plt.imshow(imgs_prefilter_local.std(axis=0), cmap='inferno')
plt.imshow(std_prefilter_local, cmap='inferno')
plt.title('std of pre-filtered')
plt.subplot(2,4,7)
#plt.imshow(imgs_detrend_local.std(axis=0), cmap='inferno')
plt.imshow(std_detrend_local, cmap='inferno')
plt.title('std of detrended')
plt.subplot(2,4,8)
#plt.imshow(imgs_postfilter_local.std(axis=0), cmap='inferno')
plt.imshow(std_postfilter_local, cmap='inferno')
plt.title('std of post-filtered')

#FigName = 'LFM_detrend_with_only_post-filtering.png'
#FigName = 'LFM_detrend_with_pre-post-filtering.png'
FigName = 'LFM_detrend_with_pre-post-filtering_HighContrast.png'
plt.savefig(FigName, transparent = True, bbox_inches='tight',pad_inches = 0,dpi=200)
plt.show()

plt.imshow(imgs_postfilter_local[110,:,:], cmap='inferno')
plt.colorbar()
plt.show()
plt.imshow(imgs_postfilter_local.std(axis=0), cmap='inferno')
plt.colorbar()
plt.show()


fig=plt.figure(figsize=(5,5))
plt.imshow(trends_mean, cmap='inferno')
plt.savefig('trends_mean.png', transparent = True, bbox_inches='tight',pad_inches = 0,dpi=200)
plt.show()

fig=plt.figure(figsize=(5,5))
plt.imshow(std_postfilter, cmap='inferno') # , vmax=30)
plt.savefig('std_postfilter.png', transparent = True, bbox_inches='tight',pad_inches = 0,dpi=200)
plt.show()

#%%
# save detrended functional LFM data
with tifffile.TiffWriter('./imgs_detrend.tif', bigtiff=True) as tif:
	for i in range(imgs_detrend.shape[0]):
		tif.save(imgs_detrend[i,:,:])
		
# save post-filtered functional LFM data
with tifffile.TiffWriter('./imgs_postfilter.tif', bigtiff=True) as tif:
	for i in range(imgs_postfilter.shape[0]):
		tif.save(imgs_postfilter[i,:,:])
		
with tifffile.TiffWriter('./std_detrend.tif', bigtiff=True) as tif:
	tif.save(std_detrend.astype('float16'))
	
with tifffile.TiffWriter('./std_postfilter.tif', bigtiff=True) as tif:
	tif.save(std_postfilter)	
	
with tifffile.TiffWriter('./trends_mean.tif', bigtiff=True) as tif:
	tif.save(trends_mean)	
	
with tifffile.TiffWriter('./trends_std.tif', bigtiff=True) as tif:
	tif.save(trends_std)	

#del imgs, imgs_prefilter, imgs_detrend, imgs_postfilter 

#%% load detrended functional LFM data
FileDir = './imgs_postfilter.tif'
imgs_postfilter = tifffile.imread(FileDir)

FileDir = './std_postfilter.tif'
std_postfilter = tifffile.imread(FileDir)

FileDir = './trends_mean.tif'
trends_mean = tifffile.imread(FileDir)



#%% ---------------------------------------------- 
# set the LFM image path and the path of a white image or a non-focused LF image.
#LF_path = r'./s1a1_LF_1P_1x1_400mA_100msExp_func_600frames_1/s1a1_LF_1P_1x1_400mA_crop537_x500y850.tif'  # LF cell data.

#LF_path = r'./s1a1_LF_1P_1x1_400mA_100msExp_func_600frames_1/s1a1_LF_1P_1x1_400mA_100msExp_func_600frames_1_MMStack_Default.ome.tif'  # Func LF cell data 4.3G.

#LF_path = r'./s1a1_LF_1P_1x1_400mA_100msExp_func_600frames_1/s1a1_LF_1P_1x1_400mA_func_128frames_SZ2048.tif'  # Func LF cell data 1.1G.

LF_path = r'./std_postfilter.tif'  # STD of smoothed and detrended Func LF cell data.
WhiteImg_path = r'./trends_mean.tif'  # STD of detrended Func LF cell data

#WhiteImg_path = r'./LF_1P_1x1_200mA_100msExp_whiteImage_below_1/LF_1P_1x1_200mA_100msExp_whiteImage_below_1_MMStack_Default.ome.tif' # whole no-focused LFM image of size 2048x2048. # LF cell data.

mydpi = 300

#%% ---------------------------------------------- 
#--Detect the rotation angle of a raw LFM image and the lenslet pitch on a white image or a out-of-focus LF image.

whiteimg = tifffile.imread(WhiteImg_path)

# Method 1: Load already detected parameters, e.g. lenslet pitch r, rotation angle, spot_period.
params = np.load('./Params_FuncLFM.npz')
r, im_angle, spot_period = params['r'], params['im_angle'], params['spot_period']
plt.imshow( skimage.transform.rotate(whiteimg,im_angle) )

## Method 2: Detect parameters, e.g. lenslet pitch r, rotation angle, spot_period on a white image or a out-of-focus LF image. We recommend to use a large LF image to ensure accuracy. For example, using a LF image of size 1044x1044 pixels, we get (r, im_angle, spot_period) = ([ 1.253 19.293] 3.715 19.33) on STD image.

#
#crop_flag = 0 # 1: crop an area; 0: no cropping
#if crop_flag > 0:
#    center = np.array([1022,803])
#    size = np.array([257,257]) # np.array([257,257]) # np.array([513,513])
#    roi = np.array([[center[0]-np.floor(size[0]/2), center[0]+np.floor(size[0]/2)], 
#                     [center[1]-np.floor(size[1]/2), center[1]+np.floor(size[1]/2)]]) # region of interest
#    roi = roi.astype(int)
#    whiteimg = whiteimg[roi[0,0]:roi[0,1]+1,roi[1,0]:roi[1,1]+1]
#    whiteimg = whiteimg/np.amax(whiteimg)
#    print('using cropped image for estimating rotation, etc.')
#else:
#    whiteimg = whiteimg/np.amax(whiteimg)
#    print('using whole image for estimating rotation, etc.')
#
#r, im_angle, spot_period = lf.find_angle(whiteimg)
#print(r, im_angle, spot_period)
#plt.imshow( skimage.transform.rotate(whiteimg,im_angle) )
#
#FileName = './Params_FuncLFM.npz'
#np.savez(FileName,r = r,im_angle=im_angle, spot_period=spot_period)

#%% ---------------------------------------------- 
#--Crop an region of interest (ROI) from each LFM frame.

ROI_index = 9 # 9 for functional LFM

if ROI_index == 1:
    # ROI 1: (x,y) = (732,1756) from frame 1 to frame 55, i.e. depth from -18 to 36 um, i.e. depth 0 at frame 19;
    center = np.array([1756,732])
    depthStart = -18 
    frameStart, frameEnd, frameStep = 0, 55, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;
    
elif ROI_index == 2:
    # ROI 2: (x,y) = (552,1585) from frame 1 to frame 55, i.e. depth from -18 to 36 um, i.e. depth 0 at frame 19;
    center = np.array([1585,552]) 
    depthStart = -18
    frameStart, frameEnd, frameStep = 0, 55, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;
    
elif ROI_index == 3:
    # ROI 3: (x,y) = (1075,1357) from frame 1 to frame 55, i.e. depth from -18 to 36 um, i.e. depth 0 at frame 19;
    center = np.array([1357,1075]) 
    depthStart = -18 
    frameStart, frameEnd, frameStep = 0, 55, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;
    
elif ROI_index == 4:
    # ROI 4: (x,y) = (1007,1808) from frame 1 to frame 55, i.e. depth from -21 to 33 um, i.e. depth 0 at frame 22;
    center = np.array([1808,1007]) 
    depthStart = -21 
    frameStart, frameEnd, frameStep = 0, 55, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;
    
elif ROI_index == 5:
    # ROI 5: (x,y) = (1209,1217) from frame 1 to frame 55, i.e. depth from -18 to 36 um, i.e. depth 0 at frame 19;
    center = np.array([1217,1209]) 
    depthStart = -18 
    frameStart, frameEnd, frameStep = 0, 55, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;
    
elif ROI_index == 6:
    # ROI 6: (x,y) = (1458,1015) from frame 1 to frame 55, i.e. depth from -25 to 29 um, i.e. depth 0 at frame 26;
    center = np.array([1015,1458]) 
    depthStart = -25 
    frameStart, frameEnd, frameStep = 0, 55, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;
    
elif ROI_index == 7:
    # ROI 7: (x,y) = (1798,1281) from frame 1 to frame 55, i.e. depth from -21 to 33 um, i.e. depth 0 at frame 22;
    center = np.array([1281,1798]) 
    depthStart = -21 
    frameStart, frameEnd, frameStep = 0, 55, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;
    
elif ROI_index == 8:
    # ROI 8: (x,y) = (1453,781) from frame 1 to frame 55, i.e. depth from -29 to 25 um, i.e. depth 0 at frame 30;
    center = np.array([781,1453]) 
    depthStart = -29 
    frameStart, frameEnd, frameStep = 0, 55, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;

elif ROI_index == 9:
    # ROI 9: 
#    center = np.array([1106,756]) 
#	center = np.array([650,280]) 
	center = np.array([300, 50]) + np.array([700, 700])//2
	depthStart = 0 
	frameStart, frameEnd, frameStep = 0, 500, 1 # Extract ROI from frame frameStart to frame frameEnd with step of frameStep;

else:
    print('Specified ROI does not exist!')
    

Num = int((frameEnd-1-frameStart)/frameStep + 1)

H, W = 700, 700 # 301, 301, # 257, 257 # 513, 513; Height and Width
size = np.array([H,W]) # Height and Width
roi = np.array([[center[0]-np.floor(size[0]/2), center[0]+np.floor(size[0]/2)], 
                 [center[1]-np.floor(size[1]/2), center[1]+np.floor(size[1]/2)]]) # region of interest  [left, right, up, down]
roi = roi.astype(int)


## load LFM image and crop ROIs
#path = r'./detrended_LFM_std.mat' 
##stack = scipy.io.loadmat(path) 
#arrays = {}
#f = h5py.File(path)
#for k, v in f.items():
#	print(k)
#	arrays[k] = np.array(v)
#stack = arrays[k]
#stack = stack[np.newaxis,:,:]
#LF_arr = stack.copy()

stack = imgs_postfilter  # std_postfilter # imgs_postfilter
#stack = tifffile.imread(LF_path) # 3D data with a format as XYZ
if stack.ndim == 2:
	stack = stack[np.newaxis,:,:]

LF_arr = np.zeros((Num,size[0],size[1]), dtype='int16') #  a set of LF images
LF_cal_arr = np.zeros((Num,size[0],size[1]), dtype='int16') # a set of calibrated LF images
ind = 0

show_fig = 0

for frameInd in np.arange(frameStart, frameEnd, frameStep ):
    img = copy.copy(stack[frameInd,roi[0,0]:roi[0,1],roi[1,0]:roi[1,1]])
    LF_arr[ind,:,:] = img
    
    maxval = np.amax(img)
    img = skimage.transform.rotate(img,im_angle)
    img = (img/np.amax(img))*maxval
    img = img.astype('uint16')
    LF_cal_arr[ind,:,:] = img
     
    if show_fig:
        # show original ROI
        plt.figure(figsize = (3,3))
        plt.imshow(LF_arr[ind,:,:], cmap='inferno', vmin=0, vmax=None) # plasma, viridis, magma,
        plt.axis('off')
        titleName = './frame'+str(frameInd+1)+'_size'+ str(size[0]) + '_orig.png'
#        plt.savefig(titleName,transparent = True,bbox_inches='tight',pad_inches = 0,dpi=300) #bbox_inches='tight',pad_inches = 0,
        plt.show()
        
        # show rotated ROI
        plt.figure(figsize = (3,3))
        plt.imshow(LF_cal_arr[ind,:,:], cmap='inferno', vmin=0)
        plt.axis('off')
#        plt.minorticks_on()
#        plt.grid(b=True, which='both',color='b', linestyle='-', linewidth=0.5)
        titleName = './frame'+str(frameInd+1)+'_size'+ str(size[0]) + '_rotate.png'
#        plt.savefig(titleName,transparent = True,bbox_inches='tight',pad_inches = 0,dpi=300) #bbox_inches='tight',pad_inches = 0,
        plt.show()
        
        # show original and rotated ROI together
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,3))
        ax1.imshow(LF_arr[ind,:,:],cmap='inferno', vmin=0); ax1.set_title('original')
        ax2.imshow(LF_cal_arr[ind,:,:],cmap='inferno', vmin=0); ax2.set_title('rotated')
        ax1.axis('off'); ax2.axis('off')
        titleName = './frame'+str(frameInd+1)+'_size'+ str(size[0]) + '_orig_rot.png'
#        plt.savefig(titleName,transparent = True,pad_inches = 0,bbox_inches='tight',dpi=300)
        plt.show()
        
    ind = ind + 1

LF_arr = LF_arr.astype('int16')
LF_cal_arr = LF_cal_arr.astype('int16')

# write the LF data in .tif image
#with tifffile.TiffWriter('./LF_arr.tif', bigtiff=True) as tif:
#    for i in range(LF_arr.shape[0]):
#        tif.save(LF_arr[i,:,:])
        
with tifffile.TiffWriter('./LF_cal_arr.tif', bigtiff=True) as tif:
    for i in range(LF_cal_arr.shape[0]):
        tif.save(LF_cal_arr[i,:,:])

## read the LF data array
#LF_cal_arr = tifffile.TiffFile('LF_cal_arr.tif').asarray()

## show LF array
#fig, axes = plt.subplots(3,3,figsize=(10,10))
#fig.subplots_adjust(hspace = 0.3, wspace=0.3)
#axes = axes.ravel()
#VMAX_LF = np.amax(LF_cal_arr)
#for i in np.arange(0, 9 ):
#    frameInd = i* (LF_cal_arr.shape[0]//9) # i*4+2
#    axes[i].imshow(LF_cal_arr[frameInd,:,:],cmap='inferno',vmin=0,vmax=VMAX_LF); # ,aspect='auto'
##    axes[i].grid(True)
#    axes[i].set_title('LF_depth_'+str(frameInd*frameStep+depthStart)+'um')
#    axes[i].set_xlabel('k axis')
#    axes[i].set_ylabel('i axis')
#
#plt.savefig('./LF_cal_arr'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
#plt.show()

# show LF array in a row
fig, axes = plt.subplots(1,6,figsize=(20,10))
fig.subplots_adjust(hspace = 0.1, wspace=0.1)
axes = axes.ravel()
VMAX_LF = np.amax(LF_cal_arr)
for i in np.arange(0, 6 ):
    frameInd = i* (LF_cal_arr.shape[0]//9) # i*4+6
    axes[i].imshow(LF_cal_arr[frameInd,:,:],cmap='inferno',vmin=0,vmax=VMAX_LF); # ,aspect='auto'
#    axes[i].grid(True)
    axes[i].axis('off')
    axes[i].set_title('LF_depth_'+str(frameInd*frameStep+depthStart)+'um')
    axes[i].set_xlabel('l axis')
    axes[i].set_ylabel('k axis')

plt.savefig('./LF_cal_arr_row'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
plt.show()

#del LF_arry

#%% ----------------------------------------------
#--Detect the centers of microlens array using an out-of-focus LFM image, instead of a white image. 

# Method 1: Load the off-the-shelf detected centers of microlens array. Then, only need to detect the center of the single lenslet located at the image central region. Finally, use the newly detected single center as the anchored to modify the off-the-shelf centers.

centers_path = 'LF_Cell_centers_ROI_x1453_y781_size513.npy'
if os.path.exists(centers_path) :
    centers=np.load(centers_path)

# load and crop a white image
whiteimg = tifffile.imread(WhiteImg_path)
whiteimg = copy.copy(whiteimg[roi[0,0]:roi[0,1]+1,roi[1,0]:roi[1,1]+1])
whiteimg0 = np.copy(whiteimg)

#whiteimg = tifffile.imread(titleName)
whiteimg = whiteimg0.copy()
whiteimg = whiteimg/np.amax(whiteimg) # whiteimg = (whiteimg- np.amin(whiteimg))/(np.amax(whiteimg) - np.amin(whiteimg))
whiteimg = skimage.transform.rotate(whiteimg,im_angle)
whiteimg = whiteimg/np.amax(whiteimg)

# binarize
threshold = 0.3*np.amax(whiteimg) # threshold for binarizing. Try 0.01, 0.015,0.05 0.02;  
whiteimg[whiteimg<=threshold] = 0
whiteimg[whiteimg>threshold] = 1

initial_center = (np.array([whiteimg.shape[0]/2, whiteimg.shape[0]/2])).astype(int)
#initial_center = (np.array([128, 128])).astype(int)
#center0 = lf.find_center(whiteimg, spot_period,initial_center,
#                         kernel_type=0,quiet=False)
center0 = lf.find_center_robust(whiteimg, spot_period,initial_center,
                     kernel_type=2,quiet=False)

# show white image with the center
plt.figure(figsize=(10,10)); 
plt.imshow(whiteimg, cmap = 'gray')
plt.scatter(center0[1],center0[0], c='b', s=5)
plt.grid()


rad_spots = np.int(np.floor((size[0]/spot_period-1)/2)-1) #11 for size 513, # 5 for size 257
im_dim = 2*rad_spots+1 # number of lenslets to be used

center_bias = center0 - centers[centers.shape[0]//2, centers.shape[1]//2, :]
centers = centers + center_bias

assert centers.shape[0] >= im_dim and centers.shape[1] >= im_dim
    

#%% ----------------------------------------------
 
#Method 2: Detect the centers of microlens array using an out-of-focus LFM image, instead of white image.

# load and crop a white image
whiteimg = tifffile.imread(WhiteImg_path)
whiteimg = copy.copy(whiteimg[roi[0,0]:roi[0,1]+1,roi[1,0]:roi[1,1]+1])
whiteimg0 = np.copy(whiteimg)

## binarize
#threshold = 0.04*np.amax(whiteimg)
#whiteimg[whiteimg<=threshold] = 0
#whiteimg[whiteimg>threshold] = 1

titleName = './WhiteImg'+'_size'+ str(size[0])
tifffile.imsave(titleName+'.tif', whiteimg)
mydpi = 300
plt.figure(figsize = (3,3))
plt.imshow(whiteimg, cmap='inferno')
plt.axis('off')
#plt.minorticks_on()
#plt.grid(b=True, which='both',color='b', linestyle='-', linewidth=0.5)
plt.savefig(titleName+'.png',transparent = True,pad_inches = 0,bbox_inches='tight',dpi=mydpi)
plt.show()


# find the centers of microlens array in the rotated image via convolution with a template.

#whiteimg = tifffile.imread(titleName)
whiteimg = whiteimg0.copy()
whiteimg = whiteimg/np.amax(whiteimg)
whiteimg = skimage.transform.rotate(whiteimg,im_angle)
whiteimg = whiteimg/np.amax(whiteimg)

titleName = './WhiteImg'+'_size'+ str(size[0]) + '_cal'
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
ax1.imshow(whiteimg0,cmap='inferno')
ax2.imshow(whiteimg,cmap='inferno')
plt.savefig(titleName+'.png',transparent = True,pad_inches = 0,bbox_inches='tight',dpi=mydpi) # 
plt.show()

# binarize
threshold = 0.2*np.amax(whiteimg) # threshold for binarizing. very important. need to be tuned well. Try 0.01, 0.015, 0.02;  
whiteimg[whiteimg<=threshold] = 0
whiteimg[whiteimg>threshold] = 1
plt.imshow(whiteimg, cmap='inferno')

initial_center = (np.array([whiteimg.shape[0]/2, whiteimg.shape[0]/2])).astype(int)
#initial_center = (np.array([255, 263])).astype(int)
#center0 = lf.find_center(whiteimg, spot_period,initial_center,
#                         kernel_type=0,quiet=False)
center0 = lf.find_center_robust(whiteimg, spot_period,initial_center,
                         kernel_type=2,quiet=False)

# show white image with the center
plt.figure(figsize=(10,10)); 
plt.imshow(whiteimg, cmap = 'gray')
plt.scatter(center0[1],center0[0], c='b', s=5)
plt.grid()

# find more centers
rad_spots = np.int(np.floor((size[0]/spot_period-1)/2)-1) #11 for size 513, # 5 for size 257
im_dim = 2*rad_spots+1
size_subimg = np.floor(spot_period).astype(int) # size of each subimg
centers = np.zeros((im_dim, im_dim,2));
#center0 = (np.array([whiteimg.shape[0]/2, whiteimg.shape[0]/2])).astype(int)
center_kl = center0
for k in np.arange(0, im_dim):
    for l in np.arange(0, im_dim):
        initial_center = (np.array([center0[0] + (k-rad_spots)*spot_period, 
                                   center0[1] + (l-rad_spots)*spot_period])).astype(int)
        center_kl = lf.find_center(whiteimg, spot_period,initial_center,
                                   kernel_type=0,quiet=True)
#        center_kl = lf.find_center_robust(whiteimg, spot_period,initial_center,
#                                   kernel_type=2,quiet=True)
        centers[k,l,:] = center_kl

centersFound = centers.copy();

# average each row and column
centersRowAve = np.around(np.array(np.mean(centers[:,:,0], axis = 1)))
centersColAve = np.around(np.array(np.mean(centers[:,:,1], axis = 0)))
centersRows = np.tile(centersRowAve,(centers.shape[1],1))
centersRows = centersRows.T
centersCols = np.tile(centersColAve,(centers.shape[0],1))
centers[:,:,0] = centersRows
centers[:,:,1] = centersCols

#centers[:,:,0] = centersRows - 1 # due to mismatch
#centers[:,:,1] = centersCols + 1

#%% show centers of microlens array.
plt.figure(figsize=(5,5)); 
plt.imshow(whiteimg, cmap='gray')
plt.scatter(centers[:,:,1].reshape(-1),centers[:,:,0].reshape(-1),c='b',s=1)
#plt.minorticks_on()
#plt.grid(b=True, which='both',color='b', linestyle='-', linewidth=0.5)
plt.savefig('./Img_Centers.png',transparent = True,pad_inches = 0,dpi=300) # ,bbox_inches='tight',
plt.show()

# save results
FileName = './Params_FuncLFM.npz'
np.savez(FileName,
         r = r,im_angle=im_angle, spot_period=spot_period,centers = centers,
         rad_spots = rad_spots,im_dim = im_dim,size_subimg = size_subimg,)

#%% show LF array with centers in each frame
show_fig = 1
if show_fig:
    fig, axes = plt.subplots(3,3,figsize=(10,10))
    fig.subplots_adjust(hspace = 0.3, wspace=0.3)
    axes = axes.ravel()
    for i in np.arange(0, 9 ):
        frameInd = i* (LF_cal_arr.shape[0]//9) # i*4 + 2
        axes[i].imshow(LF_cal_arr[frameInd,:,:],cmap='inferno', vmin=0); # ,aspect='auto'
        axes[i].scatter(centers[:,:,1].reshape(-1),centers[:,:,0].reshape(-1),c='b',s=0.2)
    #    axes[i].set_clim(0,np.amax(LF_arr))
    #    axes[i].grid(True)
        axes[i].set_title('LF_depth_'+str(frameInd*frameStep+depthStart)+'um')
        axes[i].set_xlabel('k axis')
        axes[i].set_ylabel('i axis')
    
    plt.savefig('./LF_cal_arr_centers'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
    plt.show() 
	

plt.figure()
plt.imshow(LF_cal_arr[0,:,:],cmap='inferno',vmin=0)
plt.scatter(centers[:,:,1].reshape(-1),centers[:,:,0].reshape(-1),c='b',s=0.2)
plt.savefig('./LF_cal_frame0_centers'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
plt.show() 
    
#%% ---------------------------------------------- 
#--Extract micro-images to construct 4D LFM data.


## read the LF data array
#LF_arr = tifffile.TiffFile('LF_cal_arr.tif').asarray()


radius = (np.floor(spot_period/2)).astype(int)
LF4D_arr = np.zeros((Num,size_subimg,size_subimg,im_dim,im_dim), dtype='int16') # [num,i,j,k,l]

save_views = 1

for frameInd in np.arange(Num):
   
    img = LF_cal_arr[frameInd,:,:]
#    img = img/np.amax(img)
    
    # using found centers to extract a patch corresponding to each lenslet
    for k in np.arange(0, im_dim):
        for l in np.arange(0, im_dim):
            center_kl = centers[k,l,:].astype(int)
            patch = img[center_kl[0]-radius:center_kl[0]+radius+1,
                           center_kl[1]-radius:center_kl[1]+radius+1]
            LF4D_arr[frameInd,:,:,k,l] = patch
    #        print(center_kl)
            
if save_views:
    np.save(r'./LF4D_arr',LF4D_arr)


#%% ----------------------------------------------
# verify that the splitted patches can be reorganized into the original image
verify_flag = 1 # 0: no verification; 1: verification   

for frameInd in np.arange(1): # np.arange(Num):
       
    views = LF4D_arr[frameInd,:,:,:,:]
    
    subimgs=[]
    imR = np.zeros((size_subimg,size_subimg,im_dim))
    ind = 0
    for k in range(views.shape[2]): # range(33, 53): # range(views.shape[2]):
        for l in range(views.shape[3]):
            patch = np.squeeze(views[:,:,k,l])
            imR[:,:,ind] = patch # np.expand_dims(patch,3)
            ind = ind+1
    
        ind = 0
        imR = imR.reshape((size_subimg,size_subimg*im_dim),order='F') # colum index first
        subimgs.append(imR)
    #    plt.imshow(imR, cmap='inferno');    plt.show()
        imR = np.zeros((size_subimg,size_subimg,im_dim))
    
    subimgsWhole = np.concatenate([subimg for subimg in subimgs],axis = 0)
    
    plt.figure(figsize = (3,3))
    plt.imshow(subimgsWhole, cmap='inferno', vmin=0)
#    plt.axis('off')
#    plt.minorticks_on()
#    plt.grid(b=True, which='both',color='b', linestyle='-', linewidth=0.5)
#    plt.clim(0,1)
    plt.show()
    

#%% ----------------------------------------------  
#--Convert 4D LFM data to multi-view / subaperture images.
#--Perform matrix factorization to separate the foreground and background.
#--Convert the foreground to a clean LF image.


views = LF4D_arr[-1,:,:,:,:]
background_ratio = 1
thresh = 0 # remove singular values smaller than thresh
# use the deepest frame to get the background
#Foreground, background, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For depth stack LFM, the largest singular value corresponds to the background.
background, Foreground, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For functional LFM, the largest singular value corresponds to the foreground.
show_fig = 1

for frameInd in np.arange(0-depthStart, Num, 100 ): # np.arange(Num-1, Num ):
   
    views = LF4D_arr[frameInd,:,:,:,:].copy()
    views_orig = views.copy()
    VMAX = np.amax(views)
    
    # image with background
    subimgs=[]
    imR = np.zeros((size_subimg,size_subimg,im_dim))
    ind = 0
    for k in range(views.shape[2]): # range(33, 53): # range(views.shape[2]):
        for l in range(views.shape[3]):
            patch = np.squeeze(views[:,:,k,l])
            imR[:,:,ind] = patch # np.expand_dims(patch,3)
            ind = ind+1
    
        ind = 0
        imR = imR.reshape((size_subimg,size_subimg*im_dim),order='F') # colum index first
        subimgs.append(imR)
        imR = np.zeros((size_subimg,size_subimg,im_dim))
    
    origImg = np.concatenate([subimg for subimg in subimgs],axis = 0)

    
    # image without background
    background, Foreground, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For functional LFM, the largest singular value corresponds to the foreground.
    views = views - background_ratio*background # remove background
    
    subimgs=[]
    imR = np.zeros((size_subimg,size_subimg,im_dim))
    ind = 0
    for k in range(views.shape[2]): # range(33, 53): # range(views.shape[2]):
        for l in range(views.shape[3]):
            patch = np.squeeze(views[:,:,k,l])
            imR[:,:,ind] = patch # np.expand_dims(patch,3)
            ind = ind+1
    
        ind = 0
        imR = imR.reshape((size_subimg,size_subimg*im_dim),order='F') # colum index first
        subimgs.append(imR)
        imR = np.zeros((size_subimg,size_subimg,im_dim))
    
    cleanImg = np.concatenate([subimg for subimg in subimgs],axis = 0)
    
    # background
    subimgs=[]
    imR = np.zeros((size_subimg,size_subimg,im_dim))
    ind = 0
    for k in range(background.shape[2]): 
        for l in range(background.shape[3]):
            patch = np.squeeze(background[:,:,k,l])
            imR[:,:,ind] = patch # np.expand_dims(patch,3)
            ind = ind+1
    
        ind = 0
        imR = imR.reshape((size_subimg,size_subimg*im_dim),order='F') # colum index first
        subimgs.append(imR)
        imR = np.zeros((size_subimg,size_subimg,im_dim))
    
    BG = np.concatenate([subimg for subimg in subimgs],axis = 0)  
    
    if show_fig:    
        fig, axes = plt.subplots(1,3,figsize=(15,45))
        fig.subplots_adjust(hspace = 0.3, wspace=0.3)
        axes = axes.ravel()
    
        axes[0].imshow(origImg, cmap = 'inferno', vmin = 0, vmax = VMAX) # , vmax = VMAX,aspect='auto'
        axes[1].imshow(cleanImg, cmap = 'inferno', vmin = 0, vmax = 0.8*VMAX)
        axes[2].imshow(BG, cmap = 'inferno', vmin = 0, vmax = VMAX)

        axes[0].set_title('Original')
        axes[1].set_title('Foreground')
        axes[2].set_title('Background')
        plt.savefig('./Fore_Background_Frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
        plt.show() 
    

        fig, axes = plt.subplots(11,11,figsize=(20,20)) # size_subimg
        ax_ind = 0
        fig.subplots_adjust(hspace = 0.1, wspace=0.1)
        i = 4
        for i_ind in np.arange(axes.shape[0]):
            j = 4
            for j_ind in np.arange(axes.shape[1]):
                axes[i_ind,j_ind].imshow(np.squeeze(views_orig[i,j,:,:]),cmap='inferno',vmin=0,vmax=VMAX) #,aspect='auto',origin='lower'
                axes[i_ind,j_ind].axis('off')
    #            titleName = 'i='+ '{:.0f}'.format(i)+ ', j='+'{:.0f}'.format(j)            
                j = j + 1
            i = i + 1
        titleName = 'Original' +'.png'
        plt.savefig('./Multi-view_Original_Frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
        plt.show()  

        fig, axes = plt.subplots(11,11,figsize=(20,20)) # size_subimg
        ax_ind = 0
        fig.subplots_adjust(hspace = 0.1, wspace=0.1)
        i = 4
        for i_ind in np.arange(axes.shape[0]):
            j = 4
            for j_ind in np.arange(axes.shape[1]):
                axes[i_ind,j_ind].imshow(np.squeeze(views[i,j,:,:]),cmap='inferno',vmin=0,vmax=VMAX) #,aspect='auto',origin='lower'
                axes[i_ind,j_ind].axis('off')
    #            titleName = 'i='+ '{:.0f}'.format(i)+ ', j='+'{:.0f}'.format(j)            
                j = j + 1
            i = i + 1
        titleName = 'Foreground' +'.png'
        plt.savefig('./Multi-view_Foreground_Frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
        plt.show()      
            
        fig, axes = plt.subplots(11,11,figsize=(20,20)) # size_subimg
        ax_ind = 0
        fig.subplots_adjust(hspace = 0.1, wspace=0.1)
        i = 4
        for i_ind in np.arange(axes.shape[0]):
            j = 4
            for j_ind in np.arange(axes.shape[1]):
                axes[i_ind,j_ind].imshow(np.squeeze(background[i,j,:,:]),cmap='inferno',vmin=0,vmax=VMAX) #,aspect='auto',origin='lower'
                axes[i_ind,j_ind].axis('off')
                j = j + 1
            i = i + 1
        titleName = 'Background' + '.png'
        plt.savefig('./Multi-view_Background_Frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
        plt.show()  
        

#%% ----------------------------------------------
#--Construct Epi-Polar Images (EPIs)
      
# Method 1: First, we detect the spatial positions (x,y), i.e. (k,l) on the central sub-aperture image first. Then we construct EPIs for each (x,y) position.

Thresh = 0.3 # 0.15 # find x,y positions with pixel value larger than this threshold.
source_num = 8 # 2  # maximum number of possible sources to be found in each subaperture image.
coords_list = [] 
## Alternatively, we can also manually specify the coordinates of the neurons in each frame
#coords_list = [(frameInd,np.array([[5,5]])) for frameInd in range(Num)] 

remove_BG = 0 # remove background or not
background_ratio = 1
thresh = 0 # remove singular values smaller than thresh
# use the deepest frame to get the background
views = LF4D_arr[-1,:,:,:,:]
#Foreground, background, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For depth stack LFM, the largest singular value corresponds to the background.


for frameInd in range( Num):
    views = LF4D_arr[frameInd,:,:,:,:].copy()  

    if remove_BG:
        background, Foreground, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For functional LFM, the largest singular value corresponds to the foreground.
        views = views - background_ratio*background  # remove background
	
    Vmax = views.max()
    
    i = np.int((size_subimg+1)/2-1)
    j = np.int((size_subimg+1)/2-1)
    central_SAI = np.squeeze(views[i,j,:,:]).copy()  # sub-aperture image
    
    source_index = 0
#    cell_area = np.array([[-1,-1], [-1,0], [-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]) # adjacent area to be neglected
    cell_area = np.array([[0,0]]) # adjacent area to be neglected
    
    coords_arr = [] # collect spatial coordinates (x,y) that has neurons
    
#	# find neurons according to thresholds
#    while (central_SAI.max() > Thresh*Vmax) & (source_index < source_num): 
#        coords = np.unravel_index(np.argmax(central_SAI),central_SAI.shape)
#        coords_arr.append(coords)
#               
#        zero_area = np.array(coords) + cell_area # compute neglected areas to avoid finding the same neuron more than once.
#        zero_area[zero_area>=im_dim] = im_dim-1
#        zero_area[zero_area<0] = 0
#        
#        central_SAI[zero_area[:,0], zero_area[:,1]] = 0
#        source_index = source_index + 1
#		
#    coords_arr = np.array(coords_arr) 
#    coords_list.append((frameInd,coords_arr))


    central_SAI = LF4D_arr[:,i,j,:,:].std(axis=0)
    # find neurons via perform 2D peak detection
    coords_arr = peak_local_max(central_SAI,
							 min_distance=3, threshold_rel=0.6,
							 num_peaks = 6) # threshold_abs=0.1,
    peaks2D_val = central_SAI[coords_arr[:,0],coords_arr[:,1]]
    sort_order = peaks2D_val.argsort()[::-1]
    peaks2D_val = peaks2D_val[sort_order]
    coords_arr = coords_arr[sort_order,:]
	coords_list.append((frameInd,coords_arr))

    if coords_arr.size > 0:
        plt.figure(figsize = (5,5))
        plt.imshow(central_SAI , cmap='inferno', vmin=0) #, vmax=0.7
        plt.scatter(coords_arr[:,1], coords_arr[:,0], s=50, c='b', marker='o')
        plt.grid(True)
        titleName = 'k-l image for frame '+str(frameInd)
        plt.title(titleName)
        plt.xlabel('l axis')
        plt.ylabel('k axis')
        plt.savefig(titleName+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
        plt.show()


# Method 2: Alternatively, we can also manually specify the coordinates of the neurons in each frame. Then we construct EPIs for each (x,y) position.
coords_list = [(frameInd,np.array([[im_dim//2,im_dim//2]])) for frameInd in range(Num)] 
frameInd = 0
coords_arr = coords_list[frameInd][1]

    
np.save(r'./coords_list',coords_list)
scipy.io.savemat('./coords_list'+'.mat', {'coords_list': coords_list})
print('Spatial positions (x,y) are detected on the central sub-aperture image.')

# Method 3: Alternatively, we can also load and reuse previous coordinates of the neurons detected in the STD image. Then we construct EPIs for each (x,y) position.
coords_list = np.load('coords_list.npy', allow_pickle=True)


# microscopy specifications
M=25   # 25            #magnification
d=125e-6               #pitch(m) 
Nlens = im_dim # 24; 12; 26; # even number. number of lenslets.
Npixel = 19 #21; # odd number. (Npixel x Npixel) spots per lenslet.
Nspot = 19*5 # 105; # odd number. (Nspot x Nspot) spots per lenslet.
nSamp = Nspot * Nlens # nSamp = L / dx; # or nSamp = Nspot * Nlens; number of samples
NlensVal = Nlens-1 # odd number. number of valid lenslet, always 1 less of Nlens.
dx = (d/M)/Nspot # distance bettwen two spots at the objective side.
oversamp_factor = Nspot/Npixel # oversampling on each pixel

# average x-y coordinates of neurons;
#xy_est_name = './coords_list.npy'
xy_est = coords_list # np.load(xy_est_name,allow_pickle=True)
xy_est_mean = [np.mean(xy_est[i][1],axis=0) for i in range(len(xy_est))]
xy_est_mean = np.array(xy_est_mean)
x_error = (xy_est_mean[:,0] - NlensVal//2)*(Npixel*dx*oversamp_factor)*1e6
y_error = (xy_est_mean[:,1] - NlensVal//2)*(Npixel*dx*oversamp_factor)*1e6
x_RMSE = np.sqrt(np.mean((x_error.reshape(-1))**2))
y_RMSE = np.sqrt(np.mean((y_error.reshape(-1))**2))
print('x_RMSE = {x_RMSE:.3f} um; y_RMSE = {y_RMSE:.3f} um;'
      .format(x_RMSE=x_RMSE, y_RMSE=y_RMSE))

fig, axes = plt.subplots(1,2, figsize=(6.5,3))
fig.subplots_adjust(hspace = 0.3, wspace=0.35)
axes[0].plot(x_error, 'ro', ms=6, 
    markerfacecolor="None", markeredgecolor='r', markeredgewidth=2, label='Detect')
axes[0].plot(np.zeros_like(x_error), 'b-', label='Truth')
axes[0].legend(loc='upper left', frameon=False) # ncol = 2
axes[0].set_xlabel('image index')
axes[0].set_ylabel('x estimation (um)')
axes[0].set_ylim([-20,20])
axes[0].set_yticks([-20,-10,0,10,20])
axes[0].grid()
axes[1].plot(y_error, 'ro', ms=6, 
    markerfacecolor="None", markeredgecolor='r', markeredgewidth=2, label='Detect')
axes[1].plot(np.zeros_like(y_error), 'b-', label='Truth')
axes[1].legend(loc='upper left', frameon=False) # ncol = 2
axes[1].set_xlabel('image index')
axes[1].set_ylabel('y estimation (um)')
axes[1].set_yticks([-20,-10,0,10,20])
axes[1].grid()
#plt.savefig('xy_estimation'+'.png',bbox_inches='tight',pad_inches = 0,dpi=200) # transparent = True,
plt.show()

#%% ----------------------------------------------
#--Construct Epi-Polar Images (EPIs) along j-l dimensions and i-k dimensions

remove_BG = 0 # 1: remove background; 0: do not remove background;

views = LF4D_arr[-1,:,:,:,:]
#Foreground, background, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For depth stack LFM, the largest singular value corresponds to the background.

                                                     
# ****************** construct j-l EPIs ****************** 
show_fig = 0 # show figures or not
save_EPI = 1 # save EPIs or not

EPI_jl_arr = np.empty([0,size_subimg,im_dim],dtype='int16')
Depth_jl_arr = np.empty([0,1])
for frameInd in np.arange(Num ): # np.arange(Num ):
    
    views = LF4D_arr[frameInd,:,:,:,:].copy()

    if remove_BG:
        background, Foreground, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For functional LFM, the largest singular value corresponds to the foreground.
        views = views - 1*background  # remove background
        views[views<0.1*views.max()] = 0
    
    source_index = 0
    #    coords_arr = coords_list[frameInd][1] # for depth stack LFM
    coords_arr = coords_list[0][1] # for functional LFM
    if coords_arr.size > 0:
        for k in np.unique(coords_arr[:,0]): # k = np.int((views.shape[2]+1)/2-1) #  middle row
    
            source_index = source_index + 1
            
            for i in (np.arange(0,1) + size_subimg//2): # (np.arange(-1,2) + np.int((views.shape[0]+1)/2-1)):  # np.arange(-2,3) # 9  
        #    i = np.int((size_subimg+1)/2-1) # for 19x19, the middel is 10
                
                EPI_jl = np.zeros((size_subimg,im_dim))
                for j in range(size_subimg):
                    patch = np.squeeze(views[i,j,k,:])
                    
                    # denoise patches using thresholding or deconvolution
        #            patch[patch < 0.5*views.max()] = 0 
                 
            #        # denoise pathces using BM3D
            #        patch0 = patch.copy()
            #        patch = patch.tolist()
            ##        patch = matlab.double(patch)
            #        eng = matlab.engine.start_matlab()
            #        patch = eng.BM3D_Video(patch)
                    
                    if patch.ndim > 1:
            #            EPI_jl[j,:] = patch.sum(axis=0)
                        EPI_jl[j,:] = patch.max(0) # max projection
                    else:
                        EPI_jl[j,:] = patch
                    
#                # normalization
#                EPI_jl[EPI_jl<0] = 0
#                EPI_jl = EPI_jl/(np.amax(EPI_jl)+1e-9)
                
                # thresholding
#                EPI_jl[ EPI_jl < 0.2*EPI_jl.max()] = 0
                         
                # construct EPI array
                EPI_jl = np.expand_dims(EPI_jl,0)
                EPI_jl_arr = np.concatenate((EPI_jl_arr,EPI_jl), axis=0)
                
                depth = np.array(frameInd*frameStep+depthStart).reshape(-1,1)
                Depth_jl_arr = np.concatenate((Depth_jl_arr, depth))
        
                if show_fig:
                    VMAX_EPI = EPI_jl.max()
                    plt.figure(figsize = (3,2))
                    plt.imshow(np.squeeze(EPI_jl), cmap='inferno',aspect='auto',origin='lower',vmin=0,vmax=VMAX_EPI)
                    plt.clim(0,np.amax(EPI_jl))
    #                plt.grid(True)
    #                titleName = 'j-l EPI at ' + 'i'+ '{:.0f}'.format(i) + '_k'+ '{:.0f}'.format(k)
                    FigName = 'EPI_jl'+ '_frame' + str(frameInd) +'_i'+str(i) + '_k'+ str(k)
                    plt.title(FigName)
                    plt.xlabel('l axis')
                    plt.ylabel('j axis')
#                    plt.savefig(FigName +'.png',bbox_inches='tight',pad_inches = 0,dpi=mydpi)
                    plt.show()
#                    # write image data
#                    img = (np.squeeze(EPI_jl)* (2**16-1)).astype(np.uint16)
#                    imageio.imwrite(FigName+'.tif', img)
#    #                img = imageio.imread(FigName+'.tif') # doublecheck the image dtype
#    #                print(img.dtype)


if save_EPI:
    scipy.io.savemat('./EPI_jl_arr' +'.mat',
                     {'EPI': EPI_jl_arr, 'Depths': Depth_jl_arr})
    np.savez('EPI_jl_arr.npz', EPI=EPI_jl_arr, Depths=Depth_jl_arr)

    
## show EPI jl
#if show_fig:
#    fig, axes = plt.subplots(3,3,figsize=(11,7))
#    fig.subplots_adjust(hspace = 0.5, wspace=0.3)
#    axes = axes.ravel()
#    for ii in np.arange(0, 9 ):
#        frameInd = ii* (EPI_jl_arr.shape[0]//9)
##        frameInd = 0 - depthStart + (ii-4)*10
#        VMAX_EPI = EPI_jl_arr[frameInd,:,:].max()
#        axes[ii].imshow(EPI_jl_arr[frameInd,:,:],cmap='inferno',aspect='auto',origin='lower',vmin=0,vmax=VMAX_EPI); 
#    #    axes[ii].grid(True)
##        axes[ii].set_title('depth='+str(Depth_jl_arr[frameInd].astype(int).item())+'um')
#        axes[ii].set_title('frame ' + str(frameInd))
#        axes[ii].set_xlabel('l axis')
#        axes[ii].set_ylabel('j axis')
#    
#    plt.savefig('./EPI_jl_arr' +'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
#    plt.show()
            
print('EPI j-l are generated!')


#%%    
# ****************** construct i-k EPIs ****************** 

show_fig = 0 # show figures or not
save_EPI = 1 # save EPIs or not


EPI_ik_arr = np.empty([0,size_subimg,im_dim],dtype='int16')
Depth_ik_arr = np.empty([0,1])

for frameInd in np.arange(Num ): # np.arange(Num ):
    
    views = LF4D_arr[frameInd,:,:,:,:].copy()

    if remove_BG:
        background, Foreground, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For functional LFM, the largest singular value corresponds to the foreground.
        views = views - 1*background  # remove background 
        views[views<0.1*views.max()] = 0
    
    source_index = 0
    #    coords_arr = coords_list[frameInd][1] # for depth stack LFM
    coords_arr = coords_list[0][1] # for functional LFM
    if coords_arr.size > 0:
        for l in np.unique(coords_arr[:,1]): # l = np.int((views.shape[3]+1)/2-1) #  middle col
    
            source_index = source_index + 1
            
            for j in (np.arange(0,1) + size_subimg//2): # (np.arange(-1,2) + np.int((views.shape[0]+1)/2-1)):  # np.arange(-2,3) # 9  
              
                EPI_ik = np.zeros((size_subimg,im_dim))
                for i in range(size_subimg):
                    patch = np.squeeze(views[i,j,:,l])
                    
                    if patch.ndim > 1:
            #            EPI_ik[i,:] = patch.sum(axis=0)
                        EPI_ik[i,:] = patch.max(0) # max projection
                    else:
                        EPI_ik[i,:] = patch
                    
#                # normalization
#                EPI_ik[EPI_ik<0] = 0
#                EPI_ik = EPI_ik/(np.amax(EPI_ik)+1e-9)
                
                # thresholding
#                EPI_ik[ EPI_ik < 0.2*EPI_ik.max()] = 0
                         
                # construct EPI array
                EPI_ik = np.expand_dims(EPI_ik,0)
                EPI_ik_arr = np.concatenate((EPI_ik_arr,EPI_ik), axis=0)
                
                depth = np.array(frameInd*frameStep+depthStart).reshape(-1,1)
                Depth_ik_arr = np.concatenate((Depth_ik_arr, depth))
        
                if show_fig:
                    VMAX_EPI = EPI_ik.max()
                    plt.figure(figsize = (3,2))
                    plt.imshow(np.squeeze(EPI_ik), cmap='inferno',aspect='auto',origin='lower',vmin=0,vmax=VMAX_EPI)
                    plt.clim(0,np.amax(EPI_ik))
    #                plt.grid(True)
                    FigName = 'EPI_ik'+ '_frame' + str(frameInd) +'_j'+str(j) + '_l'+ str(l)
                    plt.title(FigName)
                    plt.xlabel('k axis')
                    plt.ylabel('i axis')
#                    plt.savefig(FigName +'.png',bbox_inches='tight',pad_inches = 0,dpi=mydpi)
                    plt.show()
#                    # write image data
#                    img = (np.squeeze(EPI_ik)* (2**16-1)).astype(np.uint16)
#                    imageio.imwrite(FigName+'.tif', img)
#    #                img = imageio.imread(FigName+'.tif') # doublecheck the image dtype
#    #                print(img.dtype)


if save_EPI:
    scipy.io.savemat('./EPI_ik_arr' +'.mat',
                     {'EPI': EPI_ik_arr, 'Depths': Depth_ik_arr})
    np.savez('EPI_ik_arr.npz', EPI=EPI_ik_arr, Depths=Depth_ik_arr)

    
## show EPI ik
#if show_fig:
#	fig, axes = plt.subplots(3,3,figsize=(11,7))
#	fig.subplots_adjust(hspace = 0.5, wspace=0.3)
#	axes = axes.ravel()
#	for ii in np.arange(0, 9 ):
#		frameInd = ii*(EPI_ik_arr.shape[0]//9)
#		VMAX_EPI = EPI_ik_arr[frameInd,:,:].max()
#		axes[ii].imshow(EPI_ik_arr[frameInd,:,:],
#	  cmap='inferno',aspect='auto',origin='lower',vmin=0,vmax=VMAX_EPI)
#		#axes[ii].grid(True)
#		#axes[ii].set_title('depth='+str(Depth_ik_arr[frameInd].astype(int).item())+'um')
#		axes[ii].set_title('frame ' + str(frameInd))
#		axes[ii].set_xlabel('l axis')
#		axes[ii].set_ylabel('j axis')
#    
#	plt.savefig('./EPI_ik_arr' +'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
#	plt.show()
            
print('EPI i-k are generated!')


#%% extract temporal sequence (time series) as neural activity from EPI arrays.
for frameInd in range( Num):
	views = LF4D_arr[frameInd,:,:,:,:].copy()  
	if remove_BG:
		background, Foreground, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For functional LFM, the largest singular value corresponds to the foreground.
		views = views - background_ratio*background  # remove background
		
coords_arr = coords_list[0][1] # for functional LFM
seqs = []
for i in range(coords_arr.shape[0]):
	seq = LF4D_arr[:,size_subimg//2,size_subimg//2,coords_arr[i,0],coords_arr[i,1]]
	seqs.append(seq) 
	
seqs = np.array(seqs)


#EPI_ik = np.load('./EPI_ik_arr.npz')
#EPI_ik_arr = EPI_ik['EPI']
#Depth_ik_arr = EPI_ik['Depths']

EPI_std = EPI_ik_arr.std(axis=0)
EPI_arr = EPI_ik_arr

#EPI_std = EPI_jl_arr.std(axis=0)
#EPI_arr = EPI_jl_arr

# Perform thresholding to obtain a mask
mask = EPI_std>0.5*EPI_std.max()
# Given the mask, perform morphological operations such as open, close to refine the mask.

# Then, find connected areas.
mask_labels, num_labels = measure.label(mask, background=0, return_num=True) #  connectivity=None: ranging from 1 to input.ndim. If None, a full connectivity of input.ndim is used.

clusters = []
for i in range(1, num_labels+1):
	clusters.append(np.argwhere(mask_labels == i))

seqs = []
for i in range(num_labels):
	seq = EPI_arr[:, clusters[i][:,0], clusters[i][:,1]].mean(axis=1)
	seq2 = []
	for j in range(0, EPI_arr.shape[0], EPI_arr.shape[0]//Num):
		seq2.append( seq[j:j+EPI_arr.shape[0]//Num].mean() )
		
	seqs.append( np.asarray(seq2) )
	


plt.figure(figsize=(5, 4))
plt.subplot(121)
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(mask_labels, cmap='nipy_spectral')
plt.axis('off')
plt.tight_layout()
plt.show()


# show sequences
#clusters = [[0,1,5,6],[2,3,4,7]]
#fig=plt.figure(figsize=(5,5))
#fig.subplots_adjust(hspace = 0.3, wspace=0.3)
#plt.subplot(2,1,1)
#plt.plot(seqs[clusters[0],:].mean(axis=0), 'b-')
#plt.title('neuron 1')
#plt.subplot(2,1,2)
#plt.plot(seqs[clusters[1],:].mean(axis=0), 'r-')
#plt.title('neuron 2')

fig=plt.figure(figsize=(5,5))
fig.subplots_adjust(hspace = 0.3, wspace=0.3)
plt.subplot(2,1,1)
plt.plot(seqs[0], 'b-')
plt.title('neuron 1')
plt.subplot(2,1,2)
plt.plot(seqs[1], 'r-')
plt.title('neuron 2')

FigName = 'NeuralSeq_EPI_ik.png'
plt.savefig(FigName, transparent = True, bbox_inches='tight',pad_inches = 0,dpi=200)
plt.show()


#%% show the subaperture image and EPI simultaneously at the center i and j.
mydpi = 300
show_fig = 1 

views = LF4D_arr[0,:,:,:,:]
central_views = np.zeros((Num, views.shape[2],views.shape[3]), dtype='int16')

for frameInd in np.arange(0-depthStart, Num, 10 ): # 0-depthStart

    views = LF4D_arr[frameInd,:,:,:,:]
    
    # remove background
    if remove_BG:       
#        background_ratio = 1 # 0.1*(frameInd+1)
#        thresh = 0 #0.6
#        views, background, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For depth stack LFM, the largest singular value corresponds to the background.
#        views = views - 1*background  # remove background 
#        views[views < 0.1*views.max()] = 0 # thresholding based on max value
        background, views, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background. For functional LFM, the largest singular value corresponds to the foreground.
        views[views < 0.7*views.max()] = 0 # thresholding based on max value

    i = views.shape[0]//2 # for 19x19, the middel is 9 as index starts from 0
    j = views.shape[1]//2 
	
    patch = np.squeeze(views[i,j,:,:])
    central_views[frameInd,:,:] = patch
    
    # denoise patches using thresholding or deconvolution
#    patch[patch < 0.4*patch.max()] = 0 


if show_fig:    
    plt.figure(figsize=(6,6))
    grid = plt.GridSpec(4, 4, wspace=0.2, hspace=0.2)
    VMAX_EPI = central_views.std(axis=0).max()
	 
	# draw lower sub-figure
    main_ax = plt.subplot(grid[1:4,0:3]) # plt.subplot(grid[0:3,1:4])
    plt.imshow(central_views.std(axis=0),cmap='inferno',vmin=0,vmax=VMAX_EPI) 
#    main_ax.axis('off')
    plt.xlabel('l axis')
    plt.ylabel('k axis')
    
    depth = frameInd*frameStep+depthStart
    frameInd_EPI = np.argwhere(Depth_ik_arr == depth)
    frameInd_EPI = frameInd_EPI[0,0]
    
	# draw right sub-figure
    y_hist = plt.subplot(grid[1:4,3],yticklabels=[])
#    VMAX_EPI = EPI_ik_arr[frameInd_EPI,:,:].max()
#    plt.imshow(EPI_ik_arr[frameInd_EPI,:,:].T,cmap='inferno',aspect='auto',origin='lower',vmin=0,vmax=VMAX_EPI); # ,origin='lower'
    EPI = np.std(EPI_ik_arr,axis=0)
#    EPI[EPI<0.2*EPI.max()] = 0
    plt.imshow(EPI.T,cmap='inferno',aspect='auto',origin='lower',vmin=0,vmax=None) # ,origin='lower'
#    plt.imshow(y,60,orientation='horizontal',color='gray')#图形水平绘制
    plt.xlabel('i axis')
#    y_hist.axis('off')
    y_hist.invert_yaxis()#x轴调换方向
    
    depth = frameInd*frameStep+depthStart
    frameInd_EPI = np.argwhere(Depth_jl_arr == depth)
    frameInd_EPI = frameInd_EPI[0,0]
    
	# draw upper sub-figure
    x_hist = plt.subplot(grid[0,0:3],xticklabels=[])#和大子图共x轴, sharex=main_ax
#    VMAX_EPI = EPI_jl_arr[frameInd_EPI,:,:].max()
#    plt.imshow(EPI_jl_arr[frameInd_EPI,:,:],cmap='inferno',aspect='auto',origin='lower',vmin=0,vmax=VMAX_EPI)
    EPI = np.std(EPI_jl_arr,axis=0)
#    EPI[EPI<0.2*EPI.max()] = 0
    plt.imshow(EPI,cmap='inferno',aspect='auto',origin='lower',vmin=0,vmax=None) 
#    x_hist.axis('off')
    plt.ylabel('j axis')

    plt.savefig('Central_SubapertureImg_EPI_frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
    plt.show()


#%% load EPI array

titleName = 'EPI_ik_arr'
#EPI_path = r'./'+titleName+'.mat' 
#temp = scipy.io.loadmat(EPI_path) 
EPI_path = r'./'+titleName+'.npz' 
temp = np.load(EPI_path) 
EPI_ik_arr = temp['EPI']
Depth_ik_arr = temp['Depths']


titleName = 'EPI_jl_arr'
#EPI_path = r'./'+titleName+'.mat' 
#temp = scipy.io.loadmat(EPI_path) 
EPI_path = r'./'+titleName+'.npz' 
temp = np.load(EPI_path) 
EPI_jl_arr = temp['EPI']
Depth_jl_arr = temp['Depths']

## search and load specified EPIs
#FigName = 'EPI_ik'+ '_frame*' +'_j'+str(j) + '_l'+ str(l)
#EPI_path = glob.glob(r'./EPI_data/'+FigName+'.mat')
#for name in EPI_path:
#    print('\t', name)
#EPI_ik_arr = scipy.io.loadmat(EPI_path[0]) #
#EPI_ik_arr = EPI_ik_arr['EPI_ik_arr']    


## Try to remove background of EPI array via svd. But does not work well.
#background_ratio = 1
#thresh = 0
#EPI_ik_arr, background, S, S1, Sb = lf.RM_Background_EPI(EPI_ik_arr, background_ratio, thresh, show_fig=0) # remove background
    


#%% ----------------------------------------------
#--Construct Epi-Polar Images (EPIs) only from specified k-l positions
        
show_fig = 0
save_EPI = 1
remove_BG = 1 # 1: remove background; 0: do not remove background;
mydpi = 300

#EPI_ik_arr = np.zeros((Num,size_subimg,im_dim))
#EPI_jl_arr = np.zeros((Num,size_subimg,im_dim))

EPI_jl_arr = np.empty([0,size_subimg,im_dim])
Depth_jl_arr = np.empty([0,1])        
            
EPI_ik_arr = np.empty([0,size_subimg,im_dim])
Depth_ik_arr = np.empty([0,1])

ind_ik = 0
ind_jl = 0


views = LF4D_arr[-1,:,:,:,:]
background_ratio = 1
thresh = 0 # remove singular values smaller than thresh
Foreground, background, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background


for frameInd in np.arange(Num ): # np.arange(Num ):
   
    views = LF4D_arr[frameInd,:,:,:,:]
    
    if remove_BG:          
        views = views - 1*background # remove background
        

    # ****************** construct j-l EPIs ****************** 
    i = np.int((views.shape[0]+1)/2-1) # for 19x19, the middel is 10
    k = np.int((views.shape[2]+1)/2-1) # single middle row
    k = np.arange(k,k+1) # selected rows
    
    ii = i
    kk = k[0]
    
    EPI_jl = np.zeros((size_subimg,im_dim))
    for j in range(size_subimg):
        patch = np.squeeze(views[i,j,k,:])
        
            # denoise patches using thresholding or deconvolution
#            patch[patch < 0.5*views.max()] = 0 
         
    #        # denoise pathces using BM3D
    #        patch0 = patch.copy()
    #        patch = patch.tolist()
    ##        patch = matlab.double(patch)
    #        eng = matlab.engine.start_matlab()
    #        patch = eng.BM3D_Video(patch)        
        
        
        if patch.ndim > 1:
            EPI_jl[j,:] = patch.sum(axis=0)
        else:
            EPI_jl[j,:] = patch
        
#    # normalization
#    EPI_jl = EPI_jl/(np.amax(EPI_jl)+1e-9)
    
    # construct EPI array
    EPI_jl = np.expand_dims(EPI_jl,0)
    EPI_jl_arr = np.concatenate((EPI_jl_arr,EPI_jl), axis=0)           
#    EPI_jl_arr[frameInd,:,:]=EPI_jl
    
    depth = np.array(frameInd*frameStep+depthStart).reshape(-1,1)
    Depth_jl_arr = np.concatenate((Depth_jl_arr, depth))

    if show_fig:
        plt.figure(figsize = (3,2))
        plt.imshow(np.squeeze(EPI_jl), cmap='inferno',aspect='auto',origin='lower')
        plt.clim(0,np.amax(EPI_jl))
        plt.grid(True)
        titleName = 'j-l EPI at ' + 'i='+ '{:.0f}'.format(i)
        plt.title(titleName)
        plt.xlabel('l axis')
        plt.ylabel('j axis')
#        plt.savefig('EPI_jl_frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
    
    
    # ****************** construct i-k EPIs ****************** 
    j = np.int((views.shape[1]+1)/2-1) # for 19x19, the middel is 10
    l = np.int((views.shape[3]+1)/2-1) # single middle row
    l = np.arange(l,l+1) # selected rows
    
    jj = j
    ll = l[0]
    
    EPI_ik = np.zeros((size_subimg,im_dim))
    for i in range(size_subimg):
        patch = np.squeeze(views[i,j,:,l])
            
        # denoise patches using thresholding or deconvolution
        #patch[patch < 0.3*views.max()] = 0 
        
#        # denoise pathces using BM3D
#        patch0 = patch.copy
#        patch = patch.tolist()
#        eng = matlab.engine.start_matlab()
#        patch = eng.BM3D_Video(patch)
    
        if patch.ndim > 1:
            EPI_ik[i,:] = patch.sum(axis=0)
        else:
            EPI_ik[i,:] = patch
   
#    # normalization
#    EPI_ik = EPI_ik/np.amax(EPI_ik)
    
    # construct EPI array
    EPI_ik = np.expand_dims(EPI_ik,0)
    EPI_ik_arr = np.concatenate((EPI_ik_arr,EPI_ik), axis=0)
#    EPI_ik_arr[frameInd,:,:]= EPI_ik
    
    depth = np.array(frameInd*frameStep+depthStart).reshape(-1,1)
    Depth_ik_arr = np.concatenate((Depth_ik_arr, depth))

           
    if show_fig:
        plt.figure(figsize = (3,2))
        plt.imshow(np.squeeze(EPI_ik), cmap='inferno',aspect='auto',origin='lower')
        plt.clim(0,np.amax(EPI_ik))
        plt.grid(True)
        titleName = 'i-k EPI at ' + 'j='+ '{:.0f}'.format(j)
        plt.title(titleName)
        plt.xlabel('k axis')
        plt.ylabel('i axis') 
#        plt.savefig('EPI_ik_frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
    
    print('EPI i-k and j-l are generated!')


EPI_ik_arr[EPI_ik_arr<0] = 0;
EPI_jl_arr[EPI_jl_arr<0] = 0;


if save_EPI:
    scipy.io.savemat('./EPI_ik_arr_at_j'+str(jj)+'_l'+str(ll)+'.mat', 
                     {'EPI_ik_arr': EPI_ik_arr, 'Depths': Depth_ik_arr})
    scipy.io.savemat('./EPI_jl_arr_at_i'+str(ii)+'_k'+str(kk)+'.mat', 
                     {'EPI_jl_arr': EPI_jl_arr, 'Depths': Depth_jl_arr})
    
    np.savez('./EPI_ik_arr_at_j'+str(jj)+'_l'+str(ll), 
             EPI=EPI_ik_arr, Depths=Depth_ik_arr)
    np.savez('./EPI_jl_arr_at_i'+str(ii)+'_k'+str(kk), 
             EPI=EPI_jl_arr, Depths=Depth_jl_arr)  


#%% ----------------------------------------------  
#--Convert 4D LFM data to multi-view / subaperture images.

# show and save subaperture images for a specific i and j
mydpi = 300
show_fig = 1 

for frameInd in np.arange(Num-1, Num ): # np.arange(9, 10 )

    views = LF4D_arr[frameInd,:,:,:,:]

#    i = np.int((views.shape[0]+1)/2-1) # for 19x19, the middel is 10
    i = np.int((views.shape[0]+1)/2-1) - 1 # for 19x19, the middel is 10
    
    fig, axes = plt.subplots(4,5,figsize=(10,10)) # size_subimg
    ax_ind = 0
    VMAX_EPI = np.amax(views)
#    fig.subplots_adjust(hspace = 0.5, wspace=0.2)
    axes = axes.ravel()
    for j in np.arange(0, size_subimg ):
        axes[ax_ind].imshow(np.squeeze(views[i,j,:,:]),cmap='inferno',vmin=0,vmax=VMAX_EPI) #,aspect='auto',origin='lower'
#        axes[i].set_clim(0,0.1)
        axes[ax_ind].grid(True)
        titleName = 'i='+ '{:.0f}'.format(i)+ ', j='+'{:.0f}'.format(j)
        axes[ax_ind].set_title(titleName)
        ax_ind = ax_ind + 1
    
    plt.savefig('SubapertureImgs_horizontal_i'+str(i)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
    plt.show()
    

#    j = np.int((views.shape[1]+1)/2-1) # for 19x19, the middel is 10
    j = np.int((views.shape[1]+1)/2-1) - 1
    
    fig, axes = plt.subplots(4,5,figsize=(10,10)) # size_subimg
    ax_ind = 0
#    fig.subplots_adjust(hspace = 0.5, wspace=0.2)
    axes = axes.ravel()
    for i in np.arange(0, size_subimg ):
        axes[ax_ind].imshow(np.squeeze(views[i,j,:,:]),cmap='inferno',vmin=0,vmax=VMAX_EPI) #,aspect='auto',origin='lower'
#        axes[i].set_clim(0,0.1)
        axes[ax_ind].grid(True)
        titleName = 'i='+ '{:.0f}'.format(i)+ ', j='+'{:.0f}'.format(j)
        axes[ax_ind].set_title(titleName)
        ax_ind = ax_ind + 1
    
    plt.savefig('SubapertureImgs_vertical_j'+str(j)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
    plt.show()   
    
    
    




