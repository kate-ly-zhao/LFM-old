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


Original data is from: 
[path]/Chronos_Test/chronos_190301/MLA_1x1_f_2-8_50ms_490nm_1
Now the data is copied to:
[path]/LFMdata/LFimg.tif


References：
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
import f.general_functions as gf

#import matlab.engine
from numpy.linalg import svd


#%% ---------------------------------------------- 
#--Detect the rotation angle of a raw LFM image.
#--Detect lenslet pitch.

# find rotation angle, lenslet pitch using a non-focused LF image as the white image.
WhiteImg_path = r'./LFMdata/WhiteImg.tif' # no-focused LFM image of size 2048x2048
whiteimg = tifffile.imread(WhiteImg_path)

# we recommend to use the whole image for estimating rotation, spot_period, etc., as it is more accurate.
# If using the whole image, we get 
#(r, im_angle, spot_period) = [-0.45432528 19.68706608] -1.3220 19.6923
crop_flag = 0 # 1: crop an area; 0: no cropping
if crop_flag > 0:
    center = np.array([1756,732])
    size = np.array([513,513]) # np.array([257,257]) # np.array([513,513])
    roi = np.array([[center[0]-np.floor(size[0]/2), center[0]+np.floor(size[0]/2)], 
                     [center[1]-np.floor(size[1]/2), center[1]+np.floor(size[1]/2)]]) # region of interest
    roi = roi.astype(int)
    whiteimg = whiteimg[roi[0,0]:roi[0,1]+1,roi[1,0]:roi[1,1]+1]
    whiteimg = whiteimg/np.amax(whiteimg)
    print('using cropped image for estimating rotation, etc.')
else:
    whiteimg = whiteimg/np.amax(whiteimg)
    print('using whole image for estimating rotation, etc.')

r, im_angle, spot_period = lf.find_angle(whiteimg)
print(r, im_angle, spot_period)

FileName = './Params_LFM2D.npz'
np.savez(FileName,r = r,im_angle=im_angle, spot_period=spot_period)



#%% ---------------------------------------------- 
#--Crop an region of interest (ROI) from each LFM frame.
    
LF_path = r'./LFMdata/LFimg.tif'  # change the data path according to your situation.
center = np.array([1756,732])
size = np.array([513,513]) # np.array([257,257]), np.array([513,513])
roi = np.array([[center[0]-np.floor(size[0]/2), center[0]+np.floor(size[0]/2)], 
                 [center[1]-np.floor(size[1]/2), center[1]+np.floor(size[1]/2)]]) # region of interest
roi = roi.astype(int)

# extract frame 19 - frame 55, i.e. index is 18 - 54
frameStart = 18 # 18
frameEnd = 55 # 55
frameStep = 4 # 4

# load LFM image and crop ROIs
stack = tifffile.imread(LF_path) # 3D data with a format as XYZ
Num = int((frameEnd-1-frameStart)/frameStep + 1)
LF_arr = np.zeros((Num,size[0],size[1])) #  a set of LF images
LF_cal_arr = np.zeros((Num,size[0],size[1])) # a set of calibrated LF images
ind = 0

show_fig = 1

for frameInd in np.arange(frameStart, frameEnd, frameStep ):
    img = copy.copy(stack[frameInd,roi[0,0]:roi[0,1]+1,roi[1,0]:roi[1,1]+1])
    LF_arr[ind,:,:] = img
    
    maxval = np.amax(img)
    img = skimage.transform.rotate(img,im_angle)
    img = (img/np.amax(img))*maxval
    img = img.astype('uint16')
    LF_cal_arr[ind,:,:] = img
     
    if show_fig:
        
        # show original and rotated ROI together
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,3))
        ax1.imshow(LF_arr[ind,:,:],cmap='hot'); ax1.set_title('original')
        ax2.imshow(LF_cal_arr[ind,:,:],cmap='hot'); ax2.set_title('rotated')
        ax1.axis('off'); ax2.axis('off')
#        titleName = './frame'+str(frameInd+1)+'_size'+ str(size[0]) + '_orig_rot.png'
#        plt.savefig(titleName,transparent = True,pad_inches = 0,bbox_inches='tight',dpi=300)
        plt.show()
        
    ind = ind + 1

LF_arr = LF_arr.astype('uint16')
LF_cal_arr = LF_cal_arr.astype('uint16')
# write the LF data in .tif image
with tifffile.TiffWriter('./LF_arr.tif', bigtiff=True) as tif:
    for i in range(LF_arr.shape[0]):
        tif.save(LF_arr[i,:,:])
        
with tifffile.TiffWriter('./LF_cal_arr.tif', bigtiff=True) as tif:
    for i in range(LF_cal_arr.shape[0]):
        tif.save(LF_cal_arr[i,:,:])

## read the LF data array
#LF_cal_arr = tifffile.TiffFile('LF_cal_arr.tif').asarray()

# show LF array
fig, axes = plt.subplots(3,3,figsize=(10,10))
fig.subplots_adjust(hspace = 0.3, wspace=0.3)
axes = axes.ravel()
VMAX_LF = np.amax(LF_cal_arr)
for i in np.arange(0, 9 ):
    k = i
    axes[i].imshow(LF_cal_arr[k,:,:],cmap='hot',vmin=0,vmax=VMAX_LF); # ,aspect='auto'
#    axes[i].grid(True)
    axes[i].set_title('LF_depth_'+str(k*frameStep)+'um')
    axes[i].set_xlabel('k axis')
    axes[i].set_ylabel('i axis')

plt.savefig('./LF_cal_arr'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
plt.show()

# show LF array in a row
fig, axes = plt.subplots(1,6,figsize=(20,10))
fig.subplots_adjust(hspace = 0.1, wspace=0.1)
axes = axes.ravel()
VMAX_LF = np.amax(LF_cal_arr)
for i in np.arange(0, 6 ):
    k = i
    axes[i].imshow(LF_cal_arr[k,:,:],cmap='hot',vmin=0,vmax=VMAX_LF); # ,aspect='auto'
#    axes[i].grid(True)
    axes[i].axis('off')
    axes[i].set_title('LF_depth_'+str(k*frameStep)+'um')
    axes[i].set_xlabel('l axis')
    axes[i].set_ylabel('k axis')

plt.savefig('./LF_cal_arr_row'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
plt.show()


#%% ---------------------------------------------- 
#--Detect the centers of microlens array using an out-of-focus LFM image, instead of white image.

# load and crop a white image
whiteimg = tifffile.imread(WhiteImg_path)
whiteimg = copy.copy(whiteimg[roi[0,0]:roi[0,1]+1,roi[1,0]:roi[1,1]+1])
whiteimg0 = np.copy(whiteimg)

titleName = './WhiteImg'+'_size'+ str(size[0])
tifffile.imsave(titleName+'.tif', whiteimg)
mydpi = 300
plt.figure(figsize = (3,3))
plt.imshow(whiteimg, cmap='hot')
plt.axis('off')
#plt.minorticks_on()
#plt.grid(b=True, which='both',color='b', linestyle='-', linewidth=0.5)
plt.savefig(titleName+'.png',transparent = True,pad_inches = 0,bbox_inches='tight',dpi=mydpi)
plt.show()


# find the centers of microlens array via convolution with a template.

#whiteimg = tifffile.imread(titleName)
whiteimg = whiteimg0.copy()
whiteimg = whiteimg/np.amax(whiteimg)
whiteimg = skimage.transform.rotate(whiteimg,im_angle)
whiteimg = whiteimg/np.amax(whiteimg)

titleName = './WhiteImg'+'_size'+ str(size[0]) + '_cal'
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
ax1.imshow(whiteimg0,cmap='hot')
ax2.imshow(whiteimg,cmap='hot')
plt.savefig(titleName+'.png',transparent = True,pad_inches = 0,bbox_inches='tight',dpi=mydpi) # 
plt.show()

# binarize
threshold = 0.04*np.amax(whiteimg) #  for LFM image frame
whiteimg[whiteimg<=threshold] = 0
whiteimg[whiteimg>threshold] = 1

initial_center = (np.array([whiteimg.shape[0]/2, whiteimg.shape[0]/2])).astype(int)
#initial_center = (np.array([128, 128])).astype(int)
#center0 = lf.find_center(whiteimg, spot_period,initial_center,
#                         kernel_type=0,quiet=False)
center0 = lf.find_center_robust(whiteimg, spot_period,initial_center,
                         kernel_type=2,quiet=False)

# show white image with the center
dis=120 # 9 # 123
whiteimg_crop = whiteimg[center0[0]-dis:center0[0]+dis,
                       center0[1]-dis:center0[1]+dis]
plt.figure; plt.imshow(whiteimg_crop, cmap = 'gray')
plt.scatter(dis,dis, c='b', s=2)
plt.grid()

# find more centers
rad_spots = np.int(np.floor((size[0]/spot_period-1)/2)-1) #11 for size 513, # 5 for size 257
im_dim = 2*rad_spots+1
size_subimg = np.floor(spot_period).astype(int) # size of each subimg
centers = np.zeros((im_dim, im_dim,2));
center_kl = center0
for k in np.arange(0, im_dim):
    for l in np.arange(0, im_dim):
        initial_center = (np.array([center0[0] + (k-rad_spots)*spot_period, 
                                   center0[1] + (l-rad_spots)*spot_period])).astype(int)
#        center_kl = lf.find_center(whiteimg, spot_period,initial_center,
#                                   kernel_type=0,quiet=True)
        center_kl = lf.find_center_robust(whiteimg, spot_period,initial_center,
                                   kernel_type=2,quiet=True)
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


plt.figure(figsize=(10,10)); 
plt.imshow(whiteimg, cmap='gray')
plt.scatter(centers[:,:,1].reshape(-1),centers[:,:,0].reshape(-1),c='b',s=1)
#plt.minorticks_on()
#plt.grid(b=True, which='both',color='b', linestyle='-', linewidth=0.5)
plt.savefig('./Img_Centers.png',transparent = True,pad_inches = 0,dpi=300) # ,bbox_inches='tight',
plt.show()

# save results
FileName = './Params_LFM2D.npz'
np.savez(FileName,
         r = r,im_angle=im_angle, spot_period=spot_period,centers = centers,
         rad_spots = rad_spots,im_dim = im_dim,size_subimg = size_subimg,)

# show LF array with centers in each frame
show_fig = 1
if show_fig:
    fig, axes = plt.subplots(3,3,figsize=(10,10))
    fig.subplots_adjust(hspace = 0.3, wspace=0.3)
    axes = axes.ravel()
    for i in np.arange(0, 9 ):
        axes[i].imshow(LF_cal_arr[i,:,:],cmap='hot'); # ,aspect='auto'
        axes[i].scatter(centers[:,:,1].reshape(-1),centers[:,:,0].reshape(-1),c='b',s=1)
    #    axes[i].set_clim(0,np.amax(LF_arr))
    #    axes[i].grid(True)
        axes[i].set_title('LF_depth_'+str(i*4)+'um')
        axes[i].set_xlabel('k axis')
        axes[i].set_ylabel('i axis')
    
    plt.savefig('./LF_cal_arr_centers'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
    plt.show() 
    
 
#%% ---------------------------------------------- 
#--Extract micro-images to construct 4D LFM data.


## read the LF data array
#LF_arr = tifffile.TiffFile('LF_cal_arr.tif').asarray()

radius = (np.floor(spot_period/2)).astype(int)
LF4D_arr = np.zeros((Num,size_subimg,size_subimg,im_dim,im_dim)) # [num,i,j,k,l]

save_views = 1

for frameInd in np.arange(Num):
   
    img = LF_cal_arr[frameInd,:,:]
    img = img/np.amax(img)
    
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
#--Convert 4D LFM data to multi-view / subaperture images.
#--Perform matrix factorization to separate the foreground and background.
#--Convert the foreground to a clean LF image.


views = LF4D_arr[-1,:,:,:,:]
background_ratio = 1
thresh = 0.6
# use the deepest frame to get the background
Foreground, background, S, S1, Sb = lf.RM_Background(views, background_ratio, thresh, show_fig=0) # remove background
show_fig = 1

for frameInd in np.arange( Num ): # np.arange(Num ):
   
    views = LF4D_arr[frameInd,:,:,:,:]
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
    
    background_ratio = 1
    thresh = 0.6
    views = views - 1*background # remove background
    
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
    
        axes[0].imshow(origImg, cmap = 'hot', vmin = 0, vmax = VMAX) # ,aspect='auto'
        axes[1].imshow(cleanImg, cmap = 'hot', vmin = 0, vmax = VMAX)
        axes[2].imshow(BG, cmap = 'hot', vmin = 0, vmax = VMAX)

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
                axes[i_ind,j_ind].imshow(np.squeeze(views_orig[i,j,:,:]),cmap='hot',vmin=0,vmax=VMAX) #,aspect='auto',origin='lower'
                axes[i_ind,j_ind].axis('off')
    #            titleName = 'i='+ '{:.0f}'.format(i)+ ', j='+'{:.0f}'.format(j)            
                j = j + 1
            i = i + 1
        titleName = 'Foreground' +'.png'
        plt.savefig('./Multi-view_Original_Frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
        plt.show()  

        fig, axes = plt.subplots(11,11,figsize=(20,20)) # size_subimg
        ax_ind = 0
        fig.subplots_adjust(hspace = 0.1, wspace=0.1)
        i = 4
        for i_ind in np.arange(axes.shape[0]):
            j = 4
            for j_ind in np.arange(axes.shape[1]):
                axes[i_ind,j_ind].imshow(np.squeeze(views[i,j,:,:]),cmap='hot',vmin=0,vmax=VMAX) #,aspect='auto',origin='lower'
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
                axes[i_ind,j_ind].imshow(np.squeeze(background[i,j,:,:]),cmap='hot',vmin=0,vmax=VMAX) #,aspect='auto',origin='lower'
                axes[i_ind,j_ind].axis('off')
                j = j + 1
            i = i + 1
        titleName = 'Background' + '.png'
        plt.savefig('./Multi-view_Background_Frame'+str(frameInd)+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
        plt.show()  


    










