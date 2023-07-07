# -*- coding: utf-8 -*-
"""
This file includes some functions for light-field calibration and preprocessing, for example:

- Detect the rotation angle of a raw LFM image.
- Detect lenslet pitch.
- Detect the centers of the microlens array using an out-of-focus LFM image, instead of a white image.
- Perform matrix factorization to separate the foreground and background.


References：
----------------------------
[1] P. Song, H. Verinaz Jadan, C. L. Howe, P. Quicke, A. J. Foust and P. L. Dragotti, "3D Localization for Light-Field Microscopy via Convolutional Sparse Coding on Epipolar Images," in IEEE Transactions on Computational Imaging, doi: 10.1109/TCI.2020.2997301.
[2] P. Song, H. Verinaz Jadan, P. Quicke, C. L. Howe, A. J. Foust, and P. L. Dragotti, "Location Estimation for Light Field Microscopy based on Convolutional Sparse Coding," in Imaging and Applied Optics 2019 (COSI, IS, MATH, pcAOP), OSA Technical Digest (Optical Society of America, 2019), paper MM2D.2


Contact:
----------------------------
Please report problems to

Pingfan Song

Electronic and Electrical Engineering,

Imperial College London

p.song@imperial.ac.uk

"""


#%% ---------------------------------------------- 
import matplotlib.pyplot as plt
from . import general_functions as gf
import skimage.transform
import scipy.signal
import numpy as np
import time
import scipy.interpolate
import scipy.stats as st
from scipy import signal
from numpy.linalg import svd


#%% ---------------------------------------------- 
#--Detect the rotation angle of a raw LFM image.
#--Detect lenslet pitch.

def rotate_test(im,angles):
    ra = []
    for i in angles:
        im_r = skimage.transform.rotate(im,i)
        high_sum_x = gf.butter_filter(np.sum(im_r,0),1,100,btype = 'high')
        ra.append(high_sum_x.max()-high_sum_x.min())
    plt.plot(angles,ra)
    plt.savefig('./angle_response.png',transparent = True,pad_inches = 0,dpi=300) # ,bbox_inches='tight',
    plt.show()
    return angles[np.argmax(ra)]


def get_lf_dims(test_im):
    #find the rotation
    angles =  np.arange(-5,5,0.1)
    im_angle = rotate_test(test_im,angles)
    angles2 = np.arange(im_angle-1,im_angle+1,0.01)
    im_angle = rotate_test(test_im,angles2)
    angles3 = np.arange(im_angle-0.1,im_angle+0.1,0.001)
    im_angle = rotate_test(test_im,angles3)
    
    #now find the image spacing
    im_rotated = skimage.transform.rotate(test_im,im_angle)
    
    sum_x = np.sum(im_rotated,1)
    sum_x_highpass = gf.butter_filter(sum_x,1,100,btype = 'high')
    fftx = np.fft.fft(sum_x_highpass)
    freqs = np.fft.fftfreq(len(sum_x))
    spot_freq = freqs[np.argmax(np.abs(fftx[freqs>0]))]
    spot_period = 1/spot_freq
    print(spot_period)
      
    #now calculate vector
    r = [spot_period*np.sin(im_angle*(np.pi/180)),spot_period*np.cos(im_angle*(np.pi/180))]
    
    #now need to find the center
    center0 = (np.array(test_im.shape)/2).astype(int)
    cent_roi = test_im[center0[0]-int(spot_period):center0[0]+int(spot_period),center0[1]-int(spot_period):center0[1]+int(spot_period)]
    plt.imshow(cent_roi)
    plt.show()
    
    #algo to find a central pixel:
    #find circle location th image that maximises sum.
    #convolve with a cicle or radius r
    radius = 7
    kern = np.zeros((2*radius+1,2*radius+1))
    for idx1 in range(kern.shape[0]):
        for idx2 in range(kern.shape[1]):
            if np.sqrt((idx1-radius)**2+(idx2-radius)**2)<radius:
                kern[idx1,idx2] = 1
            else:
                continue
    conv = scipy.signal.convolve(kern,cent_roi)
    conv = conv[radius:-radius,radius:-radius]
    plt.imshow(conv)
    plt.show()
    coords = np.unravel_index(np.argmax(conv),cent_roi.shape)
    center1 = np.array(coords)+np.array([center0[0]-int(spot_period),center0[1]-int(spot_period)])
    cent_roi = test_im[center1[0]-int(spot_period):center1[0]+int(spot_period),center1[1]-int(spot_period):center1[1]+int(spot_period)]
    plt.imshow(cent_roi)
    plt.show()
    
    #now do a repeat on an interpolation
    x = np.arange(0,2*radius+5,1)
    resampled = scipy.interpolate.RectBivariateSpline(x,x,test_im[(center1[0])-radius-2:center1[0]+radius+3,center1[1]-radius-2:center1[1]+radius+3])
    resampling = 0.01
    x2 = np.arange(0,2*radius+4,resampling)
    upsampled = resampled(x2,x2)
    plt.imshow(upsampled)
    plt.show()
    k0 = (x2*np.ones_like(upsampled)-radius-2)
    k = np.sqrt((k0)**2 + (np.rollaxis(k0,1))**2) < radius
    plt.imshow(k)
    plt.show()
    conv2 = scipy.signal.convolve(k,upsampled)
    plt.imshow(conv2)
    plt.show()
    
    coords2 = np.unravel_index(np.argmax(conv2),conv2.shape)
    offset = (np.array(coords2) -np.array(conv2.shape)/2)*resampling
    
    center2 = center1+offset
    
    return r,center2, im_angle, spot_period

def rotate(vec,theta,units = 'degrees'):
    if units == 'degrees':
        theta = theta*np.pi/180
    if vec.shape != (2,1):
        vec = np.matrix(vec).transpose()
    mat = np.matrix([[np.cos(theta),-1*np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.squeeze(np.array(mat*vec))

    
def find_angle(test_im, savefig = 0):
    def rotate_test(im,angles):
        ra = []
        for i in angles:
            im_r = skimage.transform.rotate(im,i)
            high_sum_x = gf.butter_filter(np.sum(im_r,0),1,100,btype = 'high')
            ra.append(high_sum_x.max()-high_sum_x.min())
        
        plt.figure(figsize=(3,2)); 
        plt.plot(angles,ra)
        #plt.stem(angles[np.argmax(ra)],ra[np.argmax(ra)], linefmt='--',use_line_collection=True)
        plt.scatter(angles[np.argmax(ra)],ra[np.argmax(ra)], c='r',s=10)
        plt.xlabel('angle')
        plt.ylabel('intensity contrast') 
        if savefig:
            now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
            titleName = './angle_response_'+now+'.png'
            plt.savefig(titleName,transparent = True,pad_inches = 0,bbox_inches = 'tight',dpi=300) #
        plt.show()
    
        return angles[np.argmax(ra)]
    
    #find the rotation
    angles =  np.arange(-5,5,0.1)
    im_angle = rotate_test(test_im,angles)
    angles2 = np.arange(im_angle-1,im_angle+1,0.01)
    im_angle = rotate_test(test_im,angles2)
    angles3 = np.arange(im_angle-0.1,im_angle+0.1,0.001)
    im_angle = rotate_test(test_im,angles3)

    #now find the image spacing
    im_rotated = skimage.transform.rotate(test_im,im_angle)
    
    sum_x = np.sum(im_rotated,1)
    sum_x_highpass = gf.butter_filter(sum_x,1,100,btype = 'high')
    fftx = np.fft.fft(sum_x_highpass)
    freqs = np.fft.fftfreq(len(sum_x))
    max_ind = np.argmax(np.abs(fftx)) # np.argmax(np.abs(fftx[freqs>0]))
    fftx_max = np.abs(fftx[max_ind])
    spot_freq = freqs[max_ind] # find the largest frequency point
    spot_period = 1/spot_freq
    print(spot_period)
    

    fig, axes = plt.subplots(2,1,figsize=(5,4))
    fig.subplots_adjust(hspace = 0.4, wspace=0)
#    axes[0].plot(sum_x[::1])
    axes[0].plot(sum_x_highpass[::1])
    axes[0].set_xlabel('column index', fontsize=12)
    axes[0].set_ylabel('column-wise sum', fontsize=12)
    axes[1].plot(freqs, np.abs(fftx))
    axes[1].scatter(spot_freq, fftx_max, c='r',s=10)
    axes[1].set_xlabel('frequency', fontsize=12)
    if savefig:
        plt.savefig('./SearchPitch'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
    plt.show() 
    
#    # longer figure
#    fig, axes = plt.subplots(2,1,figsize=(8,2))
#    fig.subplots_adjust(hspace = 1, wspace=0)
#    axes[0].plot(sum_x_highpass[::1])
#    axes[0].set_xlabel('column index', fontsize=10)
#    axes[0].set_ylabel('col sum', fontsize=10)
#    axes[1].plot(freqs, np.abs(fftx))
#    axes[1].scatter(spot_freq, fftx_max, c='r',s=10)
#    axes[1].set_xlabel('frequency', fontsize=10)
#    plt.savefig('./SearchPitch'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=300)
      
    #now calculate vector
    r = [spot_period*np.sin(im_angle*(np.pi/180)),spot_period*np.cos(im_angle*(np.pi/180))]
    r = np.array(r)
    return r, im_angle, spot_period
    
    
def show_rotation_response():
    img = gkern_array(kernlen=19, nsig=3, row=6, col=6, angle = 3, savefig=0) # angle = 0, 5, 10
    img_sum = gf.butter_filter(np.sum(img,0),1,100,btype = 'high')
    response = img_sum.max()-img_sum.min()
    img_sum_tile = np.tile(img_sum, (10,1))
    
    
    plt.figure(figsize=(3,3)); 
    plt.imshow(img, cmap='hot',vmin=0,vmax=0.01)
    plt.savefig('./img_angle_.png',transparent = True,pad_inches = 0,bbox_inches='tight',dpi=300)
    plt.show()
    
    
    fig, axes = plt.subplots(2,1,figsize=(3,3))
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].imshow(img_sum_tile, cmap='hot',vmin=-0.5,vmax=0.5)
    axes[0].axis('off')
    axes[1].plot(img_sum)
    axes[1].set_xlabel('column index')
    axes[1].set_ylabel('column-wise sum')
    axes[1].set_ylim((-0.5, 0.5))
    plt.savefig('./imgsum_angle_.png',transparent = True,pad_inches = 0,bbox_inches='tight',dpi=300)
    plt.show()    
    

def get_views_nd_interp(stack,center,r,rad_spots,n_views):
    #also get to work on nonstacks
    if len(stack.shape) == 2:
        stack = np.expand_dims(stack,0)
    
    d = rotate(r,-90)#assuming square array
    #create an array of views
    #viewy, viewx, frame no, y,x
    views = np.zeros((n_views,n_views,stack.shape[0],rad_spots*2+1,rad_spots*2+1))
    
    view_locs = int((n_views-1)/2)
    t0 = time.time()
    ta = time.time()

    interp = scipy.interpolate.RegularGridInterpolator((np.arange(stack.shape[0]),np.arange(stack.shape[1]),np.arange(stack.shape[2])),stack)
    fr = np.arange(stack.shape[0])
    
    t0 = time.time()
    for idx1,view_y in enumerate(range(-1*view_locs,view_locs+1,1)):
        for idx2,view_x in enumerate(range(-1*view_locs,view_locs+1,1)):
            for idx3,px_y in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
                for idx4,px_x in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
                    pos = center+ px_y*d+px_x*r
                    views[idx1,idx2,:,idx3,idx4] = interp((fr,pos[0]+view_y,pos[1]+view_x))
            print(time.time()-t0)

    
    print('total time :'+str(time.time()-ta))
    return views

def get_views(stack,center,r,rad_spots,n_views):

    # remove fr in order to extract only patches
    
    d = rotate(r,-90)#assuming square array
    #create an array of views
    #viewy, viewx, frame no, y,x
    views = np.zeros((n_views,n_views,rad_spots*2+1,rad_spots*2+1))
    
    view_locs = int((n_views-1)/2)
    t0 = time.time()
    ta = time.time()

#    interp = scipy.interpolate.RegularGridInterpolator((np.arange(stack.shape[1]),np.arange(stack.shape[2])),stack)
    
    t0 = time.time()
    for idx1,view_y in enumerate(range(-1*view_locs,view_locs+1,1)):
        for idx2,view_x in enumerate(range(-1*view_locs,view_locs+1,1)):
            for idx3,px_y in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
                for idx4,px_x in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
                    pos = center+ px_y*d+px_x*r
#                    pos = pos.astype(int)
#                    views[idx1,idx2,idx3,idx4] = interp((pos[0]+view_y,pos[1]+view_x))
#                    views[idx1,idx2,idx3,idx4] = stack[pos[0]-7:pos[0]+8,pos[1]-7:pos[1]+8]
                    views[idx1,idx2,idx3,idx4] =  stack[int(np.round(pos[0]+view_y)),int(np.round(pos[1]+view_x))]
        print(time.time()-t0)

#    idx1=0
#    view_y = -7
#    idx2=0
#    view_x = -7
#    for idx3,px_y in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
#        for idx4,px_x in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
#            pos = center+ px_y*d+px_x*r
##            pos = pos.astype(int)
#            views[idx1,idx2,idx3,idx4] =  stack[int(np.round(pos[0]+view_y)),int(np.round(pos[1]+view_x))]
#    print(time.time()-t0)

    print('total time :'+str(time.time()-ta))
    return views


#views[idx1,idx2,:,idx3,idx4] =  stack[fr,int(np.round(pos[0]+view_y)),int(np.round(pos[1]+view_x))]
  
 

#%% ---------------------------------------------- 
#--Detect the centers of microlens array using an out-of-focus LFM image, instead of white image.

def find_center(test_im,spot_period,initial_center,kernel_type=0,quiet=True):
#now need to find the center
    
    center0 = initial_center
#    center0 = (np.array(test_im.shape)/2).astype(int)
    cent_roi = test_im[center0[0]-int(spot_period):center0[0]+int(spot_period)+1,
                       center0[1]-int(spot_period):center0[1]+int(spot_period)+1]

#    plt.imshow(cent_roi)
#    plt.show()
    
    # find the central pixel via convolving with a kernel
    if kernel_type == 0:
        # using a uniform pan-shape kernel      
        radius = np.amin([9, int(spot_period)])
        kern = np.zeros((2*radius+1,2*radius+1))
        for idx1 in range(kern.shape[0]):
            for idx2 in range(kern.shape[1]):
                if np.sqrt((idx1-radius)**2+(idx2-radius)**2)<radius:
                    kern[idx1,idx2] = 1
                else:
                    continue
        conv = scipy.signal.convolve(kern,cent_roi)
        conv = conv[radius:-radius,radius:-radius]

    elif kernel_type == 1:
        # using a Gaussian kernel
        radius = np.amin([15, int(spot_period)])
        kern = gkern(kernlen=radius, nsig=0.5)
        conv = signal.convolve2d(cent_roi, kern, mode='same',boundary='fill', fillvalue=0)

    else:
        conv = cent_roi
            
    
    coords = np.unravel_index(np.argmax(conv),cent_roi.shape)
    direction = np.array(coords)-np.array([int(spot_period),int(spot_period)])
    center1 = center0 + direction
    cent_roi1 = test_im[center1[0]-int(spot_period):center1[0]+int(spot_period)+1,
                       center1[1]-int(spot_period):center1[1]+int(spot_period)+1]
    
  
    if quiet != True:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(15,3))
        ax1.imshow(cent_roi,cmap='hot'); ax1.set_title('ini roi')
        ax2.imshow(kern,cmap='hot'); ax2.set_title('kernel')
        ax3.imshow(conv,cmap='hot'); ax3.set_title('convolved')
        ax3.scatter(coords[1],coords[0],c='b',s=20)
        ax4.imshow(cent_roi,cmap='hot'); ax4.set_title('ini roi with center')
        ax4.scatter(coords[1],coords[0],c='b',s=20)
        ax5.imshow(cent_roi1,cmap='hot'); ax5.set_title('new roi with center')
        ax5.scatter(int(spot_period),int(spot_period),c='b',s=20)
        plt.show()
#        plt.figure(figsize = (6,6))
#        plt.imshow(test_im,origin='lower')
#        plt.scatter(center0[0],center0[1], c='b', s= 40, marker='o')
#        plt.scatter(center1[0],center1[1], c='r', s= 40, marker='x')
        
        print("radius:", radius, "conv.shape:", conv.shape, "coords:",coords, 
              "center0:", center0, "center1:", center1)
    
    return center1


def find_center_robust(test_im,spot_period,initial_center,kernel_type=0,quiet=True):
#now need to find the center
    
    ratio = 1.5 # decide the area of ROI for dectecting the center, typically, 1.5 ~ 1.75 
    center0 = initial_center
#    center0 = (np.array(test_im.shape)/2).astype(int)
    cent_roi = test_im[center0[0]-int(ratio*spot_period):center0[0]+int(ratio*spot_period)+1,
                       center0[1]-int(ratio*spot_period):center0[1]+int(ratio*spot_period)+1]
    
    # find the central pixel via convolving with a kernel
    if kernel_type == 2:
        # using a larger kernel consisting of 9 uniform pan-shape kernels    
        
        # one pan-shape kernel
        radius = np.amin([9, int(spot_period)])
        kernUnit = np.zeros((int(spot_period),int(spot_period)))
        radiusUnit = np.array((kernUnit.shape[0]-1)/2).astype(int)
#        print(kernUnit.shape)
        for idx1 in range(kernUnit.shape[0]):
            for idx2 in range(kernUnit.shape[1]):
                if np.sqrt((idx1-radiusUnit)**2+(idx2-radiusUnit)**2)<radius:
                    kernUnit[idx1,idx2] = 1
                else:
                    continue
        
#        plt.figure
#        plt.imshow(kernUnit)
#        plt.show()
        
        # construct a larger kernel consisting of 9 pan-shape kernel
        Row = kernUnit
        for iRow in np.arange(0,2):
            Row = np.concatenate((Row,kernUnit), axis = 1)
        
        kern = Row.copy()
        for iCol in np.arange(0,2):
            kern = np.concatenate((kern,Row), axis = 0)

#        print(Row.shape, kern.shape)   
        
        conv = signal.convolve2d(cent_roi, kern, mode='same',boundary='fill', fillvalue=0)
        
#        # manual convolution
#        conv2 = scipy.signal.convolve(kern,cent_roi)
#        radius2 = np.array((kern.shape[0]-1)/2).astype(int)
#        conv2 = conv2[radius2:-radius2,radius2:-radius2]
#        conv_diff = conv - conv2
        
    else:
        conv = cent_roi
            
    
        
    coords = np.unravel_index(np.argmax(conv),cent_roi.shape)
    
    direction = np.array(coords)-np.array([int(ratio*spot_period),int(ratio*spot_period)])
    center1 = center0 + direction
    cent_roi1 = test_im[center1[0]-int(ratio*spot_period):center1[0]+int(ratio*spot_period)+1,
                       center1[1]-int(ratio*spot_period):center1[1]+int(ratio*spot_period)+1]
    
    if quiet != True:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(15,3))
        ax1.imshow(cent_roi,cmap='hot'); ax1.set_title('ini roi')
        ax2.imshow(kern,cmap='hot'); ax2.set_title('kernel')
        ax3.imshow(conv,cmap='hot'); ax3.set_title('convolved')
        ax3.scatter(coords[1],coords[0],c='b',s=20)
        ax4.imshow(cent_roi,cmap='hot'); ax4.set_title('ini roi with center')
        ax4.scatter(coords[1],coords[0],c='b',s=20)
        ax5.imshow(cent_roi1,cmap='hot'); ax5.set_title('new roi with center')
        ax5.scatter(int(ratio*spot_period),int(ratio*spot_period),c='b',s=20)
        plt.show()
#        plt.figure(figsize = (6,6))
#        plt.imshow(test_im,origin='lower')
#        plt.scatter(center0[0],center0[1], c='b', s= 40, marker='o')
#        plt.scatter(center1[0],center1[1], c='r', s= 40, marker='x')
        
        print("radius:", radius, "conv.shape:", conv.shape, "coords:",coords, 
              "center0:", center0, "center1:", center1)
    
    return center1



def get_views_center(stack,r,rad_spots,n_views):
    
    center0 = (np.array(stack.shape)/2).astype(int)
    cent_roi = stack[center0[0]-int(n_views):center0[0]+int(n_views),center0[1]-int(n_views):center0[1]+int(n_views)]
    plt.imshow(cent_roi)
    plt.show()
    
    # produce a Gaussian kernel
    kern = gkern(kernlen=n_views, nsig=20)
    conv = signal.convolve2d(cent_roi, kern, boundary='symm', mode='same')

    plt.imshow(conv)
    plt.show()
    coords = np.unravel_index(np.argmax(conv),cent_roi.shape)
    center1 = np.array(coords)+np.array([center0[0]-int(n_views),center0[1]-int(n_views)])
    cent_roi = stack[center1[0]-int(n_views):center1[0]+int(n_views),center1[1]-int(n_views):center1[1]+int(n_views)]
    plt.imshow(cent_roi)
    plt.show()

    # remove fr in order to extract only patches
    center = center1
    d = rotate(r,-90)#assuming square array
    #create an array of views
    #viewy, viewx, frame no, y,x
    views = np.zeros((n_views,n_views,rad_spots*2+1,rad_spots*2+1))
    
    view_locs = int((n_views-1)/2)
    t0 = time.time()
    ta = time.time()

#    interp = scipy.interpolate.RegularGridInterpolator((np.arange(stack.shape[1]),np.arange(stack.shape[2])),stack)
    
    t0 = time.time()
    for idx1,view_y in enumerate(range(-1*view_locs,view_locs+1,1)):
        for idx2,view_x in enumerate(range(-1*view_locs,view_locs+1,1)):
            for idx3,px_y in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
                for idx4,px_x in enumerate(np.arange(-rad_spots,rad_spots+1,1)):
                    pos = center+ px_y*d+px_x*r
                    views =  stack[int(np.round(pos[0]+view_y)),int(np.round(pos[1]+view_x))]
                    
                    
#                    pos = pos.astype(int)
#                    views[idx1,idx2,idx3,idx4] = interp((pos[0]+view_y,pos[1]+view_x))
#                    views[idx1,idx2,idx3,idx4] = stack[pos[0]-7:pos[0]+8,pos[1]-7:pos[1]+8]
                    views[idx1,idx2,idx3,idx4] =  stack[int(np.round(pos[0]+view_y)),int(np.round(pos[1]+view_x))]
        print(time.time()-t0)


def gkern(kernlen=19, nsig=3):
    """Returns a 2D Gaussian kernel."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel



def gkern_array(kernlen=19, nsig=3, row=3, col=3, angle = 0, savefig=0):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    kernUnit=kernel.copy()
    
    Row = kernUnit
    for iRow in np.arange(0,row-1):
        Row = np.concatenate((Row,kernUnit), axis = 1)
    
    kernel = Row.copy()
    for iCol in np.arange(0,col-1):
        kernel = np.concatenate((kernel,Row), axis = 0)
        
    kernel = skimage.transform.rotate(kernel,angle)
    
    plt.figure(figsize=(3,3)); 
    plt.imshow(kernel, cmap='hot')
    if savefig:
        plt.savefig('./kernel.png',transparent = True,pad_inches = 0,bbox_inches='tight',dpi=300) #
    plt.show()

    return kernel



    
#%% ----------------------------------------------  
#--Perform SVD based matrix factorization to separate the foreground and background.
#--Convert the foreground to a clean LF image.

   
def RM_Background(views, background_ratio = 0.9, thresh = 0.6, show_fig=0):
#    views = LF4D_arr[frameInd,:,:,:,:]
    
#    thresh = 0.6 # 0.2 # sparsity threshold
#    background_ratio = 0.9# 0.5 # 0.6  # background percentage in the largest singular value
    
    SZ = views.shape;
    views = views.reshape((SZ[0]* SZ[1], SZ[2]*SZ[3]))

    [U, S, Vh] = svd(views, full_matrices=False);
    U.shape, S.shape, Vh.shape
    S1 = S.copy()
    
    S1[S1<thresh] = 0
    S1[0] = (1-background_ratio)*S1[0] # remove the largest singular value in order to remove the background.
    views_new = np.dot(U * S1, Vh)
    views_new = views_new.reshape((SZ[0], SZ[1], SZ[2], SZ[3]))

    Sb = np.zeros_like(S)
    Sb[0] = background_ratio*S[0]
    background = np.dot(U * Sb, Vh) 
    background = background.reshape((SZ[0], SZ[1], SZ[2], SZ[3]))
    background[background<0] = 0;
    
    if show_fig:
        fig, axes = plt.subplots(11,11,figsize=(20,20)) # size_subimg
        ax_ind = 0
        VMAX = np.amax(views)
        fig.subplots_adjust(hspace = 0.1, wspace=0.1)
        i = 4
        for i_ind in np.arange(axes.shape[0]):
            j = 4
            for j_ind in np.arange(axes.shape[1]):
                axes[i_ind,j_ind].imshow(np.squeeze(views_new[i,j,:,:]),cmap='hot',vmin=0,vmax=VMAX) #,aspect='auto',origin='lower'
                axes[i_ind,j_ind].axis('off')
    #            titleName = 'i='+ '{:.0f}'.format(i)+ ', j='+'{:.0f}'.format(j)            
                j = j + 1
            i = i + 1
        titleName = 'Foreground' +'.png'
#        plt.savefig(titleName,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
        plt.show()      
            
        fig, axes = plt.subplots(11,11,figsize=(20,20)) # size_subimg
        ax_ind = 0
        VMAX = np.amax(views)
        fig.subplots_adjust(hspace = 0.1, wspace=0.1)
        i = 4
        for i_ind in np.arange(axes.shape[0]):
            j = 4
            for j_ind in np.arange(axes.shape[1]):
                axes[i_ind,j_ind].imshow(np.squeeze(background[i,j,:,:]),cmap='hot',vmin=0,vmax=VMAX) #,aspect='auto',origin='lower'
                axes[i_ind,j_ind].axis('off')
    #            titleName = 'i='+ '{:.0f}'.format(i)+ ', j='+'{:.0f}'.format(j)            
                j = j + 1
            i = i + 1
        titleName = 'Background' + '.png'
#        plt.savefig(titleName,bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
        plt.show()  
    
    return views_new, background, S, S1, Sb
    

#%% SVD to remove background of EPI
    
def RM_Background_EPI(EPI_array, background_ratio = 0.9, thresh = 0.6, show_fig=0):
  
#    thresh = 0.6 # 0.2 # sparsity threshold
#    background_ratio = 0.9# 0.5 # 0.6  # background percentage in the largest singular value
    
    SZ = EPI_array.shape;
    EPI_array = EPI_array.reshape((SZ[0], SZ[1]*SZ[2]))

    [U, S, Vh] = svd(EPI_array, full_matrices=False);
    U.shape, S.shape, Vh.shape
    S1 = S.copy()
    
    S1[S1<thresh] = 0
    S1[0] = (1-background_ratio)*S1[0] # remove the largest singular value in order to remove the background.
    EPI_array_new = np.dot(U * S1, Vh)
    EPI_array_new = EPI_array_new.reshape((SZ[0], SZ[1], SZ[2]))

    Sb = np.zeros_like(S)
    Sb[0] = background_ratio*S[0]
    background = np.dot(U * Sb, Vh) 
    background = background.reshape((SZ[0], SZ[1], SZ[2]))
    background[background<0] = 0;
    
    if show_fig:
        fig, axes = plt.subplots(3,3,figsize=(11,7))
        fig.subplots_adjust(hspace = 0.5, wspace=0.3)
        axes = axes.ravel()
        VMAX = np.amax(EPI_array)
        for ii in np.arange(0, 9 ):
            axes[ii].imshow(EPI_array_new[ii,:,:],cmap='hot',aspect='auto',origin='lower',vmin=0,vmax=VMAX); 
        #    axes[ii].grid(True) # axes[ii].axis('off')
            axes[ii].set_title('depth='+str(ii*4)+'um')
            axes[ii].set_xlabel('l axis')
            axes[ii].set_ylabel('j axis')
#        plt.savefig('./Foreground_EPI'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
        plt.show()
             
   
        fig, axes = plt.subplots(3,3,figsize=(11,7))
        fig.subplots_adjust(hspace = 0.5, wspace=0.3)
        axes = axes.ravel()
        VMAX = np.amax(EPI_array)
        for ii in np.arange(0, 9 ):
            axes[ii].imshow(background[ii,:,:],cmap='hot',aspect='auto',origin='lower',vmin=0,vmax=VMAX); 
        #    axes[ii].clim(0,np.amax(EPI_jl))
        #    axes[ii].grid(True)
            axes[ii].set_title('depth='+str(ii*4)+'um')
            axes[ii].set_xlabel('l axis')
            axes[ii].set_ylabel('j axis')
#        plt.savefig('./Background_EPI'+'.png',bbox_inches='tight',transparent = True,pad_inches = 0,dpi=mydpi)
        plt.show()
    return EPI_array_new, background, S, S1, Sb
    
    



















