# -*- coding: utf-8 -*-
"""
A set of general usage functions to complement IAF

Created on Thu Jul  5 10:38:52 2018

@author: peq10
"""
import numpy as np
import scipy
import tifffile
import glob
import scipy.ndimage as ndimage
import scipy.constants
import os
import scipy.stats
import matplotlib.pyplot as plt
import psutil

#if __name__ =='__main__':
#    import read_roi as rr
#else:
#    from . import read_roi as rr

def trailing_dim_mean(stack):
    return stack.reshape((stack.shape[0],-1)).mean(axis = 1)

def mem_check(max_mem_percentage = 95,response = 'throw_error'):
    #a check when loading things, etc. to try and stop freezing
    if psutil.virtual_memory().percent > max_mem_percentage:
        if response == 'throw_error':
            raise MemoryError
        else:
            return None
    else:
        return None

def read_redshirt_images(file_path):
    data = np.fromfile(file_path,dtype = np.int16)
    header_size = 2560
    header = data[:header_size]
    ncols, nrows = map(int, header[384:386])  # prevent int16 overflow
    nframes = int(header[4])
    frame_interval = header[388] / 1000
    if frame_interval >= 10:
        frame_interval *= header[390]  # dividing factor
    image_size = nrows * ncols * nframes
    bnc_start = header_size + image_size
    images = np.reshape(np.array(data[header_size:bnc_start]),
                        (nrows, ncols, nframes))
    return np.rollaxis(images,-1)

def check_and_make_directory(file_path):
     directory = os.path.dirname(file_path)
     if not os.path.isdir(directory):
         os.makedirs(directory)

def memory_efficient_average_stack(path,key,omit_fifth = True,keep_label = None,limit_repeats = False,num_repeats = None):
    filenames = getFilenamesByExtension(path)
    #excludes some of the stacks
    #get all file numbers
    nums = []
    for filename in filenames:
        loc = filename.find(key)+len(key)
        num = get_int_from_string(filename,loc)
        nums.append(num)
        
    
    if keep_label is not None:
        filenames = [f for idx,f in enumerate(filenames) if nums[idx] in keep_label]
        nums = [n for n in nums if n in keep_label]
    
    if omit_fifth:
        #find first non omitted stack
        first_loc = np.where(np.array(nums)%5 !=0)[0][0]
        mean_stack = tifffile.imread(filenames[first_loc]).astype(float)
        #now load and average stacks
        count = 1
        for idx,filename in enumerate(filenames):
            if idx == first_loc or idx in np.where(np.array(nums)%5 ==0)[0] :
                continue

            mean_stack += tifffile.imread(filename).astype(float)
            count += 1 
            if count ==num_repeats and limit_repeats:
                break
        
        mean_stack = mean_stack/len(np.where(np.array(nums)%5 !=0)[0])
        return mean_stack
    else:
        mean_stack = tifffile.imread(filenames[0]).astype(float)
        count = 1
        for idx,filename in enumerate(filenames[1:]):
            mean_stack += tifffile.imread(filename).astype(float)
            count += 1 
            if count ==num_repeats and limit_repeats:
                break
        return mean_stack/(idx+1)
        
        
        
def getFilenamesByExtension(path,fileExtension = '.tif',recursive_bool = True):
    if recursive_bool:
        return [file for file in glob.glob(path + '/**/*'+fileExtension, recursive=recursive_bool)]
    else:
        return [file for file in glob.glob(path + '/**'+fileExtension, recursive=recursive_bool)]
  


def sort_zipped_lists(list_of_lists,key_position = 0):
    res = zip(*sorted(zip(*list_of_lists),key = lambda x:x[key_position]))
    return [list(i) for i in res]

def get_int_from_string(string,loc,direction = 1):
    count = 0
    while True:
        try:
            if direction == 1:
                int(string[loc:loc + count +1])
            elif direction == -1:
                int(string[loc-count:loc+1])
            else:
                raise ValueError('Direction argument must be 1 or -1')
            count += 1
        except Exception:
            break
        
    if direction == 1:
        return int(string[loc:loc + count])
    elif direction == -1:
        return int(string[loc-count+1:loc+1])
    
    
def load_repeats_and_sort(path,key,omit_fifth = True):
    filenames = getFilenamesByExtension(path)

    if omit_fifth:
        stacks = []
        nums = []
        back_stacks = []
        for idx,filename in enumerate(filenames):
            stack = tifffile.imread(filename).astype(float)
            loc = filename.find(key)+len(key)
            num = get_int_from_string(filename,loc)
            if num%5 != 0:
                nums.append(num)
                stacks.append(stack)
            else:
                back_stacks.append(stack)
        
        sorted_stacks = sort_zipped_lists([nums,stacks])
        return sorted_stacks[1],back_stacks
    else:
        stacks = []
        nums = []
        for idx,filename in enumerate(filenames):
            stack = tifffile.imread(filename).astype(float)
            loc = filename.find(key)+len(key)
            num = get_int_from_string(filename,loc)
            nums.append(num)
            stacks.append(stack)

        
        sorted_stacks = sort_zipped_lists([nums,stacks])
        return sorted_stacks[1]
    
    
def correlationMap(stack):
    #a function that plots themean correlation of a pixel with its neighbourse
    stack = stack/np.sum(stack,0)
    correlationMap = np.zeros_like(stack[0,:,:])
    for idx1 in range(stack.shape[1]-1):
        if idx1 == 0:
            continue
        for idx2 in range(stack.shape[2]-1):
            if idx2 == 0:
                continue
            correlations = []
            for neighbour1 in [-1,0,1]:
                for neighbour2 in [-1,0,1]:
                    #measure neighbour correlation 
                    correlations.append(scipy.stats.pearsonr(stack[:,idx1,idx2],stack[:,idx1+neighbour1,idx2+neighbour2])[0])
            correlationMap[idx1,idx2] = (np.sum(correlations)-1)/(8)
    return correlationMap

def detrend_pixels(stack,fit_type = 'linear'):
    #a function to subtract a linear trend from each pixel in the stack
    detrended = np.zeros_like(stack)
    trends = np.zeros_like(stack)
    x = np.arange(stack.shape[0])
    fits = np.zeros((2,stack[0,:,:].shape[0],stack[0,:,:].shape[1]))
    for idx1 in range(stack.shape[1]):
        for idx2 in range(stack.shape[2]):
            if fit_type == 'linear':
                fit = scipy.stats.linregress(x,stack[:,idx1,idx2])
                fit_eval = fit.slope*x +fit.intercept
            elif fit_type == 'exp':
                fit = scipy.stats.linregress(x,np.log(stack[:,idx1,idx2]))
                fit_eval = np.exp(fit.slope*x +fit.intercept)
            else:
                raise ValueError('fit type not recognised')
                
            detrended[:,idx1,idx2] = stack[:,idx1,idx2] - fit_eval
            trends[:,idx1,idx2] = fit_eval
            fits[0,idx1,idx2] = fit.slope
            fits[1,idx1,idx2] = fit.intercept
    
    return detrended,trends,fits
            
def filter_pixels(stack,filter_type = 'median',kernel_size = [3,1,1]):
    
    if filter_type == 'median':
        return scipy.signal.medfilt(stack,kernel_size = [kernel_size,0,0])
    else:
        raise ValueError('Filter type not recognised')
        

def grouped_Z_project(stack,groupSize,projectType = 'mean'):
    #does a grouped z project like in imagej
    #trim stack if groupSize doesnt fit
    remainder  = stack.shape[0]%groupSize
    if remainder !=0:
        stack = stack[:-remainder,:,:]
    
    groupedStack = np.zeros((int(stack.shape[0]/groupSize),stack.shape[1],stack.shape[2])).astype(stack.dtype)
    
    for idx in range(groupedStack.shape[0]):
        if projectType == 'mean':
            groupedStack[idx,:,:] = np.mean(stack[idx*groupSize:idx*groupSize+groupSize+1,:,:],0)
        elif projectType == 'max':
            groupedStack[idx,:,:] = np.max(stack[idx*groupSize:idx*groupSize+groupSize+1,:,:],0)
        elif projectType == 'sum':
            groupedStack[idx,:,:] = np.sum(stack[idx*groupSize:idx*groupSize+groupSize+1,:,:],0)
        else:
            raise ValueError('Project type not recognised')
        #replace zeros in frame with mean of surrounding pixels 
        if (groupedStack[idx,:,:] == 0).all():
            raise ValueError('Image is all zeros')
        
   
            
    return groupedStack

def two_photon_res(wl,NA):
    return (0.383*wl)/NA**0.91

def power_to_photon_flux(wl,power,NA = 0.8):
    spot_area = 2*np.pi*two_photon_res(wl,NA)**2
    photon_flux = power*(wl/(scipy.constants.h*scipy.constants.c))
    flux_density = photon_flux/spot_area
    return flux_density

def norm(array):
    return (array - np.min(array))/(np.max(array) - np.min(array))
    
def read_roi_file(roi_filepath,im_dims = None):
    with open(roi_filepath,'rb') as f:
        roi = rr.read_roi(f)
    
    if im_dims is not None:
        im = np.zeros(im_dims)
        for pair in roi:
            im[pair[0],pair[1]] = 1            
        clos = ndimage.binary_dilation(im,iterations = 1)
        filled = ndimage.binary_fill_holes(clos)
        filled = ndimage.binary_erosion(filled)
        return roi,filled
    else:
        return roi
    
def radial_average_profile(array,center):
    #from http://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((array.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)
    
    tbin = np.bincount(r.ravel(), array.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff,fs,order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_filter(data, cutoff, fs,btype = 'high', order=5):
    if btype == 'high':
        b, a = butter_highpass(cutoff, fs, order=order)
    elif btype == 'low':
        b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y