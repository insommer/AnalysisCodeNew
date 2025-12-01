# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:17:30 20x22

@author: Sommer Lab
"""
#import rawpy
#import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ImageAnalysis import ImageAnalysisCode
import os
import pandas as pd

####################################
#Set the date, the folder name, and the file name
####################################

data_path =r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
date = '11/22/2024'
data_folder = r'\Basler\scnd pass waist v power full range'

def gaussianBeam(x, amp, center, w, offset):
     return offset + amp*np.exp(-2*(x-center)**2/w**2)

def fitgaussian(xdata, ydata,xc,do_plot=True, plot_title = False):
    popt, pcov = curve_fit(gaussianBeam, xdata, ydata,p0=[np.max(ydata), xc, 100, 5e3])
    #print(popt)
    if (do_plot):
        if plot_title:
            plt.title(plot_title)
        plt.plot(xdata,ydata)
        plt.plot(xdata,gaussianBeam(xdata, *popt))
    return popt,pcov

######################################

dataLocation = ImageAnalysisCode.GetDataLocation(date,DataPath=data_path)

images = []
data = []

for filename in os.listdir(dataLocation + data_folder):
    
    path = dataLocation + data_folder + '/' + filename
    
    parts = filename.split('W_')
    power = float(parts[0])
    meas_iter = int(parts[1].split('.')[0])
 
    camera = 2
    
    if camera == 1:
        img = np.fromfile(path, dtype = np.uint16)
        width = 1288
        height = 964
        pixelsize_um = 3.75#microns
        
    if camera == 1.2:
        img = np.fromfile(path, dtype = np.uint8)
        width = 2048
        height = 1536
        pixelsize_um = 3.45#microns
    
    elif camera == 2:
        img = np.fromfile(path, dtype = np.uint8)
        width = 3840
        height = 2160
        pixelsize_um = 2
####################################
# Choose the camera
# 1 for the old Point Grey Chameleon, 
# 1.2 for the new Point Grey Chameleon, 
# 2 for Basler dart
####################################



    img_array = np.reshape(img, (height, width))
    rowstart = 0
    rowend = -1
    columnstart = 0
    columnend = -1
    height = height + rowend - rowstart
    width = width + columnend - columnstart
    img_array = img_array[rowstart:rowend, columnstart:columnend]
    images.append(img_array)
    
    sum_vs_x = np.sum(img_array,0)
    sum_vs_y = np.sum(img_array,1)
    x0 = np.argmax(sum_vs_x)
    y0 = np.argmax(sum_vs_y)
    
    slice_vs_x = img_array[y0+10,:]
    slice_vs_y= img_array[:,x0]
    
    
    
    xvalues = pixelsize_um*np.arange(width)
    yvalues = pixelsize_um*np.arange(height)
    
    
    popt,pcov =fitgaussian(xvalues, slice_vs_x, (x0)*pixelsize_um, plot_title = "slice vs. x", do_plot = False)
    
    plt.figure()
    popt2,pcov2 = fitgaussian(yvalues, slice_vs_y, y0*pixelsize_um, plot_title = "slice vs. y", do_plot = False)
    
    Xcenter = popt[1]
    Xwidth = popt[2]
    
    Ycenter = popt2[1]
    Ywidth = popt2[2]
    
    data.append([filename, power, meas_iter, Xwidth, Ywidth, Xcenter, Ycenter])
    
#%%

colnames = ['Filename', 'Power', 'Iter', 'Xwidth', 'Ywidth', 'Xcenter', 'Ycenter']
df = pd.DataFrame(data, columns=colnames)


avg_width = df.groupby('Power')[['Xwidth', 'Ywidth']].mean().reset_index()
std_width = df.groupby('Power')[['Xwidth', 'Ywidth']].std().reset_index()

#%%
fig, ax = plt.subplots(1,2,figsize=(8,3))


ax[0].errorbar(avg_width['Power'], avg_width['Xwidth'], std_width['Xwidth'], fmt='-o', capsize=3)
ax[0].set_xlabel('Power (W)')
ax[0].set_ylabel('X width (um)')

ax[1].errorbar(avg_width['Power'], avg_width['Ywidth'], std_width['Ywidth'],  fmt='-o', capsize=3)
ax[1].set_xlabel('Power (W)')
ax[1].set_ylabel('Y width (um)')

plt.tight_layout()

