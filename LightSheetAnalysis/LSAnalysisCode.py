import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as s
import os
import pandas as pd
import re
from ImageAnalysis import ImageAnalysisCode


c = s.c # speed of light [meter/sec]
w0 = 2*np.pi* 446.799677e12 # resonance angular freq [1/sec]
lamb = 638e-9 # DMD light wavelength [meter]
om = 2*np.pi * (c / (lamb)) # DMD light angular freq [1/sec]
gamma = 36.898e6 # natural linewidth [1/s]

# dipole potential amplitude
U0 = -(3*np.pi*c**2)/(2*w0**3) * (gamma/(w0-om) + gamma/(w0+om))


def lin(x,m,b):
    y = m * x + b
    return y


def TotalCts2power(stats, plot=True):
    
    param_tot, _ = curve_fit(lin, stats['Power (W)'], stats['TotalCts_mean'])
    m,b = param_tot

    factor_CtsPerW = m # [Cts/W]

    if plot:
        # plot total cts vs. mW
        plt.figure()

        xfit = np.linspace(min(stats['Power (W)']), max(stats['Power (W)']), 100)
        yfit = lin(xfit, m, b)
        plt.plot(xfit*1e3, yfit)
        
        # convert to mW
        plt.errorbar(stats['Power (W)']*1e3, stats['TotalCts_mean'], yerr=stats['TotalCts_std'],fmt='o',capsize=3)
        plt.xlabel('Power (mW)')
        plt.ylabel('Total Cts')
        
        plt.tight_layout()
    
    return factor_CtsPerW


def MaxCts2power(stats, plot=True):
    
    param_tot, _ = curve_fit(lin, stats['Power (W)'], stats['MaxCts_mean'])
    m,b = param_tot

    factor_CtsPerW = m # [Cts/W]

    if plot:
        # plot total cts vs. mW
        plt.figure()

        xfit = np.linspace(min(stats['Power (W)']), max(stats['Power (W)']), 100)
        yfit = lin(xfit, m, b)
        plt.plot(xfit*1e3, yfit)
        
        # convert to mW
        plt.errorbar(stats['Power (W)']*1e3, stats['MaxCts_mean'], yerr=stats['MaxCts_std'],fmt='o',capsize=3)
        plt.xlabel('Power (mW)')
        plt.ylabel('MaxCts_mean Cts')
        
        plt.tight_layout()
    
    return factor_CtsPerW

def GetMaxIntensity(imagesList, df, pixArea_m, factor):
            
    maxIntensity = []
    
    for img in imagesList:
        
        power_arr = img / factor
        intensity_arr = power_arr / pixArea_m
        maxIntensity.append(np.max(intensity_arr))
    
    df['Max Intensity (W/m2)'] = maxIntensity
    
    return df


def GetDipolePotential(stats, maxCurrent_mA = 325, doPlot=True):
    
    # fit intensity data to controller current
    popt, _ = curve_fit(lin, stats['Current (mA)'], stats['Max Intensity (W/m2)'])
    
    # plot max intensity vs. controller current
    curr_fit = np.linspace(min(stats['Current (mA)']), 325, 100)
    maxI_fit = lin(curr_fit, *popt)
        
    # calculate dipole potential from max intensity
    U_dip = U0 * maxI_fit
    Temp_uK = U_dip / s.k * 1e6 # temp in microkelvin
    
    if doPlot:
        fig, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(curr_fit, Temp_uK, color="blue")
        ax1.set_xlabel('Controller current (mA)')
        ax1.set_ylabel('Temperature (uK)')
        ax1.tick_params(axis='x')
        fig.tight_layout()
        
    return U_dip, Temp_uK



def ExtractImages(fullPathList, ROI, metaData=None):
    
    if fullPathList[0].endswith('.dat') or metaData is not None:
        imgs = ImageAnalysisCode.GetImages(fullPathList, 'Andor', ROI, metaData)
        
    else:
        imgs = []
        for file in fullPathList:
            image_arr = ImageAnalysisCode.CheckFile(file)
            imgs.append(image_arr)
    
    return imgs


def ExtractRawCts(dataPathList, imagesList):
    
    df = pd.DataFrame(columns=['File',
                               'Power (W)','Current (mA)',
                               'TotalCts','MaxCts',
                               'Max Intensity (W/m2)'
                               ])
    
    # loops thru folders
    for j in range(len(imagesList)):
            
        # <anything> <number> mW <number> mA
        match = re.search(r"(\d+(?:\.\d*)?)\s*mW.*?(\d+(?:\.\d*)?)\s*mA", dataPathList[j], re.IGNORECASE)
        if match:
            power = float(match.group(1))
            current = float(match.group(2))
        else:
            power = None
            current = None            

        cts_tot = np.sum(np.sum(imagesList[j]))
        cts_max = np.max(imagesList[j])
        
        # store values in df
        df = pd.concat([df, pd.DataFrame({'File':[dataPathList[j]],
                                          'Power (W)':[power*1e-3],
                                          'Current (mA)':[current],
                                          'TotalCts':[cts_tot],
                                          'MaxCts':[cts_max],
                                          })
                        ], 
                       ignore_index=True)
    
    return df



def EstimateBGvalue(img, region_size=50, padding=10):
    
    H, W = img.shape
    
    # region boundaries
    row_end   = H - padding
    col_end   = W - padding
    row_start = row_end - region_size
    col_start = col_end - region_size

    # ensure region stays inside the image
    if row_start < 0 or col_start < 0:
        raise ValueError("Region extends outside the image. Reduce region size or padding.")
    
    # extract region
    region = img[row_start:row_end, col_start:col_end]
    
    return float(np.mean(region))



def BGsubtraction(bgFullPath, imagesList, ROI, metaData):
    
    imageBG = ExtractImages(bgFullPath, ROI, metaData)
    plt.imshow(imageBG[0])
    
    corrected_images = []
    for img in imagesList:
        corrected = img - imageBG[0]
        corrected = np.clip(corrected, 0, None)
        corrected_images.append(corrected)
    
    return corrected_images



def BGsubtraction_alt(imagesList):
    
    bgVal = EstimateBGvalue(imagesList[0])
    
    corrected_images = []
    for img in imagesList:
        corrected = img - bgVal
        # corrected = np.clip(corrected, 0, None)
        corrected_images.append(corrected)
    return corrected_images
    
        



