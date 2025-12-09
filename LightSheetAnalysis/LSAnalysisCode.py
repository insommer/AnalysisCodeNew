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


def GetMaxIntensity(dataPathList, stats, pixArea_m, factor):
            
    maxAvgIntensity = []
    
    for folder in dataPathList:

        maxI = []
        
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            image_arr = ImageAnalysisCode.CheckFile(path)
            
            power_arr = image_arr / factor
            intensity_arr = power_arr / pixArea_m
            
            maxI.append(np.max(intensity_arr))
        
        # calculate avg max intensity over all images in this folder
        avgI = np.mean(maxI)
        maxAvgIntensity.append(avgI)
    
    stats['Max Intensity (W/m2)'] = maxAvgIntensity
    
    return stats


def GetDipolePotential(stats, maxCurrent_mA = 325, doPlot=True):
    
    # fit intensity data to controller current
    popt, _ = curve_fit(lin, stats['Current (mA)'], stats['Max Intensity (W/m2)'])
    
    # plot max intensity vs. controller current
    curr_fit = np.linspace(min(stats['Current (mA)']), 325, 100)
    maxI_fit = lin(curr_fit, *popt)
        
    # calculate dipole potential from max intensity
    U_dip = U0 * maxI_fit
    Temp_uK = U_dip / s.k * 1e6
    
    if doPlot:
        fig, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(curr_fit, Temp_uK, color="blue")
        ax1.set_xlabel('Controller current (mA)')
        ax1.set_ylabel('Temperature (uK)')
        ax1.tick_params(axis='x')
        fig.tight_layout()
        
    return U_dip, Temp_uK
    

def ExtractRawCts(dataPathList, ROI, metaData=None):
    
    df = pd.DataFrame(columns=['File',
                               'Power (W)','Current (mA)',
                               'TotalCts','MaxCts',
                               'Max Intensity (W/m2)'
                               ])
    
    # loops thru folders
    for folder in dataPathList:
            
        # <anything> <number> mW <number> mA
        match = re.search(r"(\d+(?:\.\d*)?)\s*mW.*?(\d+(?:\.\d*)?)\s*mA", folder, re.IGNORECASE)
        if match:
            power = float(match.group(1))
            current = float(match.group(2))
        else:
            power = None
            current = None

        # loop thru files in folder
        for filename in os.listdir(folder):
            
            path = os.path.join(folder,filename)
            
            # for Andor files
            if filename.endswith('.dat') and metaData is not None:
                image_arr = ImageAnalysisCode.GetImages(path, 'Andor', ROI, metaData)
            else:
                image_arr = ImageAnalysisCode.CheckFile(path)
                image_arr = image_arr[ROI[0]:ROI[1], ROI[2]:ROI[3]]

            cts_tot = np.sum(np.sum(image_arr))
            cts_max = np.max(image_arr)
            
            # store values in df
            df = pd.concat([df, pd.DataFrame({'File':[path],
                                              'Power (W)':[power*1e-3],
                                              'Current (mA)':[current],
                                              'TotalCts':[cts_tot],
                                              'MaxCts':[cts_max],
                                              })
                            ], 
                           ignore_index=True)
    return df

def ExtractRawCts_v2(fullPath, ROI, metaData=None):
    
    df = pd.DataFrame(columns=['File',
                               'Power (W)','Current (mA)',
                               'TotalCts','MaxCts',
                               'Max Intensity (W/m2)'
                               ])
    
    # loops thru folders
    for file in fullPath:
            
        # <anything> <number> mW <number> mA
        match = re.search(r"(\d+(?:\.\d*)?)\s*mW.*?(\d+(?:\.\d*)?)\s*mA", file, re.IGNORECASE)
        if match:
            power = float(match.group(1))
            current = float(match.group(2))
        else:
            power = None
            current = None
        
            
        # for Andor files
        if file.endswith('.dat'):
            image_arr = ImageAnalysisCode.GetImages(file, 'Andor', ROI, metaData)
            print(image_arr)
        else:
            image_arr = ImageAnalysisCode.CheckFile(file)
            image_arr = image_arr[ROI[0]:ROI[1], ROI[2]:ROI[3]]

        cts_tot = np.sum(np.sum(image_arr))
        cts_max = np.max(image_arr)
        
        # store values in df
        df = pd.concat([df, pd.DataFrame({'File':[file],
                                          'Power (W)':[power*1e-3],
                                          'Current (mA)':[current],
                                          'TotalCts':[cts_tot],
                                          'MaxCts':[cts_max],
                                          })
                        ], 
                       ignore_index=True)
    return df