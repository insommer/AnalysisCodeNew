import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.optimize import curve_fit
from ImageAnalysis import ImageAnalysisCode
from LightSheetAnalysis import LSAnalysisCode
import datetime
import configparser
from PIL import Image
import cv2


plt.close('all')

dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
# dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'
date = '12/10/2025'

# camera = 'Basler'
powr = 15
camera = 'Andor'
data_folder = [
    # fr'{camera}/Focus LS 3.302 mm',
    # fr'{camera}/Focus LS 3.556 mm',
    # fr'{camera}/Focus LS 3.81 mm',
    # fr'{camera}/Focus LS 4.064 mm',
    fr'{camera}/Measure separation'



    ]

repetition = 6
commonPhrase = True
quantity = 'Distance (mm)'
var2plot = 'Distance'
multiG = True

doPlot = 1
angle = 0

rowstart=720
rowend=880
columnstart=175
columnend=525
ROI = [rowstart, rowend, columnstart, columnend]

dayFolder = ImageAnalysisCode.GetDataLocation(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]

if camera == 'Basler':
    pixSize = 2 #um/px
elif camera == 'FLIR':
    pixSize = 3.75 #um/px
elif camera == 'Andor':
    pixSize = 6.5 #um/pix
#%%

df = pd.DataFrame(columns=['File', 'Condition', 'Value', 'Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])

if commonPhrase:

    conditions, values, distances = ImageAnalysisCode.RecognizeCommonPhrase(dataPath, repetition)

    df['Condition'] = conditions
    df['Value'] = values
    df['Distance'] = distances
    
#%%

fullpath = ImageAnalysisCode.GetFullFilePaths(dataPath)

if camera == 'Andor':
    metaData = ImageAnalysisCode.ExtractMetaData(fullpath)
else:
    metaData = None
    
images = ImageAnalysisCode.GetImages(fullpath, camera, ROI, metaData)
images_corrected = LSAnalysisCode.BGsubtraction_alt(images, 10)


# empty lists to store fitted parameters
Xcenters = []; Ycenters = []; Xwidths = []; Ywidths = []; Xamps = []; Yamps = []

for image_arr in images:
    
    image_arr, _ = ImageAnalysisCode.Rotate(image_arr, angle)
    paramX, paramY = ImageAnalysisCode.FitGaussian(image_arr, doPlot, 'Wide')
    
    Xcenter = paramX[0]*pixSize
    Xwidth = paramX[1]*pixSize
    
    Ycenter = paramY[0]*pixSize
    Ywidth = paramY[1]*pixSize
    
    Xcenters.append(Xcenter); Ycenters.append(Ycenter)
    Xwidths.append(Xwidth); Ywidths.append(Ywidth)
    Xamps.append(paramX[2]); Yamps.append(paramY[2])
        
df['Xcenter'] = Xcenters; df['Ycenter'] = Ycenters
df['Xwidth'] = Xwidths; df['Ywidth'] = Ywidths
df['Xamp'] = Xamps; df['Yamp'] = Yamps    

#%%

colsForAnalysis = ['Xwidth', 'Ywidth']

if df['Value'].isna().any():
    stats = df.groupby(['Distance'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
    stats.columns = ['Distance'] + ['_'.join(col).strip() for col in stats.columns[1:]]
else:
    stats = df.groupby(['Distance', 'Value'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
    stats.columns = ['Distance', 'Value'] + ['_'.join(col).strip() for col in stats.columns[2:]]



#%%

# for col in colsForAnalysis:
    
#     plt.figure(figsize=(4,3))
    
#     # for condition, group in stats.groupby('Value'):
#     #     plt.errorbar(group['Distance'], group[col+'_mean'], group[col+'_std'], fmt='o-', capsize=3, label=condition)
#     plt.errorbar(stats[var2plot], stats[col+'_mean'], stats[col+'_std'], fmt='-o', capsize=3)
    
#     plt.xlabel(quantity)
#     plt.ylabel(col)
#     # plt.legend(title='Power %')
#     plt.tight_layout()
    

# ImageAnalysisCode.FitGaussianWaist(stats, colsForAnalysis)

#%% MultiGaussian option
from scipy.signal import find_peaks, savgol_filter

if multiG is True:
    
    for img in images:    
        
        xSlice, ySlice = ImageAnalysisCode.GetSlices(img)
        
        xSlice_smooth = savgol_filter(xSlice, window_length=3, polyorder=2)
        
        peaks, props = find_peaks(xSlice_smooth,
                                  prominence=np.max(xSlice_smooth)*0.05,
                                  width=(1,3),
                                  distance=5
                                  )
        print('Detected peaks: ', len(peaks))
        print('Locations: ', xSlice_smooth[peaks])


