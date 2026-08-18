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

dataRootFolder = r"D:\Lehigh University Dropbox\Ariel Sommer\Sommer Lab Shared\Data"

date = '8/10/2026'

camera = 'Basler'

data_folder = [
    # fr'{camera}/CATauxBeamInitial_with100mmLens',
    fr'{camera}/CATauxBeamInitial_with100mmLens_117.5 mm'
    # fr'{camera}/LogPDpath 331 mm',
    # fr'{camera}/LogPDpath 349 mm',
    # fr'{camera}/LogPDpath 389 mm',
    ]


repetition = 6
commonPhrase = True
quantity = 'Distance (mm)'
var2plot = 'Distance'

doPlot = 1
angle = 0


rowstart=1
rowend=-1
columnstart=1
columnend=-1
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

df = pd.DataFrame(columns=['File', 'Condition', 'Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])

if commonPhrase:

    conditions, _, distances = ImageAnalysisCode.RecognizeCommonPhrase(dataPath, repetition)

    df['Condition'] = conditions
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

stats = df.groupby(['Distance'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
stats.columns = ['Distance'] + ['_'.join(col).strip() for col in stats.columns[1:]]



#%%
for col in colsForAnalysis:
    
    plt.figure(figsize=(4,3))
    plt.errorbar(stats[var2plot], stats[col+'_mean'], stats[col+'_std'], fmt='-o', capsize=3)
    
    plt.xlabel(quantity)
    plt.ylabel(col)
    plt.tight_layout()
    
# convert distance and waists to meters
stats['Distance'] = stats['Distance']*1e-3
width_cols = [col for col in stats.columns if 'width' in col]
stats[width_cols] = stats[width_cols] * 1e-6

ImageAnalysisCode.FitGaussianWaist(stats, colsForAnalysis)
