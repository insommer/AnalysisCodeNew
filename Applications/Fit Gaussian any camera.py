import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.optimize import curve_fit
from ImageAnalysis import ImageAnalysisCode
import datetime
import configparser
from PIL import Image
import cv2


plt.close('all')

# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'
date = '12/1/2025'

camera = 'Basler'
powr = 15
# camera = 'Andor'
data_folder = [

    fr'{camera}/SPX023AR.1 110 mm power {powr}',
    fr'{camera}/SPX023AR.1 113 mm power {powr}',
    fr'{camera}/SPX023AR.1 117 mm power {powr}',
    fr'{camera}/SPX023AR.1 121 mm power {powr}',
    fr'{camera}/SPX023AR.1 124 mm power {powr}',
    fr'{camera}/SPX023AR.1 128 mm power {powr}',
    fr'{camera}/SPX023AR.1 132 mm power {powr}',

    ]

repetition = 6
commonPhrase = True
quantity = 'Distance (mm)'
var2plot = 'Distance'

doPlot = 0
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

df = pd.DataFrame(columns=['File', 'Condition', 'Value', 'Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])

if commonPhrase:

    pattern_both = re.compile(r'(?:(\d+(?:\.\d+)?)\s*mm).*?power\s*(\d+)$', re.IGNORECASE)
    pattern_distance_only = re.compile(r'(\d+(?:\.\d+)?)\s*mm', re.IGNORECASE)

    conditions = []
    values = []
    distances = []

    for name in dataPath:
        basename = os.path.basename(name)

        match_both = pattern_both.search(basename)
        match_dist = pattern_distance_only.search(basename)

        if match_both:
            distance = float(match_both.group(1))
            value = int(match_both.group(2))
            condition = re.sub(pattern_both, '', basename).strip()
        elif match_dist:
            distance = float(match_dist.group(1))
            value = np.nan
            condition = re.sub(pattern_distance_only, '', basename).strip()
        else:
            distance = np.nan
            value = np.nan
            condition = basename.strip()

        conditions.extend([condition] * repetition)
        values.extend([value] * repetition)
        distances.extend([distance] * repetition)

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

# empty lists to store fitted parameters
Xcenters = []; Ycenters = []; Xwidths = []; Ywidths = []; Xamps = []; Yamps = []

for image_arr in images:
    
    image_arr, _ = ImageAnalysisCode.Rotate(image_arr, angle)
    paramX, paramY = ImageAnalysisCode.FitGaussian(image_arr, doPlot, 'wide')
    
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

for col in colsForAnalysis:
    
    plt.figure(figsize=(4,3))
    
    # for condition, group in stats.groupby('Value'):
    #     plt.errorbar(group['Distance'], group[col+'_mean'], group[col+'_std'], fmt='o-', capsize=3, label=condition)
    plt.errorbar(stats[var2plot], stats[col+'_mean'], stats[col+'_std'], fmt='-o', capsize=3)
    
    plt.xlabel(quantity)
    plt.ylabel(col)
    # plt.legend(title='Power %')
    plt.tight_layout()
    

ImageAnalysisCode.FitGaussianWaist(stats, colsForAnalysis)



