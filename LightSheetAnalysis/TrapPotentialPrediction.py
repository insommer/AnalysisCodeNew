import numpy as np
from matplotlib import pyplot as plt
import os
from ImageAnalysis import ImageAnalysisCode
from LightSheetAnalysis import LSAnalysisCode


plt.close('all')

# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'

date = '12/9/2025'
# date = '1/31/2025'

camera = 'Andor'

data_folder = [
    fr'{camera}/One LS 0.75 mW 75.7 mA',
    fr'{camera}/One LS 2.1 mW 81.6 mA',
    fr'{camera}/One LS 3.3 mW 87 mA',
    fr'{camera}/One LS 4.65 mW 93 mA',
    # fr'{camera}/AM 5.32 mW 99.92 mA',
    # fr'{camera}/AM 8.36 mW 105.5 mA',
    # fr'{camera}/AM 11.75 mW 111.53 mA',
    # fr'{camera}/AM 14.5 mW 116.87 mA',

    
    ]

rowstart=1
rowend=-1
columnstart=1
columnend=-1
ROI = [rowstart, rowend, columnstart, columnend]

dayFolder = ImageAnalysisCode.GetDayFolder(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]
fullPath = ImageAnalysisCode.GetFullFilePaths(dataPath)

if camera == 'FLIR':
    pixSize_um = 3.75
    w = 2048
    h = 1536
    
elif camera == 'Basler':
    pixSize_um = 2
    w = 3840
    h = 2160
elif camera == 'Andor':
    pixSize_um = 6.5

pixArea_m = (pixSize_um * 1e-6) ** 2 # pixel area [meter^2]


#%% Extract counts from images

if camera == 'Andor':
    metaData = ImageAnalysisCode.ExtractMetaData(fullPath)
else:
    metaData = None
    
images = LSAnalysisCode.ExtractImages(fullPath, ROI, metaData)

bgVal = LSAnalysisCode.EstimateBGvalue(images[0])
images_corrected = [img - bgVal for img in images]

df = LSAnalysisCode.ExtractRawCts(fullPath, images_corrected)


colsForGrouping = ['Power (W)', 'Current (mA)']
colsForAnalysis = ['TotalCts', 'MaxCts']

stats = df.groupby(colsForGrouping)[colsForAnalysis].agg(['mean','std']).reset_index()
stats.columns = colsForGrouping + ['_'.join(col).strip() for col in stats.columns[2:]]


#%% Calculate dipole potential

# fit cts vs. power, get conversion factor
factor = LSAnalysisCode.TotalCts2power(stats)

# calculate intensity from the image
df = LSAnalysisCode.GetMaxIntensity(images_corrected, df, pixArea_m, factor)
stats['Max Intensity (W/m2)'] = df.groupby(colsForGrouping)['Max Intensity (W/m2)'].mean().reset_index()['Max Intensity (W/m2)']

# calculate dipole potential, fit vs. laser current
Udip, Temp_uK = LSAnalysisCode.GetDipolePotential(stats)