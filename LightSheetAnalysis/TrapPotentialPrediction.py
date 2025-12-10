import numpy as np
from matplotlib import pyplot as plt
import os
from ImageAnalysis import ImageAnalysisCode
from LightSheetAnalysis import LSAnalysisCode


plt.close('all')

dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
# dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'

date = '12/10/2025'

camera = 'Andor'

data_folder = [
    # fr'{camera}/Focused LS 4.1 mW 89.5 mA',
    # fr'{camera}/Focused LS 8.8 mW 109.6 mA',
    # fr'{camera}/Focused LS 13.45 mW 130.5 mA',
    # fr'{camera}/Two LS 10.45 mW 93.6 mA',
    # fr'{camera}/Two LS 18.18 mW 109.7 mA',
    # fr'{camera}/Two LS 28.3 mW 130.5 mA',
    fr'{camera}/Reflected LS 5.71 mW 93.6 mA',
    fr'{camera}/Reflected LS 10.5 mW 111.7 mA',
    fr'{camera}/Reflected LS 15.26 mW 130.8 mA',

    ]

rowstart=720
rowend=840
columnstart=200
columnend=500
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
images_corrected = LSAnalysisCode.BGsubtraction_alt(images, 10)

df = LSAnalysisCode.ExtractRawCts(fullPath, images_corrected)

colsForGrouping = ['Power (W)', 'Current (mA)']
colsForAnalysis = ['TotalCts', 'MaxCts']

stats = df.groupby(colsForGrouping)[colsForAnalysis].agg(['mean','std']).reset_index()
stats.columns = colsForGrouping + ['_'.join(col).strip() for col in stats.columns[2:]]


#%% Calculate dipole potential

# fit cts vs. power, get conversion factor
factor = LSAnalysisCode.TotalCts2power(stats)
# factor = LSAnalysisCode.MaxCts2power(stats)

# calculate intensity from the image
df = LSAnalysisCode.GetMaxIntensity(images_corrected, df, pixArea_m, factor)
stats['Max Intensity (W/m2)'] = df.groupby(colsForGrouping)['Max Intensity (W/m2)'].mean().reset_index()['Max Intensity (W/m2)']

# calculate dipole potential, fit vs. laser current
Udip, Temp_uK = LSAnalysisCode.GetDipolePotential(stats, maxCurrent_mA=270)