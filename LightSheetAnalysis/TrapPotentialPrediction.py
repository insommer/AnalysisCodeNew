import numpy as np
from matplotlib import pyplot as plt
import os
from ImageAnalysis import ImageAnalysisCode
from LightSheetAnalysis import LSAnalysisCode


plt.close('all')

dataRootFolder = r'C:\Users\wmmax\Documents\Lehigh\Sommer Group\Experiment Data'

date = '1/31/2025'

camera = 'FLIR'

data_folder = [
    fr'{camera}/AM 5.32 mW 99.92 mA',
    fr'{camera}/AM 8.36 mW 105.5 mA',
    fr'{camera}/AM 11.75 mW 111.53 mA',
    fr'{camera}/AM 14.5 mW 116.87 mA',
    fr'{camera}/AM 17.4 mW 122.8 mA',
    ]

rowstart=1
rowend=-1
columnstart=1
columnend=-1
ROI = [rowstart, rowend, columnstart, columnend]

dayFolder = ImageAnalysisCode.GetDayFolder(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]

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
    metaData = ImageAnalysisCode.ExtractMetaData(dataPath)
else:
    metaData = None    
    

df = LSAnalysisCode.ExtractRawCts(dataPath, ROI, metaData)


colsForGrouping = ['Power (W)', 'Current (mA)']
colsForAnalysis = ['TotalCts', 'MaxCts']

stats = df.groupby(colsForGrouping)[colsForAnalysis].agg(['mean','std']).reset_index()
stats.columns = colsForGrouping + ['_'.join(col).strip() for col in stats.columns[2:]]

#%% Calculate dipole potential

# fit cts vs. power, get conversion factor
factor = LSAnalysisCode.TotalCts2power(stats)

# Calculate intensity of image, fit intensity vs. current
stats = LSAnalysisCode.GetMaxIntensity(dataPath, stats, pixArea_m, factor)

# dipole potential
Udip, Temp_uK = LSAnalysisCode.GetDipolePotential(stats)