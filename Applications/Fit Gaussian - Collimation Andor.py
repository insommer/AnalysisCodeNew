import DMDanalysis
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.optimize import curve_fit
from ImageAnalysis import ImageAnalysisCode
import configparser


plt.close('all')

DA = DMDanalysis.DMDanalysis()

dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
date = '11/9/2025'

# camera = 'FLIR'
camera = 'Andor'
data_folder = [
    fr'{camera}/Test with image_3'
    ]

# repetition = 6
# commonPhrase = True
# quantity = 'Distance (mm)'

rowstart=1
rowend=-1
columnstart=1
columnend=-1


dayFolder = DA.GetDataLocation(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]

if camera == 'Basler':
    pixSize = 2 #um/px
elif camera == 'FLIR':
    pixSize = 3.75 #um/px
elif camera == 'Andor':
    pixSize = 6.5 #um/pix
    

#%%

def GetFullFilePaths(dataPath_list):

    fullpath = []
    
    for folder in dataPath:
        folder = folder+'/'
        
        for filename in os.listdir(folder):
            
            path = folder+filename
            fullpath.append(path)
    
    return fullpath

def ExtractMetaData(filePaths):

    metaFile = next( (f for f in filePaths if f.lower().endswith('.ini')), None)
    
    if metaFile is None:
        raise FileNotFoundError('No metadata file found')
        
    
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(metaFile, encoding='utf-8-sig')
    
    
    height = int(config['data']['AOIHeight'])
    width = int(config['data']['AOIWidth'])
    pixFormat = config['data']['PixelEncoding']
    if pixFormat.lower() == 'mono16':
        dataType = np.uint16
    
    pixNumber = height*width
    
    return height, width, pixNumber, pixFormat, dataType

def GetImages(dataPath_list, metadata=None):
    
    # for Andor files, remove first 40 bytes
    headerbytes = 40
    height = metadata[0]
    width = metadata[1]
    
    binFile = [f for f in dataPath_list if f.lower().endswith('.dat')]

    imgs = []

    for path in binFile:
        with open(path, 'rb') as f:
            f.seek(headerbytes)
            img_data = np.frombuffer(f.read(), dtype=np.uint16)
        
        expectedsize = height*width
        if img_data.size != expectedsize:
            raise ValueError(f'has {img_data.size} pixels, but expected {expectedsize}')

        img = img_data.reshape((height,width))
        imgs.append(img)
    
    return imgs
    

fullpath = GetFullFilePaths(dataPath)
metaData = ExtractMetaData(fullpath)
images = GetImages(fullpath, metaData)

#%%
Xcenters = []; Ycenters = []
Xwidths = []; Ywidths = []
Xamps = []; Yamps = []

df = pd.DataFrame(columns=['Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])


#%%
doPlot = 1
angle = 0

for image_arr in images:
    
    image_arr, _ = DA.Rotate(image_arr, angle)
        
    paramX, paramY = DA.FitGaussian(image_arr[rowstart:rowend, columnstart:columnend], doPlot)
    
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

    