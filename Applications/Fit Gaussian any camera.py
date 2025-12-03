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

if camera == 'Basler':
    pixSize = 2 #um/px
elif camera == 'FLIR':
    pixSize = 3.75 #um/px
elif camera == 'Andor':
    pixSize = 6.5 #um/pix

#%%

def GetDataLocation(Date, RootFolder):
    path = os.path.join(RootFolder, datetime.datetime.strptime(Date, '%m/%d/%Y').strftime('%Y/%m-%Y/%d %b %Y'))
    return path

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
        return None
        
    
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

def CheckFile(file):
    if isinstance(file, str) and file.endswith('.png'):  
        arr = np.array(Image.open(file).convert('L'))
        
    elif isinstance(file, np.ndarray):
        arr = file
        
    elif isinstance(file, str) and file.endswith('.raw'):
        temp = np.fromfile(file, dtype = np.uint8)
        arr = np.reshape(temp, (2160,3840)) # Basler dart resolution
        
    elif isinstance(file, str) and file.endswith('.pgm'):
        img = Image.open(file)
        arr = np.asarray(img, dtype=np.uint16)
        rows, cols = np.shape(arr)
        rows2discard = 2
        arr = arr[rows2discard:, :]    
    else:
        raise ValueError(f"Unsupported file type or input: {file}")
    
    return arr

def GetImages(dataPath_list, camera, ROI, metadata=None):
    
    imgs = []
    
    if camera == 'Andor':
        # for Andor files, remove first 40 bytes
        headerbytes = 40
        height = metadata[0]
        width = metadata[1]
        
        binFile = [f for f in dataPath_list if f.lower().endswith('.dat')]
        
        for path in binFile:
            with open(path, 'rb') as f:
                f.seek(headerbytes)
                img_data = np.frombuffer(f.read(), dtype=np.uint16)
            
            expectedsize = height*width
            if img_data.size != expectedsize:
                raise ValueError(f'has {img_data.size} pixels, but expected {expectedsize}')
    
            imgArr = img_data.reshape((height,width))
            imgArr = imgArr[ROI[0]:ROI[1], ROI[2]:ROI[3]]
            
            imgs.append(imgArr)
            
    else:
        for file in dataPath_list:
            imgArr = CheckFile(file)
            imgArr = imgArr[ROI[0]:ROI[1], ROI[2]:ROI[3]]

            imgs.append(imgArr)

    return imgs

def Rotate(image_arr, deg):
    height, width = image_arr.shape[:2]
    
    center = (width / 2, height / 2)
    
    rotationMatrix = cv2.getRotationMatrix2D(center, deg, 1.0)
    rotated = cv2.warpAffine(image_arr, rotationMatrix, (width, height))
    
    return rotated, rotationMatrix

def Gauss1D(x,xc,sigX,A, offset):
    G = A * np.exp(-2 * (x-xc)**2 / sigX**2) + offset
    return G

def FitGaussian(gaussImageFile, graph=True, graphOption='Wide'):
    
    beam = CheckFile(gaussImageFile)
    Ny, Nx = beam.shape
    x_index = np.linspace(0, Nx-1, Nx)
    y_index = np.linspace(0, Ny-1, Ny)
    
    max_index = np.unravel_index(np.argmax(beam), beam.shape)
    max_x, max_y = max_index

    vert = beam[:, max_y]
    horiz = beam[max_x, :]
    
    sigGuess = 40
    offset = 0
    
    guessX = [max_y, sigGuess, np.max(horiz), offset]
    paramX,_ = curve_fit(Gauss1D, x_index, horiz, p0=guessX)
    x_fit1 = np.linspace(0, Nx-1, 5000)
    y_fit1 = Gauss1D(x_fit1, paramX[0], paramX[1], paramX[2], paramX[3])

    guessY = [max_x, sigGuess, np.max(vert), offset]
    paramY,_ = curve_fit(Gauss1D, y_index, vert, p0=guessY)
    x_fit2 = np.linspace(0,Ny-1, 5000)
    y_fit2 = Gauss1D(x_fit2, paramY[0], paramY[1], paramY[2], paramY[3])
    
    centerX = int(paramX[0])
    centerY = int(paramY[0])
            
    if graph:
        fig, ax = plt.subplots(1,3)
        
        ax[1].plot(x_fit1, y_fit1,'r',linewidth=3)
        ax[1].scatter(x_index, horiz, s=20)
        ax[1].set_title('Fit vs. X')
        
        text_x = f"x0 = {int(paramX[0])} \nσ = {paramX[1]:.2f} px \nA = {paramX[2]:.2f}"
        ax[1].text(0.35, 0.95, text_x, transform=ax[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        ax[2].plot(x_fit2,y_fit2,'r',linewidth=3)
        ax[2].scatter(y_index, vert, s=20)
        ax[2].set_title('Fit vs. Y')
        
        text_y = f"y0 = {int(paramY[0])} \nσ = {paramY[1]:.2f} px \nA = {paramY[2]:.2f}"
        ax[2].text(0.05, 0.95, text_y, transform=ax[2].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        if graphOption == 'Narrow':
            
            ax[1].set_xlim(paramX[0]-4*paramX[1], paramX[0]+4*paramX[1])
            ax[2].set_xlim(paramY[0]-4*paramY[1], paramY[0]+4*paramY[1])
            
            ax[0].imshow(beam, extent=[paramX[0]-1, 
                                       paramX[0]+1, 
                                       paramY[0]-1, 
                                       paramY[0]+1])
        else:
            ax[0].imshow(beam*-1,cmap='binary')
            ax[0].imshow(beam,cmap='jet')
        
        ax[0].set_title('Image')        
    return paramX, paramY

#%%

dayFolder = GetDataLocation(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]

df = pd.DataFrame(columns=['File', 'Condition', 'Value', 'Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])
Xcenters = []; Ycenters = []
Xwidths = []; Ywidths = []
Xamps = []; Yamps = []

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

fullpath = GetFullFilePaths(dataPath)

if camera == 'Andor':
    metaData = ExtractMetaData(fullpath)
else:
    metaData = None
    
images = GetImages(fullpath, camera, ROI, metaData)



for image_arr in images:
    
    image_arr, _ = Rotate(image_arr, angle)
    paramX, paramY = FitGaussian(image_arr, doPlot, 'wide')
    
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
    
#%%

def FitGaussianWaist(stats, colsForAnalysis, doPlot=True):

    def w_z(z, w0, z0, zR):
        return w0 * np.sqrt(1 + ((z - z0) / zR)**2)
    
    # assumes distances in mm, waist in um
    for col in colsForAnalysis:
        z = stats['Distance'].values
        w_meas = stats[col+'_mean'].values
        w_err = stats[col+'_std'].values
        
        p0 = [min(w_meas), z[np.argmin(w_meas)], (max(z) - min(z)) / 2]
        
        popt, pcov = curve_fit(w_z, z, w_meas, p0=p0, sigma=w_err, absolute_sigma=True)
        
        w0_fit, z0_fit, zR_fit = popt
        perr = np.sqrt(np.diag(pcov))
        
        if doPlot:
            z_fit = np.linspace(min(z), max(z), 300)
            fig, ax = plt.subplots(figsize=(4,3))
            ax.errorbar(z, w_meas, yerr=w_err, fmt='o', capsize=3)
            ax.plot(z_fit, w_z(z_fit, *popt), 'r-')
            ax.set_xlabel('Distance (mm)')
            ax.set_ylabel(col+' (μm)')
            ax.text(0.3, 0.85, f'w0={w0_fit:.2f} μm\nz0={z0_fit:.2f} mm', transform=ax.transAxes, bbox=dict(facecolor='white'))
            plt.tight_layout()
            
FitGaussianWaist(stats, colsForAnalysis)



