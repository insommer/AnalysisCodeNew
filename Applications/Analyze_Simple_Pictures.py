from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import pandas as pd
import os
from scipy import constants
import glob
import datetime
import DMDanalysis

DA = DMDanalysis.DMDanalysis()

paths = [
    r'D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data\2025\02-2025\27 Feb 2025\FLIR\Second pass evap LowServo 0.5 V',
    ]

PPI = 1
doPlot = 0
angle = 0
pixSize = 3.75 #um/px


rowstart=900
rowend=1100
columnstart=750
columnend=1050


filetype = '.pgm'
examRange = [None, None]
filterLists = [[]]
loadVariableLog = 1
rebuildCatalogue = 0
skipFirstImg = 0


# Go throught the folders, examine if No of pictures are correct, and if buidling a catalogue is needed. 
N = 0 # Totol number of images in all provided folders. 
pathNeedCatalogue = []
catalogue = []
    
for path in paths:
    if not os.path.exists(path):
        print("Warning! Data folder not found:" + str(path))
        continue

    number_of_pics = len( glob.glob1(path, '*' + filetype) )
    if number_of_pics == 0:
        print('Warning!\n{}\ndoes not contain any data file!'.format(path))
    elif number_of_pics % PPI:
        raise Exception('The number of data files in\n{}\nis not correct!'.format(path))

    cataloguePath = os.path.join(path, 'Catalogue.pkl')
    existCatalogue = os.path.exists(cataloguePath)

    if loadVariableLog and (rebuildCatalogue or not existCatalogue): # If the catalogue not exist or need to be rebuilt, keep the path 
        pathNeedCatalogue.append(path)

    elif existCatalogue: # Load the catalogue otherwise.        
        df = pd.read_pickle(cataloguePath)

        # If the lengh of the catalogue is different from the iteration number, determine if rebuild it or not.
        if (len(df) != (number_of_pics / PPI)):
            # If current time is 12 hours or 7 days later than the data were took, prevent auto rebuild the catalogue. 
            dt = datetime.datetime.now() - df.index[0]

            if (df.PPI[0] != PPI) or (df.SkipFI[0] != skipFirstImg):
                if dt > pd.Timedelta(0.5, "d"):
                    raise ValueError('The input of subtract_burntin or skipFirstImg does not match the record!\nCorrect the input or set rebuildCatalogue to 1 to force rebuild the catalogue.')
            else:
                if dt > pd.Timedelta(7, "d"):
                    raise ValueError('The number of files in {}\nis different from recorded, set rebuildCatalogue to 1 to force rebuild the catalogue.')
            # Rebuild the catalogue otherwise.        
            pathNeedCatalogue.append(path)
        # Add the folder path to the datalogue and load it.                
        else:
            df['FolderPath'] = path                
            catalogue.append( df )                

    N += number_of_pics        
if N == 0:
    raise Exception('No data file was found in all provided folders!')

# Build the catalogue for the folders that need one, and append to the loaded ones.         
if loadVariableLog and pathNeedCatalogue: 
    catalogue.extend( ImageAnalysisCode.BuildCatalogue(*pathNeedCatalogue, cam='cha',
                                     picturesPerIteration=PPI, skipFirstImg=skipFirstImg,
                                     dirLevelAfterDayFolder=2) )
    
catalogue = ImageAnalysisCode.DataFilter(pd.concat(catalogue), filterLists=filterLists)[examRange[0]: examRange[1]]

if len(catalogue) == 0:
    raise ValueError('Len(Catalogue) is ZERO! No item satisfy the conditions!')

dfpaths = catalogue[['FolderPath', 'FirstImg']]

imgPaths = ImageAnalysisCode.FillFilePathsListFLIR(dfpaths, PPI)        
rawImgs = ImageAnalysisCode.loadSeriesPGMV2(imgPaths, file_encoding='binary')
rawImgs = rawImgs.reshape( -1, PPI, *rawImgs.shape[-2:] )


# analyze raw images with Gaussian
catalogue[['Xcenter', 'Ycenter', 'Xwidth', 'Ywidth']]=np.nan

for index, img in enumerate(rawImgs):
    img = img.squeeze()

    paramX, paramY = DA.FitGaussian(img[rowstart:rowend, columnstart:columnend], doPlot)

    Xcenter = paramX[0]*pixSize
    Xwidth = paramX[1]*pixSize

    Ycenter = paramY[0]*pixSize
    Ywidth = paramY[1]*pixSize
    
    rowInd = catalogue.index[index]
    
    catalogue.loc[rowInd, 'Xcenter'] = Xcenter
    catalogue.loc[rowInd, 'Ycenter'] = Ycenter
    catalogue.loc[rowInd, 'Xwidth'] = Xwidth
    catalogue.loc[rowInd, 'Ywidth'] = Ywidth
    
#%%
ImageAnalysisCode.PlotResults(catalogue, 'wait', 'Xcenter', 
                                    groupbyX=1, 
                                  threeD=0,
                                  figSize = 0.5
                                  )

ImageAnalysisCode.PlotResults(catalogue, 'wait', 'Ycenter', 
                                    groupbyX=1, 
                                  threeD=0,
                                  figSize = 0.5
                                  )

    
    
    