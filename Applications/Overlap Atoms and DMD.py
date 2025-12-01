import sys
import os
import glob
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from scipy.integrate import simpson
from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import rotate
import DMDanalysis

DA = DMDanalysis.DMDanalysis()

plt.close('all')

####################################
#Set the date and the folder name
####################################

date = '2/25/2025'
data_path = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataLocation = ImageAnalysisCode.GetDataLocation(date, DataPath=data_path)

data_folders =[
    r'/FLIR/Evap cloud motion MF 0.5 V long',
    r'/FLIR/Test DMD'
    ]

most_recent = True

data_folder_atoms = dataLocation + data_folders[0]
data_folder_DMD = dataLocation + data_folders[1]

####################################
#Parameter Setting

examNum = 3 #The number of runs to exam.
examFrom = None #Set to None if you want to check the last several runs. 
examFrom, examUntil = ImageAnalysisCode.GetExamRange(examNum, examFrom)

rowstart = 750
rowend = 1150
columnstart = 650
columnend = 1350

# rowstart=1
# rowend=-1
# columnstart=1
# columnend=-1

binsize=1
####################################
####################################
    
t_exp = 10e-6
picturesPerIteration = 3


params = ImageAnalysisCode.ExperimentParams(date, 
                                            t_exp = t_exp, 
                                            picturesPerIteration= picturesPerIteration, 
                                            axis='side', 
                                            cam_type = 'chameleon')

images_array, _ = ImageAnalysisCode.loadSeriesPGM(picturesPerIteration=picturesPerIteration, 
                                                  data_folder = data_folder_atoms, 
                                                  binsize=binsize, 
                                                  file_encoding = 'binary', 
                                                  examFrom=examFrom, 
                                                  examUntil=examUntil, 
                                                  return_fileTime=1)

_, _, _, columnDensities, _, _ = ImageAnalysisCode.absImagingSimple(images_array, 
                                                                    params=params,
                                                                    firstFrame=0, correctionFactorInput=1, 
                                                                    rowstart = rowstart, rowend = rowend, 
                                                                    columnstart = columnstart, columnend = columnend, 
                                                                    subtract_burntin=0, 
                                                                    preventNAN_and_INF=True
                                                                    )

if most_recent:
    
    files = sorted(glob.glob(os.path.join(data_folder_DMD, "*")), key=os.path.getctime, reverse=True)

    if files:
        most_recent_path = files[0]
        DMDimg = DA.CheckFile(most_recent_path)
        croppedDMD = DMDimg[rowstart:rowend, columnstart:columnend]

else:
    
    DMDimages = []
    for file in os.listdir(data_folder_DMD):
        
        path = os.path.join(data_folder_DMD, file)
        
        DMDimg = DA.CheckFile(path)
        croppedDMD = DMDimg[rowstart:rowend, columnstart:columnend]
        DMDimages.append(croppedDMD)



for count, img in enumerate(columnDensities):
        
    croppedCD = columnDensities[count][rowstart:rowend, columnstart:columnend]
    vmax = croppedCD.max()
    
    plt.figure(figsize=(6,4))    
    if most_recent:
        plt.imshow(croppedDMD, cmap = 'jet', vmin=0, vmax=croppedDMD.max(), alpha = 0.5)
    
    else:
        for k in DMDimages:
            plt.imshow(k, cmap = 'jet', vmin=0, vmax=k.max(), alpha = 0.5)
    
    plt.imshow(croppedCD, cmap='jet', vmin=0, vmax=vmax, alpha = 0.5)

