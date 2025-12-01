from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import rotate
import pandas as pd
import os
import DMDanalysis

plt.close('all')

DA = DMDanalysis.DMDanalysis()
#%%

# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'F:\Data'
date = '3/25/2025'

data_folder = [
    # r'Andor/RF spec Load D1 into ODT wait 0 ms_1',
    # r'Andor/RF spec Load D1 into ODT wait 1 ms',
    # r'Andor/RF spec Load D1 into ODT wait 2 ms',
    # r'Andor/RF spec Load D1 into ODT wait 5 ms_1',
    # r'Andor/RF spec Load D1 into ODT wait 7.5 ms_1',
    # r'Andor/RF spec Load D1 into ODT wait 10 ms',
    
    
    # r'Andor/1 ms wait D1 rf spec',
    # r'Andor/2 ms wait D1 rf spec',

    ]

reanalyze = 1
saveresults = 0
overwriteOldResults = 1

repetition = 1 #The number of identical runs to be averaged.
subtract_burntin = 1

skipFirstImg = 'auto'

rotateAngle = 0 #rotates ccw

examNum = None #The number of runs to exam.
examFrom = None #Set to None if you want to check the last several runs. 
showRawImgs = 0

variableFilterList = [
    ] 

pictureToHide = None

subtract_bg = 0
signal_feature = 'wide' 
signal_width = 10 #The narrower the signal, the bigger the number.
fitbgDeg = 5

# rowstart = 10
# rowend = -10
# columnstart = 10
# columnend = -10

rowstart = 300
rowend = 700
columnstart=300
columnend= 1700

dayfolder = ImageAnalysisCode.GetDataLocation(date, DataPath=dataRootFolder)
dataPath = [ os.path.join(dayfolder, f) for f in data_folder]

# variableLog_folder = dayFolder + r'/Variable Logs'
examFrom, examUntil = ImageAnalysisCode.GetExamRange(examNum, examFrom, repetition)

params = ImageAnalysisCode.ExperimentParams(date, t_exp = 10e-6, picturesPerIteration=None, axis='side', cam_type = "zyla")
dxMicron = params.camera.pixelsize_microns/params.magnification   #The length in micron that 1 pixel correspond to. 
dxMeter = params.camera.pixelsize_meters/params.magnification

#%%
opticalDensity, variableLog = ImageAnalysisCode.PreprocessZylaImg(*dataPath, examRange=[examFrom, examUntil], 
                                                                   rotateAngle=rotateAngle, 
                                                                   rowstart=rowstart, rowend=rowend, 
                                                                   columnstart=columnstart, columnend=columnend,
                                                                   subtract_burntin=subtract_burntin, 
                                                                   skipFirstImg=skipFirstImg, 
                                                                   showRawImgs=showRawImgs, 
                                                                   #!!!!!!!!!!!!!!!!!
                                                                   #! Keep rebuildCatalogue = 0 unless necessary!
                                                                   rebuildCatalogue=0,
                                                                   ##################
                                                                    # filterLists=[['TOF<1']]
                                                                    )

#%%

columnDensities = opticalDensity / params.cross_section        
popts, bgs = ImageAnalysisCode.FitColumnDensity(columnDensities, dx = dxMicron, mode='both', yFitMode='single',
                                                subtract_bg=subtract_bg, Xsignal_feature='wide', Ysignal_feature='wide')

results = ImageAnalysisCode.AnalyseFittingResults(popts, logTime=variableLog.index) 



if variableLog is not None:
    results = results.join(variableLog)

if saveresults:
    ImageAnalysisCode.SaveResultsDftoEachFolder(results, overwrite=overwriteOldResults)

#%%
intermediatePlot = 1
plotPWindow = 5
plotRate = 1
uniformscale = 0
rcParams = {'font.size': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
            # 'image.interpolation': 'nearest'
            }

variablesToDisplay = [
                        'wait',
                        'RF_FRQ_MHz',
                      ]
showTimestamp = False
# variablesToDisplay=None
textY = 1
textVA = 'bottom'

if intermediatePlot:
    # ImageAnalysisCode.ShowImagesTranspose(images_array, uniformscale=False)
    ImageAnalysisCode.plotImgAndFitResult(columnDensities, popts, bgs=bgs, 
                                          dx=dxMicron, 
                                          # imgs2=opticalDensity, 
                                          # addColorbar=0,
                                            # filterLists=[['LowServo1==0.5']],
                                           plotRate=plotRate, plotPWindow=plotPWindow,
                                            variablesToDisplay = variablesToDisplay,
                                           showTimestamp=showTimestamp,
                                          variableLog=variableLog, 
                                          logTime=variableLog.index,
                                          uniformscale=uniformscale,
                                          textLocationY=0.9, rcParams=rcParams,
                                          figSizeRate=1, sharey='col')

#%% Plot AtomNumber vs. Freq
fig,ax = plt.subplots(2,1)

groupby_str = 'wait'
atomNum_str1 = 'XatomNumber'
atomNum_str2 = 'YatomNumber'

for wait_value, group in results.groupby(groupby_str):
    
    group = group.sort_values(by='RF_FRQ_MHz')
    ax[0].plot(group['RF_FRQ_MHz'], group[atomNum_str1], marker='o', linestyle='-', label=f'{groupby_str}={wait_value}')
    ax[1].plot(group['RF_FRQ_MHz'], group[atomNum_str2], marker='o', linestyle='-', label=f'{groupby_str}={wait_value}')


ax[1].set_xlabel('RF_FRQ_MHz')

ax[0].set_ylabel(atomNum_str1)
ax[1].set_ylabel(atomNum_str2)
ax[0].legend()

#%% Fit Gaussian
wait_ms = []
centerFreq = []
resonanceWidth = []

plt.figure(figsize=(5,4))
for wait_value, group in results.groupby(groupby_str):
    
    group = group.sort_values(by='RF_FRQ_MHz')  
    
    
    params, FitArr = DA.FitGaussian_1D(group['RF_FRQ_MHz'].values, group[atomNum_str2].values, graph=False)
    
    plt.plot(FitArr[0],FitArr[1], label=f'{groupby_str}={wait_value}')
    plt.scatter(group['RF_FRQ_MHz'], group[atomNum_str2])

    
    wait_ms.append(wait_value)
    centerFreq.append(params[0])
    resonanceWidth.append(params[1])
plt.xlabel('RF_FRQ_MHz')
plt.ylabel(atomNum_str2)
plt.legend()
plt.tight_layout()


plt.figure(figsize=(4,3))
plt.plot(wait_ms, centerFreq, '-o')

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
plt.xlabel(groupby_str)
plt.ylabel('Center Freq (MHz)')
plt.tight_layout()

#%% Linear fit
from scipy.optimize import curve_fit


def linear(x,m,b):
    return m*x + b

slope_guess = (centerFreq[0]-centerFreq[1]) / (wait_ms[0] - wait_ms[1])

guess = [slope_guess, 228.2]
lin_params, _ = curve_fit(linear, wait_ms, centerFreq, p0=guess)

xFit_lin = np.linspace(wait_ms[0], wait_ms[-1], 100)
yFit_lin = linear(xFit_lin, *lin_params)

plt.figure(figsize=(4,3))
plt.scatter(wait_ms,centerFreq)
plt.plot(xFit_lin, yFit_lin,'r')
plt.xlabel('wait (ms)')
plt.ylabel('Center frequency (MHz)')

text = 'm = '+str(round(lin_params[0],7))+' MHz/ms\nb = '+str(round(lin_params[1],5))+' MHz'
plt.text(2, 228.2225, text)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
plt.tight_layout()

    