# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:34:22 2023

@author: Sommer Lab
"""
from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import pandas as pd
import os
from scipy import constants

####################################
#Set the date and the folder name
####################################
dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
# date = '10/07/2024'
date = '10/9/2024'
# date = '9/9/2024'

data_folder = [
    # r'Andor/Aspect ratio check',
    r'Andor/Lifetime misaligned WP 219',
    r'Andor/Lifetime misaligned WP 220_1',
    # r'Andor/lifetime Evap1_V 0.35V',
    # r'Andor/Lifetime WP 217_1',
    # r'Andor/Modulate ODT 1250_1',
    # r'Andor/1250 atom number at BEC field',
    # r'Andor/ODT 2350 scan LowServo Atom Num',
    # r'Andor/ODT 2350 scan LowServo Atom Num_1',
    # r'Andor/Modulate ODT 1800 low freq_1',
    # r'Andor\gray molasses more D1 power_1'
    # r'Andor\With Low Servo Late ODT LowServo1 0.29 Changed Lens_1',
    # r'Andor\before evap thermometry low field Digital Mag Off',
    # r'Andor\GM Temp Vary D1 Re Attn_3',
    ]
####################################
#Parameter Setting'
####################################
reanalyze = 1
saveresults = 0
overwriteOldResults = 0

repetition = 1 #The number of identical runs to be averaged.
subtract_burntin = 0

skipFirstImg = 'auto'
# skipFirstImg = 0

rotateAngle = 0 #rotates ccw
# rotateAngle = 1.5 #rotates ccw

examNum = None #The number of runs to exam.
examFrom = None #Set to None if you want to check the last several runs. 
showRawImgs = 0

# Set filters for the data, NO SPACE around the operator.
variableFilterList = [
    # [# 'wait==50', 
    # # 'VerticalBiasCurrent==0'
    # 'fmod_kHz==0',
    # # 'Evap_Tau==0.1',
    # # 'Evap_Time_1==2'
    # ], 
    # [
    # 'TOF==0',
    # 'Evap_Tau==0.1',
    # 'Evap_Time_1==2']
    ] 

pictureToHide = None
# pictureToHide = [0,1,2,3] # list(range(0,10,2))

subtract_bg = 0
signal_feature = 'wide' 
signal_width = 10 #The narrower the signal, the bigger the number.
fitbgDeg = 5

rowstart = 0
rowend = -1
columnstart = 0
columnend = -1


# columnstart=730
# columnend= 1250

# rowstart = 250
# rowend = 700

# first pass
# rowstart = 400
# rowend = 450

# second pass
rowstart = 440
rowend = 500

# rowstart -= 180
# rowend -= 180

# columnstart -= 100
# columnend -= 100

####################################
####################################
dayfolder = ImageAnalysisCode.GetDataLocation(date, DataPath=dataRootFolder)
dataPath = [ os.path.join(dayfolder, f) for f in data_folder]

# variableLog_folder = dayFolder + r'/Variable Logs'
examFrom, examUntil = ImageAnalysisCode.GetExamRange(examNum, examFrom, repetition)

params = ImageAnalysisCode.ExperimentParams(date, t_exp = 10e-6, picturesPerIteration=None, cam_type = "zyla")
dxMicron = params.camera.pixelsize_microns/params.magnification   #The length in micron that 1 pixel correspond to. 
dxMeter = params.camera.pixelsize_meters/params.magnification    #The length in meter that 1 pixel correspond to. 

#%%
# if not reanalyze:
#     resultsList = []
#     for pp in dataPath:
#         resutlsPath = os.path.join(pp, 'results.pkl')        
#         if os.path.exists(resutlsPath):
#             with open(resutlsPath, 'rb') as f:
#                 resultsList.append( pickle.load(f) )



#%%
columnDensities, variableLog = ImageAnalysisCode.PreprocessZylaImg(*dataPath, examRange=[examFrom, examUntil], 
                                                                   rotateAngle=rotateAngle, 
                                                                   rowstart=rowstart, rowend=rowend, 
                                                                   columnstart=columnstart, columnend=columnend,
                                                                   subtract_burntin=subtract_burntin, 
                                                                   skipFirstImg=skipFirstImg, 
                                                                   showRawImgs=showRawImgs, rebuildCatalogue=1,
                                                                    # filterLists=[['wait<9000']]
                                                                    )

autoCrop = 0
if autoCrop:
    columnDensities = ImageAnalysisCode.AutoCrop(columnDensities, sizes=[200, 150])
    print('ColumnDensities auto cropped.')
#%%
        
popts, bgs = ImageAnalysisCode.FitColumnDensity(columnDensities, dx = dxMicron, mode='both', yFitMode='single',
                                                subtract_bg=subtract_bg, Xsignal_feature='wide', Ysignal_feature='wide')

results = ImageAnalysisCode.AnalyseFittingResults(popts, logTime=variableLog.index) 



if variableLog is not None:
    results = results.join(variableLog)

if saveresults:
    ImageAnalysisCode.SaveResultsDftoEachFolder(results, overwrite=overwriteOldResults)    

results.to_csv('D:/Dropbox (Lehigh University)/Sommer Lab Shared/Data/2024/10-2024/09 Oct 2024/2passResult')
print('===')
#%%
# mask = (results.Ywidth < 5.5)  & (results.Ywidth > 3.9) & (results.HF_AOM_Freq<=317)
# results = results[ mask ]
# columnDensities = columnDensities[mask]
# popts[0] = np.array(popts[0])[mask]
# popts[1] = np.array(popts[1])[mask]
# bgs[0] = np.array(bgs[0])[mask]
# bgs[1] = np.array(bgs[1])[mask]

# %%
# ImageAnalysisCode.PlotFromDataCSV(results, 'HF_AOM_Freq', 'Ywidth', 
#                                    iterateVariable='Lens_Position', 
#                                    # filterLists=[['Ywidth<10']],
#                                   groupbyX=1, threeD=0,
#                                   figSize = 0.5
#                                   )

# ImageAnalysisCode.PlotFromDataCSV(results, 'RF_FRQ_MHz', 'YatomNumber', 
# #                                   # iterateVariable='VerticalBiasCurrent', 
# #                                   # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
#                                   groupbyX=1, threeD=0,
#                                   figSize = 0.5
#                                   )
# ImageAnalysisCode.PlotFromDataCSV(results, 'LF_AOM_freq', 'Ywidth', 
#                                    iterateVariable='Lens_Position', 
#                                    # filterLists=[['wait>=10', 'wait<=100']],
#                                    groupbyX=1, threeD=0,
#                                    figSize = 0.5, do_fit=0
#                                    )

ImageAnalysisCode.PlotFromDataCSV(results, 'Evap1_V', 'Ycenter', 
                                  # iterateVariable='VerticalBiasCurrent', 
                                  # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
                                  # groupby='ODT_Position', 
                                    groupbyX=1, 
                                  threeD=0,
                                  figSize = 0.5
                                  )


ImageAnalysisCode.PlotFromDataCSV(results, 'wait', 'YatomNumber', 
                                  # iterateVariable='VerticalBiasCurrent', 
                                  # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
                                  # groupby='ODT_Position', 
                                    groupbyX=1, 
                                  threeD=0,
                                  figSize = 0.5
                                  )

# ImageAnalysisCode.PlotFromDataCSV(results, 'fmod_kHz', 'Ywidth', 
#                                   # iterateVariable='VerticalBiasCurrent', 
#                                   # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
#                                   # groupby='ODT_Position', 
#                                     groupbyX=1, 
#                                   threeD=0,
#                                   figSize = 0.5
#                                   )


ImageAnalysisCode.PlotFromDataCSV(results, 'LowServo1', 'YatomNumber', 
                                  # iterateVariable='VerticalBiasCurrent', 
                                  # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
                                  # groupby='ODT_Position', 
                                    groupbyX=1, 
                                  threeD=0,
                                  figSize = 0.5
                                  )

# ImageAnalysisCode.PlotFromDataCSV(results, 'CamBiasCurrent', 'YatomNumber', 
#                                   # iterateVariable='VerticalBiasCurrent', 
#                                   # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
#                                   # groupby='ODT_Position', 
#                                     groupbyX=1, 
#                                   threeD=0,
#                                   figSize = 0.5
#                                   )

# fig, ax = plt.subplots(figsize=(5,4), layout='constrained') 
# results.YatomNumber.plot(title='Atom Number', linestyle='', marker='.')

# fig, ax = plt.subplots(figsize=(5,4), layout='constrained') 
# results.Ycenter.plot(title='y Position', linestyle='', marker='.')

# fig, ax = plt.subplots(figsize=(5,4), layout='constrained') 
# results.Xcenter.plot(title='x Position', linestyle='', marker='.')

# %%

intermediatePlot = 1
plotPWindow = 3
plotRate = 1
uniformscale = 0
rcParams = {'font.size': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9}

variablesToDisplay = [
                    # # 'Coil_medB', 
                        # 'TOF',
                        'Evap1_V',
                        # 'LowServo1',
                        # 'Evap_time_2'
                        # 'Evap_timestep'
                        'wait',
                        'Evap_Time_2',
                        'WP_angle',
                        # 'Lens_Position',
                        # 'StopEvap_LowServo',
                        # 'StopEvap_Time',
                        # 'LF_AOM_freq',
                        # 'Lens_Position',
                        # 'FB Voltage',
                        # 'B_Field',
                        # 'ODT_Position',
                        # 'fmod_kHz',
                        # 'tmod_ms',
                        # 'Cycles_num',
                        # 'Mod_amp',
                        # 'Evap_Tau',
                        # 'VerticalBiasCurrent',
                        # 'B_spikeTime',
                        # 'HF_AOM_Freq',
                        # 'CamBiasCurrent',
                        #'Lens Position',
                        # 'IR_Waveplate',
                        # 'B_Field',
                        # 'BEC_fieldRamp_ms',
                        
                      ]
showTimestamp = False
# variablesToDisplay=None
textY = 1
textVA = 'bottom'

if intermediatePlot:
    # ImageAnalysisCode.ShowImagesTranspose(images_array, uniformscale=False)
    ImageAnalysisCode.plotImgAndFitResult(columnDensities, popts, bgs=bgs, 
                                          dx=dxMicron, 
                                            # filterLists=[['Evap_Time_2==1.5']],
                                           plotRate=plotRate, plotPWindow=plotPWindow,
                                            variablesToDisplay = variablesToDisplay,
                                           showTimestamp=showTimestamp,
                                          variableLog=variableLog, 
                                          logTime=variableLog.index,
                                          uniformscale=uniformscale,
                                          textLocationY=0.9, rcParams=rcParams,
                                          figSizeRate=1, sharey='col')
    
    # ImageAnalysisCode.plotImgAndFitResult(columnDensities, popts, bgs=bgs, dx=dx, 
    #                                       plotRate=1, plotPWindow=plotPWindow,
    #                                       variablesToDisplay = variablesToDisplay,
    #                                       variableLog=variableLog, logTime=variableLog.index,
    #                                       textLocationY=0.8, rcParams=rcParams)

    # xx = np.arange(len(imgs_oneD[0]))
    # fig, axes = plt.subplots(fileNo, 1, sharex=True, layout='constrained')
    # for ii in range(fileNo):        
    #     axes[ii].plot(imgs_oneD[ii], '.')
    #     axes[ii].plot(xx, ImageAnalysisCode.Gaussian(xx, *popt_Basler[ii]))
    #     axes[ii].text(0.9,0.8, files[ii], transform=axes[ii].transAxes)

    # c, w = np.array(popt_Basler).mean(axis=0)[1:-1]
    # axes[-1].set(xlim=[c-15*w, c+15*w])
    
#%% LINEAR FIT
# fit atom number vs. lowServo to a line

var_indep = 'LowServo1'
# var_dep = 'YatomNumber'

# groupedData = results.groupby(var_indep)[var_dep].mean()

# Xdata = groupedData.index.to_numpy()
# Ydata = groupedData.values

# coeff = np.polyfit(Xdata, Ydata, 1)

# lin_x = np.linspace(min(Xdata), max(Xdata), 50)
# lin_y = coeff[0] * lin_x + coeff[1]

# plt.figure()
# plt.scatter(Xdata, Ydata)
# plt.plot(lin_x, lin_y)

# plt.suptitle( 'Equation: ' + str(round(coeff[0],2)) + '*x + ' + str(round(coeff[1])) )
# plt.xlabel(var_indep)
# plt.ylabel(var_dep)
# plt.tight_layout()

#%%
results = pd.read_csv('D:/Dropbox (Lehigh University)/Sommer Lab Shared/Data/2024/10-2024/09 Oct 2024/misalignedLifetime')
results2 = results.groupby('WP_angle')

k = []

for WP_angle in results2:
    l = []
    popt, pcov = ImageAnalysisCode.fit_exponential(WP_angle[1]['wait'], WP_angle[1]['YatomNumber'],
        dx=1, doplot = True, label="", title="Trap Lifetime", newfig=True, xlabel="Wait Time (ms)", ylabel="Y Atom Number", 
        offset = 0, 
        legend=True)
    matrix = np.array(pcov)
    diag = np.diagonal(matrix)
    l.append(WP_angle[0])
    l.append(popt[1])
    l.append(np.sqrt(diag[1]))
    # print(l)
    # # print(popt[1])
    # print('===')
    # print(pcov)
    # print(diag[1])
    # print('===')
    # print(WP_angle)
    k.append(l)
# print(k)
k = np.array(k)
print(k)
np.savetxt('D:/Dropbox (Lehigh University)/Sommer Lab Shared/Data/2024/10-2024/09 Oct 2024/Andor/life_WP_angle_misaligned.dat', k)

#%%
c = np.loadtxt('D:/Dropbox (Lehigh University)/Sommer Lab Shared/Data/2024/10-2024/09 Oct 2024/Andor/life_WP_angle_misaligned.dat', dtype=float)
print(c)
#%%

plt.figure(figsize=(5,4))
plt.title('')
plt.xlabel('Waveplate angle', fontsize=12)
plt.ylabel('Lifetime (ms)', fontsize=12)

k = np.array(k)
plt.errorbar(k[:,0], k[:,1], yerr=k[:,2], fmt='o')
plt.tight_layout()

#%% LIFETIME MEASUREMENT

# popt,_ = ImageAnalysisCode.fit_exponential(results['wait'], results['YatomNumber'],
#     dx=1, doplot = True, label="", title="Trap Lifetime", newfig=True, xlabel="wait (ms)", ylabel="Y Atom Number", 
#     offset = 0, 
#     legend=True)

# print('Lifetime: ', round(popt[1]*10**(-3), 3), ' s')

#%% OSCILLATION OF CLOUD

# dfmean = results.groupby('wait')[['Xcenter', 'Ycenter']].mean().reset_index()
# dfstd = results.groupby('wait')[['Xcenter', 'Ycenter']].std().reset_index()

# plt.figure()
# plt.errorbar(dfmean['wait'], dfmean['Ycenter'], yerr=dfstd['Ycenter'], fmt='-o')
# plt.ylabel('Ycenter')
# plt.xlabel('wait')
# plt.tight_layout()

# plt.figure()
# plt.errorbar(dfmean['wait'], dfmean['Xcenter'], yerr=dfstd['Xcenter'], fmt='-o')
# plt.ylabel('Xcenter')
# plt.xlabel('wait')
# plt.tight_layout()

# %% THERMOMETRY

# # var2 = 'D1_Re_Attn'
# var2 = 'Evap_timestep'
# # var2 = 'LowServo1'
# # var2 = 'Evap_Time_2'
# df = ImageAnalysisCode.multiVariableThermometry(
#                                                 results, 
#                                                 # var1, 
#                                                 var2, 
#                                                 fitXVar='TOF',  fitYVar='Ywidth',do_plot=1, add_Text=1)


#%% ASPECT RATIO CALCULATION
# filteredTimestep = results.query('Evap_timestep == 2')

# aspectRatio = filteredTimestep['Ywidth'] / filteredTimestep['Xwidth']

# plt.figure(figsize=(4,3))
# plt.scatter(filteredTimestep['TOF'], aspectRatio.values)
# # plt.errorbar(width_mean['TOF'], aspectRatio, fmt='o')
# plt.xlabel('Time-of-flight (ms)')
# plt.ylabel('Aspect Ratio')
# plt.tight_layout()

# %% 2-D plot when have two variable parameters
# cols = ['PSD', 'T (K)', 'AtomNum']

# for r in cols:
#     df1 = df[r].unstack()
#     fig, ax = plt.subplots(1,1, figsize=[4,3], layout='constrained')
#     cax = ax.pcolormesh(df1.columns, df1.index, df1.values, cmap='viridis')
#     ax.set(xlabel=var2, ylabel=var1, title=r)
#     fig.colorbar(cax, ax=ax)


# %% 1-D plot when only vary one parameter
# plt.rcParams['figure.figsize'] = [4, 3]

# colnames = ['PSD', 'AtomNum', 'T (K)']

# fig, axes = plt.subplots(1,3, figsize=[8, 2.5], layout='constrained')
# for ii, ax in enumerate(axes): 
#     df[colnames[ii]].plot(ls='',marker='x', ax=ax)
#     ax.set(title=colnames[ii], )
#     # ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
#     ax.set_yscale('log')

# fig, ax = plt.subplots(1,1, figsize=[4, 3], layout='constrained')

# Amean = results[results.TOF==0].groupby([var1, var2]).mean().YatomNumber.values
# Astd = results[results.TOF==0].groupby([var1, var2]).std().YatomNumber.values
# ax.errorbar(np.arange(len(Amean)), Amean, Astd, marker='.')
# ax.set(title=colnames[1])
# ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))



# fig, ax = plt.subplots(1,1, layout='constrained')
# ax.plot(df.AtomNum, df.PSD, '.')
# ax.set(xlabel='AtomNum', ylabel='PSD')
# ax.set_yscale('log')
# ax.set_xscale('log')

# fig, ax = plt.subplots(1,1, layout='constrained')
# ax.plot(df.index, df['T (K)'], '.')
# ax.set(xlabel='Time (s)', ylabel='T (K)')
# ax.set_yscale('log')
# # ax.set_xscale('log')

# fig, ax = plt.subplots(1,1, layout='constrained')
# ax.plot(df.index, df.AtomNum, '.')
# ax.set(xlabel='Time (s)', ylabel='Atom Number')
# ax.set_yscale('log')

# # fig, ax = plt.subplots(1,1, layout='constrained')
# # ax.plot(df['AtomNum'], df.PSD, '.')
# # ax.set(xlabel='AtomNum', ylabel='PSD')

# # %%
# print('======================')
# print('The phase space density is:\n{}'.format(df[['AtomNum', 'T (K)']]))
