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

plt.close('all')

####################################
#Set the date and the folder name
####################################
# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'F:\Data'

date = '04/04/2025'
# date = '11/08/2024'
data_folder = [
    # r'Andor/ODT temp MF waveplate 220_1',
    # r'Andor/cMOT rf 220-228 MHz fine',
    # 'RF spectroscopy test'
    # r'Andor/cMOT rf 228-236 MHz fine',
    # r'Andor/Evap cloud Chop',
    # r'Andor/lifetime Evap1_V 0.35V',
    # r'Andor/MOT temp check',
    # r'Andor/D1 temp_1'
    # r'Andor/ODT temp check no evap_1'
    # r'Andor/D1 thermometry Cooling AOM 80.5 MHz_1'
    r'Andor/Small coil mot char 1V vert bias'
    # r'Andor/D1 temp check'


    ]

####################################
#Parameter Setting'
####################################
reanalyze = 1
saveresults = 0
overwriteOldResults = 1

repetition = 1 #The number of identical runs to be averaged.
subtract_burntin = 1

skipFirstImg = 'auto'
# skipFirstImg = 0

rotateAngle = 0 #rotates ccw
# rotateAngle = 0.5 #rotates ccw

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
    ] 

pictureToHide = None
# pictureToHide = [0,1,2,3] # list(range(0,10,2))

subtract_bg = 0
signal_feature = 'narrow' 
signal_width = 10 #The narrower the signal, the bigger the number.
fitbgDeg = 5

rowstart = 10
rowend = -10
columnstart = 10
columnend = -10

# rowstart = 200
# rowend = 700
# columnstart = 500
# columnend= 1200

# ODT 3300
# rowstart = 10
# rowend = 250
# columnstart = 500
# columnend = 1100

# ODT 2550
# columnstart=800
# columnend= 1000
# rowstart = 425
# rowend = 500

# ODT 1800
# columnstart=700
# columnend= 1200
# rowstart = 500
# rowend = 750

# ODT 1050
# columnstart=800
# columnend= 1300
# rowstart = 700
# rowend = 950

# ODT 300
# columnstart=900
# columnend= 1300
# rowstart = 900
# rowend = 1250

####################################
####################################
dayfolder = ImageAnalysisCode.GetDayFolder(date, root=dataRootFolder)
dataPath = [ os.path.join(dayfolder, f) for f in data_folder]

# variableLog_folder = dayFolder + r'/Variable Logs'
examFrom, examUntil = ImageAnalysisCode.GetExamRange(examNum, examFrom, repetition)

params = ImageAnalysisCode.ExperimentParams(date, axis='side', cam_type = "zyla")
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
                                                                    # filterLists=[['TOF<1.3']]
                                                                    )

autoCrop = 0
if autoCrop:
    opticalDensity = ImageAnalysisCode.AutoCrop(opticalDensity, sizes=[120, 70])
    print('opticalDensity auto cropped.')
#%%

columnDensities = opticalDensity / params.cross_section        
popts, bgs = ImageAnalysisCode.FitColumnDensity(columnDensities, dx = dxMicron, mode='both', yFitMode='single',
                                                subtract_bg=subtract_bg, Xsignal_feature='wide', Ysignal_feature='wide')

results = ImageAnalysisCode.AnalyseFittingResults(popts, logTime=variableLog.index) 



if variableLog is not None:
    results = results.join(variableLog)

if saveresults:
    ImageAnalysisCode.SaveResultsDftoEachFolder(results, overwrite=overwriteOldResults)    


# %%
ImageAnalysisCode.PlotFromDataCSV(results, 'TOF', 'Ywidth', 
                                   # iterateVariable='Lens_Position', 
                                   # filterLists=[['Ywidth<10']],
                                  groupbyX=1, threeD=0,
                                  figSize = 0.5
                                  )

# ImageAnalysisCode.PlotResults(results, 'RF_FRQ_MHz', 'YatomNumber', 

#                                   # iterateVariable='VerticalBiasCurrent', 
#                                   # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
#                                   # groupby='ODT_Position', 
#                                     groupbyX=1, 
#                                   threeD=0,
#                                   figSize = 0.5
#                                   )


# fig, ax = plt.subplots(figsize=(5,4), layout='constrained') 
# results.Xcenter.plot(title='x Position', linestyle='', marker='.')

# %%

intermediatePlot = 1
plotPWindow = 5
plotRate = 1
uniformscale = 0
rcParams = {'font.size': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
            # 'image.interpolation': 'nearest'
            }

variablesToDisplay = [
                    # # 'Coil_medB', 
                        'TOF',
                        # 'ODT_Misalign',
                        # 'Evap1_V',
                        # 'LowServo1',
                        # 'Evap_time_2'
                        # 'Evap_timestep',
                        'wait',
                        # 'VerticalBiasCurrent',
                        'ZSBiasCurrent',
                        # 'FieldRamp_ms',
                        # 'HoldTime_ms',
                        # 'ODT_Position',
                        # 'fmod_kHz',
                        # 'tmod_ms',
                        # 'Cycles_num',
                        # 'Mod_amp',
                        # 'RF_FRQ_MHz'
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
    
    
#%% LINEAR FIT
# fit atom number vs. lowServo to a line

# var_indep = 'LowServo1'
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
# results = pd.read_csv('D:/Dropbox (Lehigh University)/Sommer Lab Shared/Data/2024/10-2024/09 Oct 2024/misalignedLifetime')
# results2 = results.groupby('WP_angle')

# k = []

# for WP_angle in results2:
#     l = []
#     popt, pcov = ImageAnalysisCode.fit_exponential(WP_angle[1]['wait'], WP_angle[1]['YatomNumber'],
#         dx=1, doplot = True, label="", title="Trap Lifetime", newfig=True, xlabel="Wait Time (ms)", ylabel="Y Atom Number", 
#         offset = 0, 
#         legend=True)
#     matrix = np.array(pcov)
#     diag = np.diagonal(matrix)
#     l.append(WP_angle[0])
#     l.append(popt[1])
#     l.append(np.sqrt(diag[1]))
#     # print(l)
#     # # print(popt[1])
#     # print('===')
#     # print(pcov)
#     # print(diag[1])
#     # print('===')
#     # print(WP_angle)
#     k.append(l)
# # print(k)
# k = np.array(k)
# print(k)
# np.savetxt('D:/Dropbox (Lehigh University)/Sommer Lab Shared/Data/2024/10-2024/09 Oct 2024/Andor/life_WP_angle_misaligned.dat', k)


# c = np.loadtxt('D:/Dropbox (Lehigh University)/Sommer Lab Shared/Data/2024/10-2024/09 Oct 2024/Andor/life_WP_angle_misaligned.dat', dtype=float)
# print(c)


# plt.figure(figsize=(5,4))
# plt.title('')
# plt.xlabel('Waveplate angle', fontsize=12)
# plt.ylabel('Lifetime (ms)', fontsize=12)

# k = np.array(k)
# plt.errorbar(k[:,0], k[:,1], yerr=k[:,2], fmt='o')
# plt.tight_layout()

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

# filterLists = [['LowServo1>0.6'], ['LowServo1==0.6','TOF<1.5'], ['LowServo1==0.5', 'TOF<0.9']]
# filterLists = []
# fltedData = ImageAnalysisCode.DataFilter(results, filterLists=filterLists)



# var2 = 'wait'
# var2 = 'Evap_timestep'
var2 = 'VerticalBiasVoltage'
# var2 = 'Evap_Time_2'
df = ImageAnalysisCode.multiVariableThermometry(    
                                                results, 
                                                # fltedData,
                                                # var1, 
                                                var2, 
                                                fitXVar='TOF',  fitYVar='Ywidth',do_plot=1, add_Text=1)
results['zyla']['T (K)'] = df



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
