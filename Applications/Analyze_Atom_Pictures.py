from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import pandas as pd
import os
from scipy import constants

####################################
#Set the date and the folder name
#################################### 1.0 = 85.5A , 0.9 = 77.4A , 0.8 = 69.5A , 0.7 = 61.6A 
dataRootFolder = r"F:\Data"
# dataRootFolder = r'F:\Data'

date = '11/25/2025'

data_folder = [
    # 'MOT atoms vs LF attn TOF 0.2 ms_1',
    # 'MOT atoms vs LF VCO Attn 4.5 V TOF 0.2 ms',
    # 'MOT atoms vs LF VCO Attn 4.5 V TOF 0.2 ms_1',
    # 'cMOT atoms vs LF attn TOF 0.5 ms'
    # 'cMOT atoms vs LF VCO Attn 5 V TOF 0.5 ms'
    # 'D1 thermo_1',
    # '10WRF Driver test 50 khz setp center 228.2 2W_1',
    # '10WRF Driver test 50 khz setp center 228.2 3W',
    # 'RF_scan_cmot228_228.4_tof0.5_25kHzStep_2W',
    # 'RF_scan_cmot228_228.4_tof0.5_25kHzStep_3W',
    # 'RF_scan_cmot228_228.4_tof0.5_25kHzStep_3W',
    # 'RF_scan_cmot228.4_228.8_tof0.5_25kHzStep_3W_1',
    # # 'RF_scan_cmot227.6_228_tof0.5_25kHzStep_4W',
    # 'RF_scan_cmot228.8_229.2_tof0.5_25kHzStep_3W',
    # 'RF_scan_cmot229.2_230.2_tof0.5_50kHzStep_3W',
    # 'RF_scan_cmot229.175_230.175_tof0.5_50kHzStep_3W',
    # 'RF_scan_cmot228_228.4_tof0.5_10kHzStep_4W',
    'RF_scan_D1_227_229.5_tof0.5_50kHzStep_2W_vert1.03A',
    # 'RF_scan_D1_228_228.5_tof0.5_25kHzStep_2W',
    # 'D1 test_1'
    # 'RF_scan_D1_227_229.5_tof0.5_50kHzStep_3W_1',
    # 'RF_scan_D1_227_229.5_tof0.5_50kHzStep_3W_2',
    # 'RF_scan_D1_227_229.5_tof0.5_50kHzStep_2W'

]

####################################
# Parameter Setting'
####################################
cameras = [
    'zyla',
    # 'chameleon'
]

reanalyze = 1
saveresults = 0
overwriteOldResults = 1

examNum = None #The number of runs to exam.
examFrom = None #Set to None if you want to check the last several runs. 
autoCrop = 0
showRawImgs = 0


# in the format of [zyla, chameleon]
runParams = {
    'subtract_burntin': [1, 0],
    'skip_first_img': ['auto', 0],
    'rotate_angle': [0, 40], #rotates ccw
    'ROI': [
        # rowStart, rowEnd, colStart, colEnd, for each camera
        [10, -10, 10, -10], [10, -10, 10, -10],
        # [420, 520, 700, 1000],        
        # [850, 975, 750, 1250]
    ], 
    
    'subtract_bg': [0, 0], 
    'y_feature': ['wide', 'wide'], 
    'x_feature': ['wide', 'wide'], 
    'y_peak_width': [10, 10], # The narrower the signal, the bigger the number.
    'x_peak_width': [10, 10], # The narrower the signal, the bigger the number.
    'fitbgDeg': [5, 5],
    
    'optical_path': ['side', 'top']
}

# runParams['ROI'] = [[300, 700, 300, 1100], [850, 1025, 800, 1050]]

# Set filters for the data, NO SPACE around the operator.
filterLists = [[]] 

####################################
dayfolder = ImageAnalysisCode.GetDayFolder(date, root=dataRootFolder)
paths_zyl = [ os.path.join(dayfolder, 'Andor', f) for f in data_folder]
paths_cha = [ os.path.join(dayfolder, 'FLIR', f) for f in data_folder]
runParams['paths'] = [paths_zyl, paths_cha]

runParams['expmntParams'] = np.vectorize(ImageAnalysisCode.ExperimentParams)(
    date, axis=runParams['optical_path'], cam_type=cameras)

runParams['dx_micron'] = np.vectorize(lambda a: a.camera.pixelsize_microns / a.magnification)(runParams['expmntParams'])

runParams = pd.DataFrame.from_dict(runParams, orient='index', columns=['zyla', 'chameleon'])
examRange = ImageAnalysisCode.GetExamRange(examNum, examFrom)

####################################
####################################


# %%
# if not reanalyze:
#     resultsList = []
#     for pp in dataPath:
#         resutlsPath = os.path.join(pp, 'results.pkl')        
#         if os.path.exists(resutlsPath):
#             with open(resutlsPath, 'rb') as f:
#                 resultsList.append( pickle.load(f) )



#%%
OD = {}
varLog = {}
fits = {}
results = {}

for cam in cameras:
    params = runParams[cam]

    OD[cam], varLog[cam] = ImageAnalysisCode.PreprocessBinImgs(*params.paths, camera=cam, examRange=examRange,
                                                     rotateAngle=params.rotate_angle, 
                                                               ROI=params.ROI,
                                                      subtract_burntin=params.subtract_burntin, 
                                                      skipFirstImg=params.skip_first_img,
                                                      showRawImgs=showRawImgs, 
                                                      #!!!!!!!!!!!!!!!!!
                                                      #! Keep rebuildCatalogue = 0 unless necessary!
                                                      rebuildCatalogue=0,
                                                      ##################
                                                      # filterLists=[['TOF!=0']]
                                                      # filterLists=[['D1_AOM_Attn>7']]
                                                      # filterLists=[['D1CoolingPowerRamp_mW==6']]
                                                     )

    if autoCrop:
        OD[cam] = ImageAnalysisCode.AutoCrop(OD[cam], sizes=[120, 70])
        print('opticalDensity auto cropped.')

    # columnDensities[cam] = OD[cam] / params.expmntParams.cross_section
    # popts[cam], bgs[cam]
    fits[cam] = ImageAnalysisCode.FitColumnDensity(OD[cam]/params.expmntParams.cross_section, 
                                                    dx = params.dx_micron, mode='both', yFitMode='single',
                                                    subtract_bg=params.subtract_bg, Xsignal_feature=params.x_feature, 
                                                              Ysignal_feature=params.y_feature)

    results[cam] = ImageAnalysisCode.AnalyseFittingResults(fits[cam][0], logTime=varLog[cam].index)
    results[cam] = results[cam].join(varLog[cam])

    if saveresults:
        ImageAnalysisCode.SaveResultsDftoEachFolder(results[cam], overwrite=overwriteOldResults)    

    print('='*20)
    
# %% Filter zyla df if there are bad fits

# col1 = 'Ywidth'
# # col2 = 'YatomNumber'
# thresh = 1800

# if (results['zyla'][col1] > thresh).any(): #or (results['zyla'][col2] > thresh).any():

#     results['zyla'] = ImageAnalysisCode.FilterDataframe(results['zyla'], col1, thresh)

# %%

for cam in cameras:
    
    # ImageAnalysisCode.PlotResults(results[cam], 'RF_FRQ_MHz', 'XatomNumber',
    #                               filterLists=filterLists,
    #                               # iterateVariable='VerticalBiasCurrent', 
    #                               # groupby='ODT_Position', 
    #                                 groupbyX=1, 
    #                               threeD=0,
    #                               figSize = 0.5
    #                               )    
    
    # ImageAnalysisCode.PlotResults(results[cam], 'RF_FRQ_MHz', 'YatomNumber',
    #                               filterLists=filterLists,
    #                               # iterateVariable='VerticalBiasCurrent', 
    #                               # groupby='ODT_Position', 
    #                                 groupbyX=1, 
    #                               threeD=0,
    #                               figSize = 0.5
    #                               )    
    
    # # ImageAnalysisCode.PlotResults(results[cam], 'cMOT_dwell_ms', 'XatomNumber',
    # #                               filterLists=filterLists,
    # #                               # iterateVariable='VerticalBiasCurrent', 
    # #                               # groupby='ODT_Position', 
    # #                                 groupbyX=1, 
    #                               threeD=0,
    #                               figSize = 0.5
    #                               )
########################
########################
    intermediatePlot = 1
    plotPWindow = 5
    plotRate = 1
    uniformscale = 1
    rcParams = {'font.size': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                # 'image.interpolation': 'nearest'
                }

    variablesToDisplay = [
                        # # 'Coil_medB', 
                            'TOF',
                            # 'D1_AOM_VCO',
                            # 'D1_Re_VCO'
                            # 'D1Time_ms',
                            # 'D1RampTime_ms',
                            # 'D1CoolingPowerRamp_mW'
                            # 'D1Cooling_RampFinalV',
                            # 'D1Repump_RampFinalV'
                            # 'LFImg_Atten'
                            # 'D1_Cooling_FRQ',
                            # 'D1_Re_FRQ',
                            # 'Delta1_MHz',
                            # 'RamanDelta_MHz',
                            # 'D1_Re_VCO'
                            # 'wait',
                            # 'D1Time_ms',
                            # 'CamBiasCurrent',
                            # 'ZSBiasCurrent',
                            # 'VerticalBiasCurrent',
                            # 'RF_FRQ_MHz',
                            # 'RF_pulsetime_us'
                            # 'LowServo1',
                            # 'Lens_Position',
                            'RF_FRQ_MHz',
                            # 'RF_pulsetime_us'
                            # 'VericalBiasCurrent',
                            # 'D1_Re_VCO',
                            # 'Lens_pos'
                          ]
    showTimestamp = False
    textY = 1
    textVA = 'bottom'

    # filterLists = [['RF_FRQ_MHz>228.26']]

    if intermediatePlot:
        ImageAnalysisCode.plotImgAndFitResult(OD[cam]/runParams[cam].expmntParams.cross_section, 
                                              fits[cam][0], bgs=fits[cam][1], 
                                              dx=runParams[cam].dx_micron, 
                                              imgs2=OD[cam],
                                              
                                              filterLists=filterLists,
                                               plotRate=plotRate, plotPWindow=plotPWindow,
                                                variablesToDisplay = variablesToDisplay,
                                               showTimestamp=showTimestamp,
                                              variableLog=results[cam], 
                                              # logTime=varLog[cam].index,
                                              uniformscale=uniformscale,
                                              fontSizeRate=1.6,
                                              textLocationY=0.1, rcParams=rcParams,
                                              figSizeRate=1, 
                                              sharey='col'
                                             )

#%% GENERAL 2D SCAN FIGURE

# scanVar1 = 'D1CoolingPowerRamp_mW'
# scanVar2 = 'D1RampTime_ms'

# scanVar1 = 'D1Time_ms'
# # scanVar2 = 'D1PowerRatio'

scanVar1 = 'RF_FRQ_MHz'
scanVar2 = 'RF_AntennaPower'


dependentVar = 'YatomNumber'
ImageAnalysisCode.Plot_2Dscan_Errbars(results['zyla'], scanVar1, scanVar2, dependentVar)

# %%

scanVar1 = 'RF_FRQ_MHz'
scanVar2 = 'RF_AntennaPower'


dependentVar = 'XatomNumber'
ImageAnalysisCode.Plot_2Dscan_Errbars(results['zyla'], scanVar1, scanVar2, dependentVar)


# %% THERMOMETRY

# filterLists = [['LowServo1>0.6'], ['LowServo1==0.6','TOF<1.5'], ['LowServo1==0.5', 'TOF<0.9']]
# filterLists = [['TOF>0.1']]
# fltedData = ImageAnalysisCode.DataFilter(results['zyla'], filterLists=filterLists)


# var1 = 'D1_Cooling_FRQ'
# var2 = 'D1_Re_FRQ'
# var1 = 'RamanDelta_MHz'
# var2 = 'Delta1_MHz'
# var2 = 'D1Time_ms'

# var1 = 'D1RampTime_ms'
# var2 = 'D1Time_ms'

# # # # var2 = 'D1RampTime_ms'
# # # # var1 = 'D1TotalTime_ms'

# # # # # var1 = 'D1CoolingPowerRamp_mW'
# # # # # var2 = 'D1RampTime_ms'

# # # # var1 =  'D1_AOM_Attn' #'D1Cooling_PowerPerBeam_mW'
# # # # var2 = 'D1_Re_Attn' #'D1Repump_PowerPerBeam_mW'
# fitYVar = 'Ywidth'

# df1 = ImageAnalysisCode.multiVariableThermometry(results['zyla'], 
#                                             #fltedData,
#                                             var1, 
#                                             var2, 
#                                             fitXVar='TOF',
#                                             fitYVar=fitYVar,
#                                             do_plot=1, add_Text=1)


# df1 = df1.reset_index()


# plt.figure(figsize=(5,4))
# plt.plot(df1[var2], df1['T (K)']*1e6, '-o')
# plt.xlabel(var2); plt.ylabel('T (uK)'); plt.tight_layout()
# plt.title('T measured using '+ fitYVar)
# plt.tight_layout()

#%%
# ImageAnalysisCode.Plot_2Dscan_Errbars(df1, var1, var2, 'T (K)', 1e6)
# plt.title('T measured using '+ fitYVar)


# %% LIFETIME MEASUREMENT
# for cam in cameras:
#     popt,_ = ImageAnalysisCode.fit_exponential(results[cam]['wait'], results[cam]['YatomNumber'],
#         dx=1, doplot = True, label="", title="Trap Lifetime", newfig=True, xlabel="wait (s)", ylabel="Y Atom Number", 
#         offset = 0, 
#         legend=True)

#     print('Lifetime: ', round(popt[1]*10**(3), 3), ' s')