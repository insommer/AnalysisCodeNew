from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import rotate
import pandas as pd
import os
# from scipy import constants

plt.close('all')

####################################
#Set the date and the folder name
#################################### 1.0 = 85.5A , 0.9 = 77.4A , 0.8 = 69.5A , 0.7 = 61.6A 
# dataRootFolder = r"C:\Users\insommer\Lehigh University Dropbox\Ariel Sommer\Sommer Lab Shared\Data"
dataRootFolder = r'D:\Lehigh University Dropbox\Ariel Sommer\Sommer Lab Shared\Data'


date = '8/17/2026'

data_folder = [
    # 'ODT lin setpoint ramp_vary ramp time',
    # 'ODT lin setpoint ramp_vary ramp time_1'
    # 'ODT lin setpoint ramp_vary ramp time_final set point 1 V'
    # 'High Field Round-Trip Survival'
    # 'ODT_thermo_sameIRrampAsMF'
    # 'ODT_NoTOF_VaryFinalRampVal',
    # 'ODT_NoTOF_VaryFinalRampVal_1',
    # 'ODT_NoTOF_VaryFinalRampVal_2',
    # 'ODT_NoTOF_2DscanEvapTimeAndFinalVoltage',
    # 'ODT_NoTOF_2DscanEvapTimeAndFinalVoltage_1',
    # 'ODT_NoTOF_2DscanEvapTimeAndFinalVoltage_2'
    # 'ODT_Thermo_linSetRamp1.67_1.4V'
    # 'ODT_atoms_NoTOF_HFramp_HighSevo2_1.3_RampTime_1.75s'
    # 'ODT_atoms_varyRampTime_hold30ms_1',
    # 'ODT_atoms_varyRampTime_hold30ms'
    # 'odt load from d1 initial_2'
    # 'odt load from d1 MF atom number vs wait_1'
    # 'odt load from d1 MF thermo'
    # 'odt load from d1 MF thermo var HighServo1',
    # 'odt load from d1 MF thermo var HighServo1_1',
    # 'd1 thermo_2'
    # 'cmot cloud position_1'
    # 'odt evap MF atom number vs HighServo2_1'
    # 'd1 cloud position TOF 1 to 2 shift top cam',
    # 'd1 cloud position TOF 1 to 2'
    # 'd1 TOF 0 to 1'
    # 'd1 cloud position TOF 0 to 1 shift top cam'
    'd1 placement'
    
]

####################################
# Parameter Setting'
####################################
cameras = [
    # 'zyla',
    'chameleon'
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
    'rotate_angle': [0, 0], #rotates ccw
    'ROI': [
        # rowStart, rowEnd, colStart, colEnd, for each camera
        # [500, 1000, 100, -100], 
        [10, -10, 10, -10],
        [10, -10, 10, -10],
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

# col1 = 'YatomNumber'
# col2 = None
# thresh = 1e7

# if (results['zyla'][col1] > thresh).any() or (results['zyla'][col2] > thresh).any():

#     results['zyla'] = ImageAnalysisCode.FilterDataframe(results['zyla'], col1, thresh, col2=col2)
    
# results['zyla']['XatomNumber'] = np.clip(results['zyla']['XatomNumber'], a_min=0, a_max=1e7)

# %%

# results['zyla'] = results['zyla'].dropna()

# %%

for cam in cameras:
    
    # ImageAnalysisCode.PlotResults(results[cam], 'TOF', 'Xwidth',
    #                               filterLists=filterLists,
    #                               # iterateVariable='VerticalBiasCurrent', 
    #                               # groupby='ODT_Position', 
    #                                 groupbyX=1, 
    #                               threeD=0,
    #                               figSize = 0.5
    #                               )    
    
    # ImageAnalysisCode.PlotResults(results[cam], 'TOF', 'Ywidth',
    #                               filterLists=filterLists,
    #                               # iterateVariable='VerticalBiasCurrent', 
    #                               # groupby='ODT_Position', 
    #                                 groupbyX=1, 
    #                               threeD=0,
    #                               figSize = 0.5
    #                               )    
    
    ImageAnalysisCode.PlotResults(results[cam], 'HighServo2', 'YatomNumber',
                                  filterLists=filterLists,
                                  # iterateVariable='VerticalBiasCurrent', 
                                  # groupby='ODT_Position', 
                                    groupbyX=1, 
                                  threeD=0,
                                  figSize = 0.5
                                  )
#######################
#######################
    intermediatePlot = 1
    plotPWindow = 6
    plotRate = 1
    uniformscale = 1
    rcParams = {'font.size': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                # 'image.interpolation': 'nearest'
                }

    variablesToDisplay = [
                        # # 'Coil_medB', 
                            # 'RF_FRQ_MHz',
                            # 'D1_AOM_VCO',
                            # 'D1_Re_VCO'
                            # 'D1Time_ms',
                            # 'D1RampTime_ms',
                            # 'D1CoolingPowerRamp_mW'
                            # 'D1Cooling_RampFinalV',
                            # 'D1Repump_RampFinalV'
                            # 'LF_AOM_freq',
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
                            # 'RF_FRQ_MHz',
                            # 'RF_pulsetime_us'
                            # 'VericalBiasCurrent',
                            # 'D1_Re_VCO',
                            'TOF',
                            # 'MedB_Hold',
                            # 'HighServo1',
                            # 'HighServo2',
                            # 'Evap_Time_1'
                            # 'MedB_time',
                            # 'YatomNumber'
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
                                              fontSizeRate=1.8,
                                              textLocationY=0.1, rcParams=rcParams,
                                              figSizeRate=1, 
                                              sharey='col'
                                             )

#%% GENERAL 2D SCAN FIGURE

scanVar1 = 'HighServo2'
scanVar2 = 'wait'

dependentVar = 'YatomNumber'
ImageAnalysisCode.Plot_2Dscan_Errbars(results['zyla'], scanVar1, scanVar2, dependentVar)
plt.tight_layout()


# %% THERMOMETRY

# filterLists = [['LowServo1>0.6'], ['LowServo1==0.6','TOF<1.5'], ['LowServo1==0.5', 'TOF<0.9']]
filterLists = [['TOF>0']]
fltedData = ImageAnalysisCode.DataFilter(results['zyla'], filterLists=filterLists)


# var1 = 'Evap_timestep'
# var1 = 'D1Time_ms'

# var1 = 'D1_AOM_Attn'
# # var2 = 'D1_Re_Attn'
# var2 = 'CamBiasCurrent'

var1 = 'HighServo1'
var2 = 'Evap_Time_1'

fitYVar = 'Ywidth'

df1 = ImageAnalysisCode.multiVariableThermometry_v2(#results['zyla'], 
                                            fltedData,
                                            var1, 
                                            var2, 
                                            fitXVar='TOF',
                                            fitYVar=fitYVar,
                                            do_plot=1, add_Text=1)

df1 = df1.reset_index()

# df1 = df1[df1['T (K)'] > 1e-7]

plt.figure(figsize=(5,4))
for val2, group in df1.groupby(var2):
    plt.errorbar(group[var1], group['T (K)']*1e6, yerr=group['T error (K)']*1e6,
                 marker='o', label=f'{var2}={val2:.2f}', capsize=3)

plt.xlabel(var1)
plt.ylabel('T ($\mu$K)')
plt.legend()
plt.tight_layout()

# plt.figure(figsize=(5,4))
# plt.plot(df1[var1], df1['T (K)']*1e6, '-o')
# plt.xlabel(var1); plt.ylabel('T (uK)'); plt.tight_layout()
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

#     print('Lifetime: ', round(popt[1]/10**(3), 3), ' s')

#%% Save results dataframe?

# ImageAnalysisCode.saveResultsDF(results['zyla'], dayfolder)
