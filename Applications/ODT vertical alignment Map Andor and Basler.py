from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.close('all')

####################################
#Set the date and the folder name
####################################

dataRootFolder = r'C:\Users\wmmax\Documents\Lehigh\Sommer Group\Experiment Data'

date = '10/4/2026'

data_folder = [
    'ODT 1620 Misalign'
]

####################################
# Parameter Setting'
####################################
cameras = [
    'zyla',
]

REPETITION = 3


reanalyze = 1
saveresults = 0
overwriteOldResults = 1

examNum = None #The number of runs to exam.
examFrom = None #Set to None if you want to check the last several runs. 
autoCrop = 0
showRawImgs = 0


# in the format of [zyla, chameleon]
runParams = {
    'subtract_burntin': [0, 0],
    'skip_first_img': ['auto', 0],
    'rotate_angle': [-1.5, 0], #rotates ccw
    'ROI': [
        # rowStart, rowEnd, colStart, colEnd, for each camera
        # [500, 1000, 100, -100], 
        [500, 1000, 100, -100], 
        # [10, -10, 10, -10],
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


#%% Atomic Absorption Imaging Analysis

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
                                                    dx = params.dx_micron, 
                                                    mode='both', 
                                                    yFitMode='single',
                                                    subtract_bg=params.subtract_bg, 
                                                    Xsignal_feature=params.x_feature,
                                                    Ysignal_feature=params.y_feature
                                                    )

    results[cam] = ImageAnalysisCode.AnalyseFittingResults(fits[cam][0], logTime=varLog[cam].index)
    results[cam] = results[cam].join(varLog[cam])
    
    if saveresults:
        ImageAnalysisCode.SaveResultsDftoEachFolder(results[cam], overwrite=overwriteOldResults)    

    print('='*20)



centers_andor = ImageAnalysisCode.FitColumnDensity_MultiGaussianY(
    OD['zyla']/params.expmntParams.cross_section,
    centersOnly=True,
    doPlot=False
    )

#%% Beam dump cam Analysis (Basler)

paths_bas = [ os.path.join(dayfolder, 'Basler', f) for f in data_folder]
ROI = [1, -1, 1, -1]

df_bas = ImageAnalysisCode.ExtractGaussianCenter_Basler(paths_bas, ROI)  

# assuming vertical misalignment changes Xcenter on Basler
centers_basler = np.array(df_bas['Xcenter'])

#%% Map between Andor and Basler cameras

pass1center_andor, pass2center_andor = ImageAnalysisCode.ID_misaligned_beams(centers_andor)

targetpos_andor = np.mean(pass1center_andor)

# avg over the images at a given misalignment position
ANDOR_COOR = pass2center_andor.reshape(-1, REPETITION).mean(axis=1)
ANDOR_COOR_std = pass2center_andor.reshape(-1, REPETITION).std(axis=1)

BASLER_COOR = centers_basler.reshape(-1, REPETITION).mean(axis=1)
BASLER_COOR_std = centers_basler.reshape(-1, REPETITION).std(axis=1)


# map between andor and basler -- for a target Andor position given by pass1, determine Basler position
# basler_coor = m*andor_coor + b
m, b = np.polyfit(ANDOR_COOR, BASLER_COOR, 1)

pass2_target_andor = m * targetpos_andor + b

plt.figure(figsize=(5,4))

plt.errorbar(ANDOR_COOR, BASLER_COOR, yerr=BASLER_COOR_std, xerr=ANDOR_COOR_std, fmt='-o', capsize=3)
plt.plot(targetpos_andor, pass2_target_andor, '*r', label=f'Basler target = {int(pass2_target_andor)}')

plt.axhline(pass2_target_andor, ls='--', color='r', alpha=0.3)
plt.axvline(targetpos_andor, ls='--', color='r', alpha=0.3)


plt.xlabel('Andor Y coordinate')
plt.ylabel('Basler X coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


# %%

for cam in cameras:
    
    intermediatePlot = 1
    plotPWindow = 6
    plotRate = 1
    uniformscale = 0.5
    rcParams = {'font.size': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                # 'image.interpolation': 'nearest'
                }

    variablesToDisplay = [
                            'ODT_Misalign'
                          ]
    showTimestamp = False
    textY = 1
    textVA = 'bottom'

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



