from ImageAnalysisMaster import ImageAnalysisCode
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
dataRootFolder = r'C:\Users\wmmax\Documents\Lehigh\Sommer Group\Experiment Data'

date = '9/17/2025'
# date = '9/29/2025'

data_folder = [
    
    # r'cMOT thermo'
    # r'RF after D1 cam bias 0.25 A track higher resonance',
    # r'RF after D1 cam bias 0.25 A track lower resonance',
    r'RF after D1 cam bias 0.25 A scan about 228.2 MHz',
    r'RF after D1 cam bias 0.25 A scan about 228.2 MHz_1',

    ]
####################################
#Parameter Setting'
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
    'subtract_burntin': [0, 0],
    'skip_first_img': ['auto', 0],
    'rotate_angle': [0, 0], #rotates ccw
    'ROI': [
        [10, -10, 10, -10],
        # [200, -100, 550, -350],
        # [10, -10, 450, -300]
    ], # rowStart, rowEnd, colStart, colEnd
    
    'subtract_bg': [0, 0], 
    'y_feature': ['wide', 'wide'], 
    'x_feature': ['wide', 'wide'], 
    'y_peak_width': [10, 10], # The narrower the signal, the bigger the number.
    'x_peak_width': [10, 10], # The narrower the signal, the bigger the number.
    'fitbgDeg': [5, 5],
    
    'optical_path': ['side', 'top']
}

# Set filters for the data, NO SPACE around the operator.
variableFilterList = [
    # [# 'wait==50', 
    # # 'VerticalBiasCurrent==0'
    # 'fmod_kHz==0',
    # # 'Evap_Tau==0.1',
    # # 'Evap_Time_1==2'
    # ], 
    ] 

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


#%%
OD = {}
varLog = {}
fits = {}
results = {}
rawImgs = {}

for cam in cameras:
    params = runParams[cam]

    OD[cam], varLog[cam], rawImgs[cam] = ImageAnalysisCode.PreprocessBinImgs(*params.paths, camera=cam, examRange=examRange,
                                                     rotateAngle=params.rotate_angle, 
                                                               ROI=params.ROI,
                                                      subtract_burntin=params.subtract_burntin, 
                                                      skipFirstImg=params.skip_first_img,
                                                      showRawImgs=showRawImgs,
                                                      returnRawImgs=1,
                                                      #!!!!!!!!!!!!!!!!!
                                                      #! Keep rebuildCatalogue = 0 unless necessary!
                                                      rebuildCatalogue=0,
                                                      ##################
                                                      # filterLists=[['TOF<1']]
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


# %%

plt.rcParams['font.size'] = 14

for cam in cameras:
    ImageAnalysisCode.PlotResults(results[cam], 'RF_FRQ_MHz', 'YatomNumber', 
                                  # iterateVariable='VerticalBiasCurrent', 
                                  # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
                                  # groupby='ODT_Position', 
                                    groupbyX=1, 
                                  threeD=0,
                                  figSize = 0.5
                                  )
    
    ImageAnalysisCode.PlotResults(results[cam], 'RF_FRQ_MHz', 'XatomNumber', 
                                  # iterateVariable='VerticalBiasCurrent', 
                                  # filterByAnd=['VerticalBiasCurrent>7.6', 'VerticalBiasCurrent<8'],
                                  # groupby='ODT_Position', 
                                    groupbyX=1, 
                                  threeD=0,
                                  figSize = 0.5
                                  )

# %%

    intermediatePlot = 1
    plotPWindow = 5
    plotRate = 1
    uniformscale = 0
    rcParams = {'font.size': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                # 'image.interpolation': 'nearest'
                }

    variablesToDisplay = [
                            'RF_FRQ_MHz'
                          ]
    showTimestamp = True
    textY = 1
    textVA = 'bottom'

    if intermediatePlot:
        ImageAnalysisCode.plotImgAndFitResult(OD[cam]/runParams[cam].expmntParams.cross_section, 
                                              fits[cam][0], bgs=fits[cam][1], 
                                              dx=runParams[cam].dx_micron, 
                                              imgs2=OD[cam], 
                                                # filterLists=[['LowServo1==0.5']],
                                               plotRate=plotRate, plotPWindow=plotPWindow,
                                                variablesToDisplay = variablesToDisplay,
                                               showTimestamp=showTimestamp,
                                              variableLog=varLog[cam], 
                                              logTime=varLog[cam].index,
                                              uniformscale=uniformscale,
                                              textLocationY=0.9, rcParams=rcParams,
                                              figSizeRate=1, 
                                              sharey=False
                                             )

#%%
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

def multi_gaussian(x, *params):
    # params = [A1, mu1, sigma1,  A2, mu2, sigma2, ...]
    n = len(params) // 3
    y = np.zeros_like(x, dtype=float)
    
    for i in range(n):
        A = params[3*i]
        mu = params[3*i + 1]
        sigma = params[3*i + 2]
        y += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    return y

# these parameters might have to be changed depending on the dataset
peak_sep_MHz = 0.15
sigma_guess = 0.05

df = results['zyla'].sort_values('RF_FRQ_MHz')
Freq = df['RF_FRQ_MHz'].values
Response = df['XatomNumber'].values

ResponseSmoothed = savgol_filter(Response, window_length=7, polyorder=2)


# Peak detection
freq_step = np.mean(np.diff(Freq))

peaks, props = find_peaks(
    ResponseSmoothed,
    prominence=np.max(ResponseSmoothed)*0.05,   # 5% prominence
    width=(1, 10),                              # flexible gaussian widths
    distance=int(peak_sep_MHz / freq_step)      # adjustable peak separation
)

print('Detected peaks:', len(peaks))
print('Peak freq:', Freq[peaks])


# initial parameters for fitting
p0 = []
for p in peaks:
    A_guess = Response[peaks].max()
    mu_guess = Freq[p]
    p0 += [A_guess, mu_guess, sigma_guess]

p0 = np.array(p0)


popt, pcov = curve_fit(multi_gaussian, Freq, Response, p0=p0)

n = len(popt) // 3

centers = []; center_err = []; widths = []; width_err = []

for i in range(n):
    mu_index = 3*i + 1
    sigma_index = 3*i + 2

    centers.append(popt[mu_index])
    widths.append(popt[sigma_index])

    center_err.append(np.sqrt(pcov[mu_index, mu_index]))
    width_err.append(np.sqrt(pcov[sigma_index, sigma_index]))

# keep track of B field conditions
vertBias = df['VerticalBiasCurrent'].iloc[0]
zsBias = df['ZSBiasCurrent'].iloc[0]
camBias = df['CamBiasCurrent'].iloc[0]

# Create dataframe
stats = pd.DataFrame({
    'Center_MHz': centers,
    'CenterErr_MHz': center_err,
    'Width_MHz': widths,
    'WidthErr_MHz': width_err,
    'VerticalBiasCurrent': [vertBias]*n,
    'ZSBiasCurrent': [zsBias]*n,
    'CamBiasCurrent': [camBias]*n,
})

FreqFit = np.linspace(min(Freq), max(Freq), 2000)

plt.figure(figsize=(8,5))
plt.plot(Freq, Response, 'o-')
plt.plot(Freq, ResponseSmoothed, '-', alpha=0.8, label='Smoothed')
plt.plot(FreqFit, multi_gaussian(FreqFit, *popt), 'r-', linewidth=2, label='Fit')
plt.plot(Freq[peaks], Response[peaks], 'gx', markersize=12, label='Detected peaks')

plt.legend()
plt.xlabel('RF_FRQ_MHz')
plt.ylabel('XatomNumber')
plt.tight_layout()