from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import SpectrumAnalysisCode

plt.close('all')

<<<<<<< HEAD
dataRootFolder = r'C:\My Programs\Sommer lab data analysis\Data'
=======
dataRootFolder = r'C:\Users\insommer\Lehigh University Dropbox\Ariel Sommer\Sommer Lab Shared\Data'
>>>>>>> bdee12dffc7947dec816955a413b9abe3f288ca5

# date = '9/17/2025'
date = '12/3/2025'

data_folder = [
    
    # r'cMOT thermo'
    # r'RF after D1 cam bias 0.25 A track higher resonance',
    # r'RF after D1 cam bias 0.25 A track lower resonance',
    # r'RF after D1 cam bias 0.25 A scan about 228.2 MHz',
    # r'RF after D1 cam bias 0.25 A scan about 228.2 MHz_1',
    # r'D1_RFscan_noRamp_Vert1.33A_ScanZS_0_0.3A_cam0.11A'
    'D1_RFscan_noRamp_Vert1.33A_ZS0.38A_Cam0.15A',
    'D1_RFscan_noRamp_Vert1.33A_ZS0.38A_Cam0.15A_1',
    'D1_RFscan_noRamp_Vert1.33A_ZS0.38A_Cam0.15A_2',
    'D1_RFscan_noRamp_Vert1.33A_ZS0.38A_Cam0.15A_3',
<<<<<<< HEAD
=======
    
>>>>>>> bdee12dffc7947dec816955a413b9abe3f288ca5

    ]
####################################
#Parameter Setting'
####################################
cameras = ['zyla']
runParams = {}

dayfolder = ImageAnalysisCode.GetDayFolder(date, root=dataRootFolder)
paths_zyl = [ os.path.join(dayfolder, 'Andor', f) for f in data_folder]

runParams['paths'] = [paths_zyl]
runParams['expmntParams'] = np.vectorize(ImageAnalysisCode.ExperimentParams)(
    date, axis='side', cam_type=cameras)
runParams['dx_micron'] = np.vectorize(lambda a: a.camera.pixelsize_microns / a.magnification)(runParams['expmntParams'])

dfs = []
for path in paths_zyl:
    csvpath = os.path.join(path, 'results.csv')
    dfs.append(pd.read_csv(csvpath))

results = pd.concat(dfs, ignore_index=True)

# %%

plt.rcParams['font.size'] = 14

<<<<<<< HEAD
peak_sep_MHz = 0.15
sigma_guess = 0.25

stats = SpectrumAnalysisCode.FitRFspectrum(results, peak_sep_MHz, sigma_guess)

centers = 'Center_MHz'
widths = 'Width_MHz'
B = SpectrumAnalysisCode_Test.BfieldFromRF(stats, centers, widths)
=======

peak_sep_MHz = 0.15
peak_prominence = 0.05
sigma_guess = 0.5
window_length = 7
polyorder=2

stats = SpectrumAnalysisCode.FitRFspectrum(results, peak_sep_MHz, peak_prominence, sigma_guess,
                                           window_length, polyorder
                                           )
>>>>>>> bdee12dffc7947dec816955a413b9abe3f288ca5
