from ImageAnalysis import ImageAnalysisCode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import SpectrumAnalysisCode

plt.close('all')

dataRootFolder = r'D:\Lehigh University Dropbox\Ariel Sommer\Sommer Lab Shared\Data'
date = '9/8/2026'

data_folder = [
    
    # 'ODT RF spec test',
    # 'ODT RF spec_vary cam_clock transition',
    # 'ODT RF spec_vary cam_clock transition_1'
    # 'ODT RF spec_vary cam_0 to 5 transition'
    # 'ODT RF spec_cam bias 0.2_0 to 5 transition resolve'
    # 'ODT RF spec_cam bias 0.2 A'
    # 'ODT RF spec_cam bias 0.2 A_vary ZS bias_1'
    # 'ODT RF spec_cam bias 0.2 A ZS bias 0.675 A_vary vert bias_clock transition'
    'ODT RF spec_ZS 0.675 Vert 1.7_vary cam bias_0 to 5 transition FINE SCAN',
    'ODT RF spec_ZS 0.675 Vert 1.7_vary cam bias_0 to 5 transition FINE SCAN_1',
    'ODT RF spec_ZS 0.675 Vert 1.7_vary cam bias_0 to 5 transition FINE SCAN_2'




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


# results = results[results['RF_FRQ_MHz'] > 229.012]
# results = results[results['RF_FRQ_MHz'] < 229.095]


peak_sep_MHz = 0.5
peak_prominence = 0.1
sigma_guess = 0.005
window_length = 3
polyorder=2


biasAxis = 'CamBiasCurrent'

for biasVal in np.unique(results[biasAxis]):
    
    # df = results.groupby(biasAxis)
    df = results[results[biasAxis] == biasVal]

    stats = SpectrumAnalysisCode.FitRFspectrum(df, peak_sep_MHz, peak_prominence, sigma_guess,
                                               window_length, polyorder, atomNumberType='YatomNumber'
                                               )
    plt.title(f'{biasAxis} = {biasVal}')
    plt.tight_layout()
    
    
    print('-------- Fitting Results --------')
    print(f'{biasAxis} = {biasVal}')
    print(f'Center freq = {stats['Center_MHz']} MHz')
    print(f'Width = {stats['Width_MHz']*1e3} kHz')