import DMDanalysis
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.optimize import curve_fit


plt.close('all')

DA = DMDanalysis.DMDanalysis()

dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
date = '11/4/2025'

# camera = 'FLIR'
camera = 'Basler'
data_folder = [
    
    # fr'{camera}/Last 554 mm',
    fr'{camera}/Last 776 mm',
    fr'{camera}/Last 953 mm',
    fr'{camera}/Last 1178 mm',
    
    # fr'{camera}/After cyl 2.11 mm',



    ]

repetition = 6
commonPhrase = True
quantity = 'Distance (mm)'

rowstart=1
rowend=-1
columnstart=1
columnend=-1


dayFolder = DA.GetDataLocation(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]

if camera == 'Basler':
    pixSize = 2 #um/px
elif camera == 'FLIR':
    pixSize = 3.75 #um/px
    

#%%

doPlot = 1
angle = 0

df = pd.DataFrame(columns=['File', 'Condition', 'Value', 'Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])

if commonPhrase:
    # Match either: distance + power OR distance only
    pattern_both = re.compile(r'(?:(\d+(?:\.\d+)?)\s*mm).*?power\s*(\d+)$', re.IGNORECASE)
    pattern_distance_only = re.compile(r'(\d+(?:\.\d+)?)\s*mm', re.IGNORECASE)

    conditions = []
    values = []
    distances = []

    for name in dataPath:
        basename = os.path.basename(name)

        match_both = pattern_both.search(basename)
        match_dist = pattern_distance_only.search(basename)

        if match_both:
            distance = float(match_both.group(1))
            value = int(match_both.group(2))
            condition = re.sub(pattern_both, '', basename).strip()
        elif match_dist:
            distance = float(match_dist.group(1))
            value = np.nan
            condition = re.sub(pattern_distance_only, '', basename).strip()
        else:
            distance = np.nan
            value = np.nan
            condition = basename.strip()

        conditions.extend([condition] * repetition)
        values.extend([value] * repetition)
        distances.extend([distance] * repetition)

    df['Condition'] = conditions
    df['Value'] = values
    df['Distance'] = distances


#%%
filenames = []
Xcenters = []; Ycenters = []
Xwidths = []; Ywidths = []
Xamps = []; Yamps = []

for folder in dataPath:
    
    folder = folder+'/'
    
    for filename in os.listdir(folder):
        
        path = folder+filename
        image_arr = DA.CheckFile(path)    
        image_arr, _ = DA.Rotate(image_arr, angle)
        
        paramX, paramY = DA.FitGaussian(image_arr[rowstart:rowend, columnstart:columnend], doPlot)
        
        Xcenter = paramX[0]*pixSize
        Xwidth = paramX[1]*pixSize
        
        Ycenter = paramY[0]*pixSize
        Ywidth = paramY[1]*pixSize
        
        Xcenters.append(Xcenter); Ycenters.append(Ycenter)
        Xwidths.append(Xwidth); Ywidths.append(Ywidth)
        Xamps.append(paramX[2]); Yamps.append(paramY[2])
        filenames.append(path)
        
df['Xcenter'] = Xcenters; df['Ycenter'] = Ycenters
df['Xwidth'] = Xwidths; df['Ywidth'] = Ywidths
df['Xamp'] = Xamps; df['Yamp'] = Yamps    
df['File'] = filenames    

#%%

colsForAnalysis = ['Xwidth', 'Ywidth']

if df['Value'].isna().any():
    stats = df.groupby(['Distance'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
    stats.columns = ['Distance'] + ['_'.join(col).strip() for col in stats.columns[1:]]
else:
    stats = df.groupby(['Distance', 'Value'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
    stats.columns = ['Distance', 'Value'] + ['_'.join(col).strip() for col in stats.columns[2:]]



#%%

for col in colsForAnalysis:
    
    plt.figure(figsize=(4,3))
    
    # for condition, group in stats.groupby('Value'):
    #     plt.errorbar(group['Distance'], group[col+'_mean'], group[col+'_std'], fmt='o-', capsize=3, label=condition)
    plt.errorbar(stats['Distance'], stats[col+'_mean'], stats[col+'_std'], fmt='-o', capsize=3)
    
    plt.xlabel(quantity)
    plt.ylabel(col)
    # plt.legend(title='Power %')
    plt.tight_layout()
    
#%%
def FitGaussianWaist(stats, colsForAnalysis, doPlot=True):

    def w_z(z, w0, z0, zR):
        return w0 * np.sqrt(1 + ((z - z0) / zR)**2)
    
    # assumes distances in mm, waist in um
    for col in colsForAnalysis:
        z = stats['Distance'].values
        w_meas = stats[col+'_mean'].values
        w_err = stats[col+'_std'].values
        
        p0 = [min(w_meas), z[np.argmin(w_meas)], (max(z) - min(z)) / 2]
        
        popt, pcov = curve_fit(w_z, z, w_meas, p0=p0, sigma=w_err, absolute_sigma=True)
        
        w0_fit, z0_fit, zR_fit = popt
        perr = np.sqrt(np.diag(pcov))
        
        if doPlot:
            z_fit = np.linspace(min(z), max(z), 300)
            fig, ax = plt.subplots(figsize=(4,3))
            ax.errorbar(z, w_meas, yerr=w_err, fmt='o', capsize=3)
            ax.plot(z_fit, w_z(z_fit, *popt), 'r-')
            ax.set_xlabel('Distance (mm)')
            ax.set_ylabel(col+' (μm)')
            ax.text(0.05, 0.85, f'w0={w0_fit:.2f} μm\nz0={z0_fit:.2f} mm', transform=ax.transAxes, bbox=dict(facecolor='white'))
            plt.tight_layout()
            
FitGaussianWaist(stats, colsForAnalysis)
    

        
        



