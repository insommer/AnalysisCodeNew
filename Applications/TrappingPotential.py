import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as s
import DMDanalysis
import os
import pandas as pd
import re

plt.close('all')

DA = DMDanalysis.DMDanalysis()

dataRootFolder = r'C:\Users\wmmax\Documents\Lehigh\Sommer Group\DMD\For main computer'
date = '2/3/2025'

camera = 'FLIR'
data_folder = [
    # fr'{camera}/AM 3.33 96.61',
    # fr'{camera}/AM 5.41 100.41',
    # fr'{camera}/AM 7.88 105.38',
    fr'{camera}/New LS 0.51 94.47',

    ]

dayFolder = DA.GetDataLocation(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]


repetition = 5


#%% Physical constants & parameters
def lin(x,m,b):
    y = m * x + b
    return y

if camera == 'FLIR':
    pixSize_um = 3.75
    w = 2048
    h = 1536
    
elif camera == 'Basler':
    pixSize_um = 2
    w = 3840
    h = 2160

pixArea_m = (pixSize_um * 1e-6) ** 2 # pixel area [meter^2]


c = s.c # speed of light [meter/sec]
w0 = 2*np.pi* 446.799677e12 # resonance angular freq [1/sec]
lamb = 638e-9 # DMD light wavelength [meter]
om = 2*np.pi * (c / (lamb)) # DMD light angular freq [1/sec]
gamma = 36.898e6 # natural linewidth [1/s]

# dipole potential amplitude
U0 = -(3*np.pi*c**2)/(2*w0**3) * (gamma/(w0-om) + gamma/(w0+om))


#%% Extract counts from images

current = [96.42, 100.9, 106.5, 113.8, 119.1, 122.8]

df = pd.DataFrame(columns=['File','Power (W)','Current (mA)','TotalCts','MaxCts','Max Intensity (W/m2)'])

for folder in dataPath:
        
    match = re.search(r"(\d+\.\d+)\s+(\d+\.\d+)", folder)
    if match:
        power = float(match.group(1))
        current = float(match.group(2))
    else:
        power = None
        current = None
        
    folder = folder+'/' 


    for filename in os.listdir(folder):
        
        path = folder+filename
        image_arr = DA.CheckFile(path)
                
        cts_tot = np.sum(np.sum(image_arr))
        cts_max = np.max(image_arr)
        
        
        df = pd.concat([df, pd.DataFrame({'File':[path],
                                          'Power (W)':[power*1e-3],
                                          'Current (mA)':[current],
                                          'TotalCts':[cts_tot],
                                          'MaxCts':[cts_max],
                                          })
                        ], 
                       ignore_index=True)

#%% Take average over # of images = reptition
df['Group'] = df.index // repetition

df_avg = df.groupby('Group').agg({
    'Power (W)': 'mean',
    'Current (mA)':'mean',
    'TotalCts': 'mean',
    'MaxCts': 'mean',
    'Max Intensity (W/m2)':'mean',
}).reset_index()


# fit cts vs. power
param_tot, _ = curve_fit(lin, df_avg['Power (W)'], df_avg['TotalCts'])
m,b = param_tot


factor = m # [Cts/W]

plot = 0
if plot:
    xfit = np.linspace(min(df_avg['Power (W)']), max(df_avg['Power (W)']), 100)
    yfit = lin(xfit, m, b)
    
    plt.figure()
    plt.scatter(df_avg['Power (W)'], df_avg['TotalCts'])
    plt.plot(xfit, yfit)
    
    plt.xlabel('Power (W)')
    plt.ylabel('Total Cts')
    
    plt.tight_layout()

#%% Calculate intensity of image, fit intensity vs. current

maxIntensity = []

for folder in dataPath:
    
    folder = folder+'/' 

    for filename in os.listdir(folder):
        path = folder+filename
        image_arr = DA.CheckFile(path)
        
        power_arr = image_arr / factor
        intensity_arr = power_arr / pixArea_m
        
        maxIntensity.append(np.max(intensity_arr))
        
df['Max Intensity (W/m2)'] = maxIntensity


df_avg = df.groupby('Group').agg({
    'Power (W)': 'mean',
    'TotalCts': 'mean',
    'Current (mA)':'mean',
    'MaxCts': 'mean',
    'Max Intensity (W/m2)':'mean',
}).reset_index()

# fit intensity data to controller current
param_I,_ = curve_fit(lin, df_avg['Current (mA)'], df_avg['Max Intensity (W/m2)'])
m_curr, b_curr = param_I

maxCurrent_mA = 325
xfit_curr = np.linspace(min(df_avg['Current (mA)']), maxCurrent_mA, 100)
yfit_curr = lin(xfit_curr, m_curr, b_curr)


U_dip = U0 * yfit_curr

Temp_uK = U_dip / s.k * 1e6

#%%
fig, ax1 = plt.subplots(figsize=(4, 3))

# Plot the data
ax1.plot(xfit_curr, Temp_uK, color="blue")
ax1.set_xlabel('Controller current (mA)')
ax1.set_ylabel('Temperature (uK)')
ax1.tick_params(axis='x')


fig.tight_layout()