import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.optimize import curve_fit
from ImageAnalysis import ImageAnalysisCode
from LightSheetAnalysis import LSAnalysisCode
import datetime
import configparser
from PIL import Image
import cv2
import os
import glob

plt.close('all')

dataRootFolder = r"C:\Users\insommer\Lehigh University Dropbox\Ariel Sommer\Sommer Lab Shared\Data"
date = '6/15/2026'

camera = 'Basler'

data_folder = [
    # fr'{camera}/VIDEO focus after BSPM 361.4 mm power 50 to 30 acqRate 25 ms',
    fr'{camera}/VIDEO MOT to D1'
    
    ]

acquisitionRate_ms = 1


doPlot = 0
angle = 0

rowstart=1150
rowend=1800
columnstart=750
columnend=1600
# rowstart=1
# rowend=-1
# columnstart=1
# columnend=-1
ROI = [rowstart, rowend, columnstart, columnend]

dayFolder = ImageAnalysisCode.GetDataLocation(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]

if camera == 'Basler':
    pixSize = 2 #um/px
elif camera == 'FLIR':
    pixSize = 3.75 #um/px
elif camera == 'Andor':
    pixSize = 6.5 #um/pix
#%%

df = pd.DataFrame(columns=['File', 'Time', 'Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])
   
#%%

fullpath = ImageAnalysisCode.GetFullFilePaths(dataPath)


pattern = os.path.join(dataPath[0], "*_[0-9][0-9][0-9][0-9].raw")

files = glob.glob(pattern)
files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

print(f"Found {len(files)} images. Starting with {os.path.basename(files[0])}")


image_list = []
for f in files:
    # with Image.open(f) as img:
        # image_list.append(np.array(img))
    image_list.append(ImageAnalysisCode.CheckFile(f))

# Convert list to a 3D array (Sequence, Height, Width)
image_stack = np.stack(image_list, axis=0)

print(f"Final array shape: {image_stack.shape}")

# empty lists to store fitted parameters
Xcenters = []; Ycenters = []; Xwidths = []; Ywidths = []; Xamps = []; Yamps = []; Time = []

for i, frame in enumerate(image_stack):
   
    Time.append(i * acquisitionRate_ms / 1000) # in seconds
   
    try:
        # frame, _ = ImageAnalysisCode.Rotate(frame, angle)
                
        paramX, paramY = ImageAnalysisCode.FitGaussian(frame, doPlot, 'Wide')
       
        Xcenter = paramX[0]*pixSize
        Xwidth = paramX[1]*pixSize
       
        Ycenter = paramY[0]*pixSize
        Ywidth = paramY[1]*pixSize
       
        Xcenters.append(Xcenter); Ycenters.append(Ycenter)
        Xwidths.append(Xwidth); Ywidths.append(Ywidth)
        Xamps.append(paramX[2]); Yamps.append(paramY[2])
       
    except Exception as e:
        print(f"Fit failed on frame {i}: {e}")
        Xcenters.append(None); Ycenters.append(None)
        Xwidths.append(None); Ywidths.append(None)
        Xamps.append(None); Yamps.append(None)

       
df['Xcenter'] = Xcenters; df['Ycenter'] = Ycenters
df['Xwidth'] = Xwidths; df['Ywidth'] = Ywidths
df['Xamp'] = Xamps; df['Yamp'] = Yamps    
df['Time'] = Time

#%%
# cols = ['Xwidth', 'Ywidth']
# thresh = 100

# for c in cols:
#     if (df[c] < thresh).any(): #or (results['zyla'][col2] > thresh).any():
    
#         df = ImageAnalysisCode.FilterDataframe(df, c, thresh)


fig, ax = plt.subplots(1,3, figsize=(9,4))

ax[0].plot(df['Time'], df['Xwidth'], label='Xwidth')
ax[0].plot(df['Time'], df['Ywidth'], label='Ywidth')
ax[0].set_title('Radius')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('um')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].plot(df['Time'], df['Xamp'], label='Xamp')
ax[1].plot(df['Time'], df['Yamp'], label='Yamp')
ax[1].set_title('Amplitude')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('A.U.')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

ax[2].plot(df['Time'], df['Xcenter']/1000, label='Xcenter')
ax[2].plot(df['Time'], df['Ycenter']/1000, label='Ycenter')
ax[2].set_title('Center')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('mm')
ax[2].legend()
ax[2].grid(True, alpha=0.3)

plt.tight_layout()

#%%

saveFileName = os.path.join(dataPath[0], 'results.csv')

df.to_csv(saveFileName, index=False)
