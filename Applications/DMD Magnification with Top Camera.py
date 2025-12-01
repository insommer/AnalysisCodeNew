import DMDanalysis
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from ImageAnalysis import ImageAnalysisCode

plt.close('all')

DA = DMDanalysis.DMDanalysis()

dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
date = '2/19/2025'

plane_for_analysis = 'Atom'

camera = 'FLIR'
# camera = 'Basler'

height = 150
width = 100

data_folder = [
    # fr'{camera}/Ref all mirrors',
    # fr'{camera}/Ref bg',
    fr'{camera}/All mirrors',
    # fr'{camera}/Mag 40 wide {height} tall',
    # fr'{camera}/Mag 50 wide {height} tall',
    # fr'{camera}/Mag 60 wide {height} tall',
    # fr'{camera}/Mag 70 wide {height} tall',
    # fr'{camera}/Mag 80 wide {height} tall',
    # fr'{camera}/Mag 90 wide {height} tall',
    # fr'{camera}/Mag 100 wide {height} tall',

    ]

doPlot = 1
repetition = 5
fit_type = 'Gaussian'



rowstart = 800
rowend = 1100
columnstart = 1000
columnend = 1150


# rowstart = 1
# rowend = -1
# columnstart = 1
# columnend = -1


rowstart += -100
rowend += -250
columnstart += -250
columnend += -100

dayFolder = DA.GetDataLocation(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]

if camera == 'Basler':
    pixSize = 2 #um/px
elif camera == 'FLIR':
    pixSize = 3.75 #um/px
    
DMD_pixSize = 7.56 #um/px
params = ImageAnalysisCode.ExperimentParams(date, axis='top', cam_type = 'chameleon')
#%%
savePath_ref = dayFolder + f'/{camera}/Ref all mirrors/ref.npy'
savePath_bg = dayFolder + f'/{camera}/Ref bg/bg.npy'

angle = 0
# angle = 40
angle = 58

df = pd.DataFrame(columns=['File', 'Xcenter (um)', 'Ycenter (um)', 'Xwidth (um)', 'Ywidth (um)', 'Xamp', 'Yamp', 'Rotation angle', 'Position'])

for folder in dataPath:
    
    folderName = folder
    folder = folder+'/'
    
    imgs2avg = []; bg2avg = []
    
    for filename in os.listdir(folder):
        
        path = folder+filename
        image_arr = DA.CheckFile(path)    
        image_arr, _ = DA.Rotate(image_arr, angle)
        
        image_arr_crop = image_arr[rowstart:rowend, columnstart:columnend]
        
        if os.path.basename(folderName).lower() in {'ref', 'ref all mirrors'}:
            imgs2avg.append(image_arr_crop)
        
        elif os.path.basename(folderName).lower() in {'ref bg', 'bg', 'background'}:
            bg2avg.append(image_arr_crop)
        
        else:
            # ref_img = np.load(savePath_ref)
            # bg_img = np.load(savePath_bg)
            
            # ratio = (image_arr_crop-bg_img) / (ref_img-bg_img)
            ratio = image_arr_crop
        
            Mag = 1
            if plane_for_analysis == 'Atom' or plane_for_analysis == 'atom':
                Mag = params.magnification
                            
            
            if fit_type == 'Gaussian' or fit_type == 'gaussian':
                
                paramX, paramY = DA.FitGaussian(ratio, doPlot)
    
                Xcenter = paramX[0]*pixSize / Mag
                Xwidth = paramX[1]*pixSize / Mag
                
                Ycenter = paramY[0]*pixSize / Mag
                Ywidth = paramY[1]*pixSize / Mag
                
                Resolution = None
            
            elif fit_type == 'Conv':
                
                paramX, paramY = DA.FitConvultion(ratio, doPlot)
                
                Xcenter = np.mean([paramX[0], paramX[1]]) * pixSize / Mag
                Xwidth = np.abs(paramX[0] - paramX[1]) * pixSize / Mag
                
                Ycenter = np.mean([paramY[0], paramY[1]]) * pixSize / Mag
                Ywidth = np.abs(paramX[0] - paramY[1]) * pixSize / Mag
                
                Resolution = (paramX[2]* pixSize / Mag, paramY[2]* pixSize / Mag)
        
            
        
            df = pd.concat([df, pd.DataFrame({'File':[path],
                                              'Xcenter (um)':[Xcenter],
                                              'Ycenter (um)':[Ycenter],
                                              'Xwidth (um)':[Xwidth] ,
                                              'Ywidth (um)':[Ywidth],
                                              'Resolution (um)':[Resolution],
                                              'Rotation angle':[angle],
                                              })
                            ], 
                           ignore_index=True)

if os.path.basename(folderName).lower() in {'ref', 'ref all mirrors'}:
    avg_img = np.mean(imgs2avg, axis=0)
    np.save(savePath_ref, avg_img)
    
elif os.path.basename(folderName).lower() in {'ref bg', 'bg', 'background'}:
    bg_img = np.mean(bg2avg, axis=0)
    np.save(savePath_bg, bg_img)
#%%
DMDwidth = []
DMDheight = []
for k in data_folder:

    numbers = re.findall(r"\d+", k)
    if len(numbers) >= 2:
        num1, num2 = map(int, numbers[:2])
        DMDwidth.append(int(num1))
        DMDheight.append(int(num2))
    else:
        DMDwidth.append(None)
        DMDheight.append(None)

if DMDwidth[0] is not None:
    DMDwidth = np.array(DMDwidth) * DMD_pixSize
    DMDheight = np.array(DMDheight) * DMD_pixSize


#%%

df['Group'] = df.index // repetition

df_avg = df.groupby('Group').agg({
    'Xcenter (um)': 'mean',
    'Ycenter (um)': 'mean',
    'Xwidth (um)': 'mean',
    'Ywidth (um)':'mean',
}).reset_index()

df_avg['DMD width (px)'] = DMDwidth / DMD_pixSize
df_avg['DMD width (um)'] = DMDwidth
df_avg['DMD height (px)'] = DMDheight / DMD_pixSize
df_avg['DMD height (um)'] = DMDheight
df_avg['Mag X'] = df_avg['Xwidth (um)'] / df_avg['DMD width (um)']
df_avg['Mag Y'] = df_avg['Ywidth (um)'] / df_avg['DMD height (um)']

df_std = df.groupby('Group').agg({
    'Xcenter (um)': 'std',
    'Ycenter (um)': 'std',
    'Xwidth (um)': 'std',
    'Ywidth (um)':'std',
}).reset_index()
df_std['Mag X'] = df_std['Xwidth (um)'] / df_avg['DMD width (um)']
df_std['Mag Y'] = df_std['Ywidth (um)'] / df_avg['DMD height (um)']

#%%

plt.figure(figsize=(4,3))

strX = 'DMD width (px)'
strY1 = 'Mag X'
# strY1 = 'Ywidth (um)'
strY2 = 'Mag Y'

plt.errorbar(df_avg[strX], df_avg[strY1], df_std[strY1], fmt='-o')
plt.errorbar(df_avg[strX], df_avg[strY2], df_std[strY2], fmt='-o')
plt.xlabel(strX)
# plt.ylabel(strY1)
# plt.ylabel('Ywidth in atom plane (um)')
plt.ylabel('M (' + plane_for_analysis + ' plane)'); plt.legend(['$M_x$', '$M_y$'])
plt.tight_layout()







        
        



