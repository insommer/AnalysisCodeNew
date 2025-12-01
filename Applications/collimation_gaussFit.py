import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit

def CheckFile(file):
    
    if isinstance(file, str) and file.endswith('.png'):  
        arr = np.array(Image.open(file).convert('L'))
        
    elif isinstance(file, np.ndarray):
        arr = file
        
    elif isinstance(file, str) and file.endswith('.raw'):
        
        temp = np.fromfile(file, dtype=np.uint8)
        arr = np.reshape(temp, (2160,3840)) # Basler dart resolution
        
    return arr

def Gauss1D(x,xc,sigX,A, offset):
     
    G = A * np.exp(-2 * (x-xc)**2 / sigX**2) + offset
    
    return G

def FitGaussian(gaussImageFile, graph=True, graphOption='Wide'):
     
    beam = CheckFile(gaussImageFile)
     
    Ny, Nx = beam.shape
    x_index = np.linspace(0, Nx-1, Nx)
    y_index = np.linspace(0, Ny-1, Ny)
     
    max_index = np.unravel_index(np.argmax(beam), beam.shape)
    max_x, max_y = max_index

    vert = beam[:, max_y]
    horiz = beam[max_x, :]
     
    sigGuess = 40
    offset = 5
     
    guessX = [max_y, sigGuess, np.max(horiz), offset]
    paramX,_ = curve_fit(Gauss1D, x_index, horiz, p0=guessX)
    x_fit1 = np.linspace(0, Nx-1, Nx*5)
    y_fit1 = Gauss1D(x_fit1, paramX[0], paramX[1], paramX[2], paramX[3])

    guessY = [max_x, sigGuess, np.max(vert), offset]
    paramY,_ = curve_fit(Gauss1D, y_index, vert, p0=guessY)
    x_fit2 = np.linspace(0, Ny-1, Ny*5)
    y_fit2 = Gauss1D(x_fit2, paramY[0], paramY[1], paramY[2], paramY[3])
             
    if graph:
        fig, ax = plt.subplots(1,3,figsize=(8,4))
         
        ax[1].plot(x_fit1, y_fit1,'r',linewidth=3)
        ax[1].scatter(x_index, horiz, s=20)
        ax[1].set_title('Fit vs. X')
         
        ax[2].plot(x_fit2,y_fit2,'r',linewidth=3)
        ax[2].scatter(y_index, vert, s=20)
        ax[2].set_title('Fit vs. Y')
         
        if graphOption == 'Narrow':
             
            ax[1].set_xlim(paramX[0]-4*paramX[1], paramX[0]+4*paramX[1])
            ax[2].set_xlim(paramY[0]-4*paramY[1], paramY[0]+4*paramY[1])
             
            ax[0].imshow(beam, extent=[paramX[0]-1, 
                                       paramX[0]+1, 
                                       paramY[0]-1, 
                                       paramY[0]+1])
        else:
            ax[0].imshow(beam*-1,cmap='binary')
         
        ax[0].set_title('Image')
        plt.tight_layout()
    
    return paramX, paramY

#%%

stem = 'D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data/2024/09-2024/04 Sep 2024\Basler\Lens collimation'


pos1 = stem + '\pos1 final.raw'
pos2 = stem + '\pos2 final.raw'

# 2 um/px
camPx = 2


g1 = FitGaussian(pos1, 1)
print('Position 1:')
print('Width X = ', round(g1[0][1]*camPx,3), ' um')
print('Width Y = ', round(g1[1][1]*camPx,3), ' um')


g2 = FitGaussian(pos2, 1)
print('Position 2:')
print('Width X = ', round(g2[0][1]*camPx,3), ' um')
print('Width Y = ', round(g2[1][1]*camPx,3), ' um')
