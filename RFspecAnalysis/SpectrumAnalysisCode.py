from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def multi_gaussian(x, *params):
    # params = [A1, mu1, sigma1, A2, mu2, sigma2, ...]
    n = len(params) // 3
    y = np.zeros_like(x, dtype=float)
    
    for i in range(n):
        A = params[3*i]
        mu = params[3*i + 1]
        sigma = params[3*i + 2]
        y += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    return y

def FitRFspectrum(dataFrame, peak_sep_MHz=0.15, sigma_guess=0.05, doPlot=True):

    # assumes XatomNumber for analysis
    if isinstance(dataFrame, pd.DataFrame):
        df = dataFrame.sort_values('RF_FRQ_MHz')
    else:
        df = dataFrame['zyla'].sort_values('RF_FRQ_MHz')
    Freq = df['RF_FRQ_MHz'].values
    Response = df['XatomNumber'].interpolate().values # in case there are nan values

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

    if doPlot:
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
        
    return stats