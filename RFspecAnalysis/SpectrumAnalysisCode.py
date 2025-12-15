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

    ResponseSmoothed = savgol_filter(Response, window_length=3, polyorder=2)

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

def LandeG(J,L,S,gL,gS):
    '''
    Parameters
    ----------
    J : Float
        Total angular momentum
    L : Float
        Orbital angular momenum.
    S : Float
        Spin.
    gL : Float
        Orbital g factor.
    gS : Float
        Spin g factor.

    Returns
    -------
    Float
        Lande G factor for a given coupled angular momenta. Returns hyperfine g if 
        J->F, L->J, S->I

    '''
    
    return (gL * ((J * (J + 1) + L * (L + 1) - S * (S + 1))/(2 * J + 1))
            + gS * ((J * (J + 1) + S * (S + 1) - L * (L + 1))/(2 * J + 1)))

def NumStates(J):
    '''
    Given a value for the angular momentum, returns number of mf values
    '''
    if not(2 * J == int(2 * J)):
        raise ValueError('J must be a half integer')
    return int(2 * J + 1)

def BfieldFromRF(stats, centers, widths, VerticalBiasCurrent, ZSBiasCurrent, CamBiasCurrent):
    '''
    Parameters
    ----------
    stats : TYPE
        DESCRIPTION.
    centers : TYPE
        DESCRIPTION.
    widths : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # muB / hbar
    muB = 1.399624604
    
    gS = 2.0023010
    gL = 0.99999587
    gI=-0.0004476540
    
    def gJ(state):
        return LandeG(state[2], state[0], state[1], gL, gS)
    def gF(state, gj):
        return LandeG(state[3], state[2], state[4], gj, gI)
    def ResonanceShift(stats, centers):
        energyshift = []
        for i in range(len(stats[centers])):
            energyshift.append(float(stats[centers][i] - 228.205))
        stats['energyShift (MHz)'] = pd.DataFrame(energyshift)
        return stats
    
    # List of hyperfine structure energy states
    # state = np.array([l, s, j, f, i])
    
    # Ground States = 0,1 , D1 States = 2,3 , D2 States = 4,5,6
    states = [[0, 0.5, 0.5, 0.5, 1], [0, 0.5, 0.5, 1.5, 1]]
    
    # np.array([1, 0.5, 0.5, 0.5, 1]),np.array([1, 0.5, 0.5, 1.5, 1]),
    # np.array([1, 0.5, 1.5, 0.5, 1]), np.array([1, 0.5, 1.5, 1.5, 1]),
    # np.array([1, 0.5, 1.5, 2.5, 1])]

    gj = []
    for i in range(len(states)):
        gj.append(gJ(states[i]))
        
    gf = []
    for i in range(len(states)):
        gf.append(gF(states[i], gj[i]))
    
    B = []
    
    statsEnergy = ResonanceShift(stats, centers)
    avg = []
    for l in range(len(stats[centers])):
        
        E = statsEnergy['energyShift (MHz)'][l]
        B.append([])
        
        
        for i in range(len(states)):
            
            N = NumStates(states[i][3])
            mf = np.linspace(-states[i][3], states[i][3], N)
            coef = np.zeros(N, dtype=float)
            B[l].append([])
            
            
            for k in range(len(mf)):
                
                coef[k] = gf[i] * muB * mf[k]
                
                B[l][i].append(float(E/coef[k]))
        # print(B[l][0][1], B[l][1][1])
        
        avg.append(abs(round(((B[l][0][1] + B[l][1][1]) / 2), 3)))
            # print(B[l][i])
            # avg.append(B[l][i][1])
        
        
    # print(avg) 
    # stats['B (G)'] = B
    stats['B (G)'] = avg
    
    print('B Field Strength from Peaks:', avg)  
    
    return stats