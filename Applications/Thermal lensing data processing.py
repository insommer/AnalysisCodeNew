import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

P = [4.5, 38, 83, 127]
# z0 = [508.86, 504.88, 501.46, 500.2]
z0 = [302.3, 300.58, 297.17, 296]

def lin(x, m, b):
    return m*x + b

guess = [z0[0]-z0[-1]/(P[0]-P[-1]) , z0[0]]


x = np.linspace(P[0], 135, 500)
param, _ = curve_fit(lin, P, z0, p0=guess)

plt.figure(figsize=(5,4))
plt.scatter(P, z0)
plt.plot(x, lin(x, *param))
plt.xlabel('Power (W)'); plt.ylabel('Focus position (mm)')
plt.grid(True)
plt.tight_layout()