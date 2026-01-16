"""
Plot mechanical-loss spectra for a-Si and a-TiO2 at 300 K, comparing:
  - "CN" (network / connected-network) dissipation spectra, and
  - an independent TLS sum.

The script loads precomputed spectra from disk, rescales by network size, and
plots both materials on the same axes. The plot is displayed (save is available
but commented out).

Outputs:
  - On-screen figure (optionally save by uncommenting the fig.savefig line)

Data dependencies:
  - ./a_Si/Heat_Data/...
  - ./a_TiO2/TiO2_data_out/...
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats

# --- Plot style ---
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# ===============================
# Common plotting parameters
# ===============================
Gamma = 0.01  # coupling prefactor used in normalization
conversion = 0.006242  # eV/(GPa*A^3), used to form VC = (volume)*(elastic modulus)*conversion

fig, axs = plt.subplots(1, 1, figsize=(3, 2.25))
axs.axvspan(10**0, 10**4, alpha=0.1, color='k')  # highlight low-frequency band

# ============================================================
# a-Si (300 K): load spectra, rescale by network size, plot
# ============================================================
k_B = 8.617e-5
T = 300

# Material-specific volume used to construct VC
V_si = 27.43**3
C = 50  # GPa
VC = V_si * C * conversion  # used as denominator in the final plotted loss

Q_CN = []
Q_TLS = []
for i in range(0, 10):
    Q_CN.append(np.load('./a_Si/Heat_Data/Q_L_300K_sample_' + str(i + 1) + '.npy')[:, 0])
    Q_TLS.append(np.load('./a_Si/Heat_Data/Heat_TLS_300K_sample_' + str(i + 1) + '.npy'))

Q_CN = np.array(Q_CN)
Q_CN = np.where(Q_CN < 0, 0, Q_CN)  # clip negative values to zero

# Network sizes used to scale each sample
N_nodes = np.array([1844, 1860, 1915, 1847, 1575, 1880, 1905, 1965, 1765, 1901])
Q_CN = np.einsum('ij,i->ij', Q_CN, N_nodes)

Q_TLS = np.array(Q_TLS)

# Frequencies for CN curve and TLS curve
f = np.load('./a_Si/Heat_Data/frequency_300K.npy')[:, 0]
f_TLS = np.logspace(-6, 13, 2001)

# SEMs computed in original script (not plotted)
Q_CN_SEM = stats.sem(Q_CN / (4 * Gamma**2.0 * np.pi), axis=0)
Q_TLS_SEM = stats.sem(Q_TLS, axis=0)

axs.loglog(f, (np.nanmean(Q_CN, axis=0)) / (4 * Gamma**2.0 * np.pi * VC), 'k')
axs.semilogy(f_TLS, np.mean(Q_TLS, axis=0) / VC, 'k--')

# ============================================================
# TiO2 (300 K): load spectra, rescale by network size, plot
# ============================================================
# Material-specific volume used to construct VC
V_tio2 = 22.9705**2 * 14.794499874
C = 50  # GPa
VC = V_tio2 * C * conversion

Q_CN = []
Q_TLS = []
for i in range(0, 10):
    Q_CN.append(np.load('./a_TiO2/TiO2_data_out/Heat_L_CN_v_w_300K_sample_' + str(i + 1) + '.npy'))
    Q_TLS.append(np.load('./a_TiO2/TiO2_data_out/Heat_TLS_v_w_300K_sample_' + str(i + 1) + '.npy'))

Q_CN = np.array(Q_CN)
Q_CN = np.where(Q_CN < 0, 0, Q_CN)
Q_TLS = np.array(Q_TLS)

# TiO2 network sizes used to scale each sample
Ti_network_size = np.array([431, 483, 9, 641, 505, 1349, 205, 1316, 319, 121])
for i in range(0, 10):
    Q_CN[i, :] = Q_CN[i, :] * Ti_network_size[i]

# Frequency grids for TiO2 CN and TLS curves
f = np.load('./a_TiO2/TiO2_data_out/w_CN_v_w_300K_sample_1.npy')
f_TLS = np.load('./a_TiO2/TiO2_data_out/w_TLS_v_w_300K_sample_1.npy')

# SEMs computed in original script (not plotted)
Q_CN_SEM = stats.sem(Q_CN / (4 * Gamma**2.0 * np.pi), axis=0)
Q_TLS_SEM = stats.sem(Q_TLS, axis=0)

axs.loglog(f, (np.nanmean(Q_CN, axis=0)) / (4 * Gamma**2.0 * np.pi * VC), 'r')
axs.semilogy(f_TLS, np.mean(Q_TLS, axis=0) / VC, 'r--')

# ===============================
# Axes formatting
# ===============================
axs.set_xlabel('Frequency (Hz)', fontsize=14)
axs.set_ylabel('Mechanical Loss', fontsize=14)
axs.set_ylim(0.5 * 10**(-4), 10**2)
axs.set_xlim(10**(-1), 10**13)
axs.set_xticks([10**0, 10**4, 10**8, 10**12],
               ['$10^{0}$', '$10^{4}$', '$10^{8}$', '$10^{12}$'],
               fontsize=12)
axs.set_yticks([10**(-4), 10**(-2), 10**0, 10**2],
               ['$10^{-4}$', '$10^{-2}$', '$10^{0}$', '$10^{2}$'],
               fontsize=12)

plt.tight_layout()

# fig.savefig('Q_Si_TiO2_combined_frequency_only.png', dpi=600)
plt.show()
