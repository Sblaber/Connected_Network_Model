import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.linalg import eig
from numpy.linalg import pinv

# --- Plot style  ---
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# -------------------------------
# Parameters 
# -------------------------------
N_omega = 201
k_B = 8.6175e-5  # eV/K
T = 300          # K
Gamma = 1
N_nodes = 4

# Sweep two cases 
Energy_vec = np.array((0, 0.4))

for Energy_index in range(0, 2):

    # Energies of the 4 nodes
    Energy = np.array((0, 0, 0, 0))  
    beta = 1 / (k_B * T)

    # Barrier/edge parameters 
    V12 = 0.6
    V13 = 0
    V14 = Energy_vec[Energy_index]
    V23 = 0.4
    V24 = 0
    V34 = 0.4

    k0 = 10**13  # s^{-1}

    # Symmetric barrier matrix Vij 
    Vij = np.array(
        ([0,   V12, V13, V14],
         [V12, 0,   V23, V24],
         [V13, V23, 0,   V34],
         [V14, V24, V34, 0])
    )

    # Alternating-sign vector a and corresponding Gamma_ij outer product
    a = np.ones(N_nodes)
    for i in range(0, N_nodes):
        a[i] = a[i] * (-1)**i
    Gamma_ij = np.einsum('i,j->ij', a, a)

    # -------------------------------
    # Build rate/generator matrix R
    # Convention: off-diagonal R[i,j] are transition rates j -> i
    # Diagonal R[i,i] = -sum_i R[i, i_col] so each column sums to zero
    # -------------------------------
    R = np.zeros((N_nodes, N_nodes))
    for i in range(0, N_nodes):
        for j in range(0, N_nodes):
            if Vij[i, j] != 0 and i != j:
                R[i, j] = k0 * np.exp(beta * Energy[j]) * np.exp(-beta * Vij[i, j])

    for i in range(0, N_nodes):
        R[i, i] = -np.sum(R[:, i])

    # -------------------------------
    # Eigen-decomposition of R 
    # -------------------------------
    eigenvalues, eigenvectors = eig(R, left=False)

    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    eigenvectors_inv = pinv(eigenvectors)

    # Equilibrium distribution over nodes 
    P_0 = np.exp(-beta * Energy) / np.sum(np.exp(-beta * Energy))

    # Arrays used in final plot
    Q = np.zeros(N_omega)
    frequency = np.zeros(N_omega)

    # -------------------------------
    # Frequency sweep: compute Q(omega)
    # -------------------------------
    for Omega_index in range(0, N_omega):
        print(Omega_index)

        # Frequency grid definition 
        omega = 1.25**(-Omega_index) * 1e15 / np.pi

        # Response factors in eigenmode basis 
        D = np.zeros(N_nodes)
        D_2 = np.zeros(N_nodes)
        for i in range(0, len(eigenvalues) - 1):
            if eigenvalues[i] < -10**(-30.0):
                D[i] = (2 * np.pi * omega * eigenvalues[i]) / (eigenvalues[i]**2.0 + (2 * np.pi * omega)**2.0)
                D_2[i] = 1 / (eigenvalues[i]**2.0 + (2 * np.pi * omega)**2.0)

        # Construct T_ij and compute scalar Q 
        T_ij = -np.pi * beta * Gamma**2.0 * np.einsum(
            'ij,j,jk,k->ik',
            eigenvectors, D, eigenvectors_inv, P_0,
            optimize='optimal'
        )

        # The plotted quantity uses Q = sum(abs(T_ij)) equivalent to normalized alternating sign perturbation for 4-state system
        Q[Omega_index] = np.sum(np.abs(T_ij))
        frequency[Omega_index] = omega

    # -------------------------------
    # "TLS sum" analytic curve block 
    # -------------------------------
    V_TLS = np.array([
        V12 - 0.5 * (Energy[0] + Energy[1]),
        0,
        0,
        V23 - 0.5 * (Energy[1] + Energy[2]),
        0,
        V34 - 0.5 * (Energy[2] + Energy[3])
    ])

    D_TLS = np.array([
        Energy[1] - Energy[0],
        Energy[2] - Energy[0],
        Energy[3] - Energy[0],
        Energy[2] - Energy[1],
        Energy[3] - Energy[1],
        Energy[3] - Energy[2]
    ])

    w = np.logspace(-3, 15, 2001)
    Q_inv = np.zeros((len(D_TLS), len(w)))

    T_TLS = 1.0 * T
    t_0 = 1 / k0
    gamma_TLS = 1

    for i in range(0, len(D_TLS)):
        if V_TLS[i] != 0:
            t_TLS = t_0 * (np.cosh(D_TLS[i] / (2.0 * k_B * T_TLS)))**(-1.0) * np.exp(V_TLS[i] / (k_B * T_TLS)) / 2.0
            A = (1.0 / (4.0 * k_B * T_TLS)) * np.cosh(D_TLS[i] / (2.0 * k_B * T_TLS))**(-2.0)
            Q_inv[i, :] = A * ((w * t_TLS) / (1 + w**2.0 * t_TLS**2.0)) * gamma_TLS

    # -------------------------------
    # Plotting 
    # -------------------------------
    plt.figure(1, figsize=(4, 3))

    if Energy_index == 0:
        plt.loglog(2 * np.pi * frequency, N_nodes * Q / (4 * Gamma**2.0 * np.pi), 'r', label='$V_{14} = 0$')
        plt.semilogx(w, np.sum(Q_inv, axis=0), 'k')

    if Energy_index == 1:
        plt.loglog(2 * np.pi * frequency, N_nodes * Q / (4 * Gamma**2.0 * np.pi), 'r--', label='$V_{14} = 0.4$')
        plt.semilogx(w, np.sum(Q_inv, axis=0), 'k--')

# Final figure formatting and save 
plt.legend(frameon=False)
plt.xlabel('Frequency (Hz)', fontsize=14)
plt.ylabel('Dissipation$^*$', fontsize=14)
plt.xticks([10**0, 10**4, 10**8, 10**12], ['$10^0$', '$10^4$', '$10^8$', '$10^{12}$'])
plt.yticks([10**(-6), 10**-4, 10**-2, 10**0, 10**2], ['$10^{-6}$', '$10^{-4}$', '$10^{-2}$', '$10^{0}$', '$10^2$'])
plt.xlim(10**(0), 10**12)
plt.ylim(10**(-6), 10**2)
plt.tight_layout()
plt.savefig('Q_cycle_v_chain.pdf')
plt.show()
