import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy.linalg import pinv
from scipy.linalg import eig

# --- Plot style ---
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# ================================
# Size / sampling parameters
# Note that with N_subgraph = 1000 
# and N_T = 100 it may take hours 
# to run
# ================================
N_subgraph = 1000   # number of nodes
N_omega = 301       # number of driving frequencies
N_T = 100           # number of graph/energy realizations (indexed by T_index; temperature is fixed below)
N_m = 4             # number of BA attachment parameters (m = 2^(m_index+1))

# Storage for spectra and TLS comparison curves (these drive the final plot)
Q = np.zeros((N_omega, N_T, N_m))
Q_TLS = np.zeros((2001, N_T, N_m))
frequency = np.zeros((N_omega, N_T))

# ===============================
# Physical parameters 
# ===============================
k0 = 10                 # rate constant in ps^{-1}
t_0 = 10**(-13.0)        # s
k_B = 8.617e-5           # eV/K
T = 300                  # K
beta = (k_B * T)**(-1.0)
Gamma = 1

# Working arrays (reused each realization)
Energy = np.ones(N_subgraph)
Gamma_ij = np.zeros((N_subgraph, N_subgraph))

# Lists used to build TLS distributions for each graph realization
V_TLS = []
D_TLS = []

Gamma_vec = np.ones(N_subgraph)

for m_index in range(0, N_m):
    for T_index in range(0, N_T):
        print(T_index)

        # -------------------------------
        # Random graph generation
        # -------------------------------
        G = nx.barabasi_albert_graph(
            n=N_subgraph,
            m=2**(m_index + 1),
            seed=T_index * 100,
            initial_graph=None
        )

        # -------------------------------
        # Random node energies and random edge barriers
        # -------------------------------
        np.random.seed(T_index * 7)
        Energy = np.random.uniform(0, 0.5, N_subgraph)

        V_ij = np.zeros((N_subgraph, N_subgraph))
        V_TLS_1 = []
        D_TLS_1 = []

        # Assign barrier to each undirected edge and store TLS parameters
        edge_index = 0
        for u, v in G.edges:
            E1_temp = Energy[u]
            E2_temp = Energy[v]

            V_min = np.amax([E1_temp + 0.01, E2_temp + 0.01])
            V_temp = np.random.uniform(V_min, V_min + 1)

            V_ij[u, v] = V_temp
            V_ij[v, u] = V_temp

            V_TLS_1.append(V_temp - (E1_temp + E2_temp) / 2)
            D_TLS_1.append(E1_temp - E2_temp)

            # Edge attributes are not used in the final plot but are kept for provenance/debugging parity
            G.add_edge(
                u, v,
                E1=E1_temp,
                E2=E2_temp,
                V=V_temp,
                energy_barrier=V_temp - (E1_temp + E2_temp) / 2,
                energy_asymmetry=(E1_temp - E2_temp),
                index=edge_index
            )
            edge_index += 1

        V_TLS = np.array(V_TLS_1)
        D_TLS = np.array(D_TLS_1)

        # -------------------------------
        # Build rate/generator matrix R
        # Convention: off-diagonal R[i,j] are transition rates j -> i
        # Diagonal enforces column-sum zero.
        # -------------------------------
        R = np.zeros((N_subgraph, N_subgraph))

        # gamma_TLS is used later as an edge-weighting factor in the TLS sum.
        gamma_TLS = np.zeros(len(G.edges))

        E = Energy
        for i in range(0, N_subgraph):
            for j in range(0, N_subgraph):
                if i != j and V_ij[i, j] != 0:
                    R[i, j] = k0 * np.exp(beta * (E[j] - V_ij[i, j]))

                    gamma_TLS[i] = 1

                    # Alternating-sign assignment used to construct Gamma_ij
                    if Gamma_vec[i] == 1:
                        Gamma_vec[j] = -1
                    else:
                        Gamma_vec[j] = 1

        for i in range(0, N_subgraph):
            R[i, i] = -np.sum(R[:, i])

        Gamma_ij = np.einsum('i,j->ij', Gamma_vec, Gamma_vec)

        # -------------------------------
        # Eigen-decomposition and equilibrium distribution
        # -------------------------------
        eigenvalues, eigenvectors = eig(R, left=False)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        eigenvectors_inv = pinv(eigenvectors)

        P_0 = np.exp(-beta * Energy) / np.sum(np.exp(-beta * Energy))

        # -------------------------------
        # Frequency sweep: compute dissipation spectrum Q(omega)
        # -------------------------------
        for Omega_index in range(0, N_omega):
            omega = 1.25**(-Omega_index + 100) * 10**13.0 * 10**(-12.0) / np.pi  # ps^{-1} scale

            D = np.zeros(N_subgraph)
            for i in range(0, len(eigenvalues) - 1):
                if eigenvalues[i] < -10**(-30.0):
                    D[i] = (2 * np.pi * omega * eigenvalues[i]) / (eigenvalues[i]**2.0 + (2 * np.pi * omega)**2.0)

            T_ij = -np.pi * beta * Gamma**2.0 * np.einsum(
                'ij,j,jk,k->ik',
                eigenvectors, D, eigenvectors_inv, P_0,
                optimize='optimal'
            )

            Q[Omega_index, T_index, m_index] = np.einsum(
                'ij,ij->',
                T_ij, Gamma_ij,
                optimize='optimal'
            )

            frequency[Omega_index, T_index] = omega * 10**12.0  # Hz

        # -------------------------------
        # Independent TLS sum (analytic comparison curve)
        # -------------------------------
        w = np.logspace(-6, 18, 2001)
        T_TLS = T
        Q_inv = np.zeros((len(D_TLS), len(w)))

        for i in range(0, len(D_TLS)):
            t_TLS = t_0 * (np.cosh(D_TLS[i] / (2.0 * k_B * T_TLS)))**(-1.0) * np.exp(V_TLS[i] / (k_B * T_TLS)) / 2.0
            A = (1.0 / (4.0 * k_B * T_TLS)) * np.cosh(D_TLS[i] / (2.0 * k_B * T_TLS))**(-2.0)
            Q_inv[i, :] = A * ((w * t_TLS) / (1 + w**2.0 * t_TLS**2.0)) * gamma_TLS[i]

        Q_TLS[:, T_index, m_index] = np.sum(Q_inv, axis=0)

    # Connectivity diagnostic
    print(' Percent connected')
    print(100 * len(G.edges) / N_subgraph**2.0)

# ===============================
# Final plot
# ===============================
plt.figure()
plt.axvspan(10**0, 10**4, alpha=0.1, color='k')

# TLS mean curve (sparsely sampled points)
plt.loglog(
    w[0:-1:50],
    np.mean(np.mean(Q_TLS[:, :, :], axis=1), axis=1)[0:-1:50],
    'ks'
)

# Dissipation spectra for each m_index with shaded uncertainty bands
x = 2 * np.pi * frequency[:, 0]

# m_index = 0 (red)
plt.loglog(2 * np.pi * frequency[:, 0], N_subgraph * np.mean(Q[:, :, 0], axis=1) / (4 * Gamma**2.0 * np.pi), 'r')
mean = N_subgraph * np.mean(Q[:, :, 0], axis=1) / (4 * Gamma**2.0 * np.pi)
std = N_T * np.std(np.array(Q[:, :, 0]) / (4 * Gamma**2.0 * np.pi), axis=1)
plt.fill_between(x, mean - std, y2=mean + std, color='r', alpha=0.3)

# m_index = 1 (green dashed)
plt.loglog(2 * np.pi * frequency[:, 0], N_subgraph * np.mean(Q[:, :, 1], axis=1) / (4 * Gamma**2.0 * np.pi), 'g--')
mean = N_subgraph * np.mean(Q[:, :, 1], axis=1) / (4 * Gamma**2.0 * np.pi)
std = N_T * np.std(np.array(Q[:, :, 1]) / (4 * Gamma**2.0 * np.pi), axis=1)
plt.fill_between(x, mean - std, y2=mean + std, color='g', alpha=0.3)

# m_index = 2 (blue dotted)
plt.loglog(2 * np.pi * frequency[:, 0], N_subgraph * np.mean(Q[:, :, 2], axis=1) / (4 * Gamma**2.0 * np.pi), 'b:')
mean = N_subgraph * np.mean(Q[:, :, 2], axis=1) / (4 * Gamma**2.0 * np.pi)
std = N_T * np.std(np.array(Q[:, :, 2]) / (4 * Gamma**2.0 * np.pi), axis=1)
plt.fill_between(x, mean - std, y2=mean + std, color='b', alpha=0.3)

# m_index = 3 (magenta dash-dot)
plt.loglog(2 * np.pi * frequency[:, 0], N_subgraph * np.mean(Q[:, :, 3], axis=1) / (4 * Gamma**2.0 * np.pi), 'm-.')
mean = N_subgraph * np.mean(Q[:, :, 3], axis=1) / (4 * Gamma**2.0 * np.pi)
std = N_T * np.std(np.array(Q[:, :, 3]) / (4 * Gamma**2.0 * np.pi), axis=1)
plt.fill_between(x, mean - std, y2=mean + std, color='m', alpha=0.3)

plt.xlim(10**(-1), 10**11)
plt.ylim(10**0, 2 * 10**3)

plt.ylabel('Dissipation$^*$', fontsize=26)
plt.xlabel('Frequency (Hz)', fontsize=26)

plt.xticks([10**(-1), 10**3, 10**7, 10**11], fontsize=16)
plt.yticks([10**0, 10**1, 10**2, 10**3], fontsize=16)

plt.tight_layout()
plt.show()
