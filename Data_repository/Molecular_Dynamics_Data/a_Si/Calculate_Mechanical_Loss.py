"""
Compute frequency-dependent dissipation measures (Q, Q_L, Q_T, etc.) for a-Si
configuration-network dynamics, and compare to a TLS-style sum over individual
transitions.

This script is intended to generate (and optionally save) the frequency-dependent
quantities Q_L and Q_T. The .npy save calls are included but left commented out.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import networkx as nx
from numpy.linalg import pinv
from scipy.linalg import eig
from datetime import datetime

# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)

startTime = datetime.now()

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
Gamma = 0.01 # amplitude of oscillations. This does not change Q as currently defined
k0 = 10  # rate constant in ps^{-1}

k_B = 8.617e-5  # Blotzmann constant in eV/K
T = 300         # Temperature in K 
beta = 1.0 / (k_B * T)

N_omega = 201 # number of frequency steps
N_T = 1 # number of temperature steps
N_graph_max = 3000 # Largest network

# Data directory
base = "./Data"

# -----------------------------------------------------------------------------
# Loop over samples
# -----------------------------------------------------------------------------
for sample_index in range(0, 10):
    N_subgraph_0 = N_graph_max
    N_subgraph = 1 * N_subgraph_0
    N_sample = 1 + sample_index

    # --- Load raw arrays for this sample ---
    # Fot a-Si all the data is in one big array with different samples along the first axis.
    V = np.load(f"{base}/V_monotonicP_wofirstlast_sample_{10}.npy")[sample_index, :, :]
    V = V[~np.isnan(V)][0:N_graph_max]

    print(N_graph_max)
    print(V[np.where(V < 666)].shape)

    Connections = np.load(f"{base}/Connections{10}.npy")[0:N_graph_max]
    D = np.load(f"{base}/D_asymm_sanity_monotonicP_wofirstlast_sample_{10}.npy")[sample_index, :, :]
    D = D[~np.isnan(D)][0:N_graph_max]

    E_1 = np.load(f"{base}/E_initial_monotonicP_wofirstlast_sample_{10}.npy")[sample_index, :, :]
    E_2 = np.load(f"{base}/E_final_monotonicP_wofirstlast_sample_{10}.npy")[sample_index, :, :]

    E_1 = E_1[~np.isnan(E_1)][0:N_graph_max]
    E_offset = E_1.min()

    # Energies shifted by E_offset
    E_1 = E_1[~np.isnan(E_1)] - E_offset
    E_2 = E_2[~np.isnan(E_2)][0:N_graph_max] - E_offset
    E_array = np.transpose(np.vstack((E_1, E_2)))

    # Barriers shifted by same offset
    V = V - E_offset

    # Determine unique node labels excluding failed NEB calculations with error code '666' 
    C_flat = Connections.flatten()
    E_flat = E_array.flatten()
    unique_values, unique_indices = np.unique(
        C_flat[(E_flat + E_offset) != 666], return_index=True
    )
    E_unique = E_flat[(E_flat + E_offset) != 666][unique_indices]

    # --- Load stress data using list_*.txt (sample-specific indexing) ---
    list_1 = np.loadtxt(f"{base}/list_{sample_index + 1}.txt").astype(int)

    stress_conv = 6.242e-7  # eV/A^3 conversion factor

    s_1 = np.load(f"{base}/s_1.npy")[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_1_before = np.load(f"{base}/s_1.npy")[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv

    s_2 = np.load(f"{base}/s_2.npy")[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_2_before = np.load(f"{base}/s_2.npy")[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv

    s_3 = np.load(f"{base}/s_3.npy")[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_3_before = np.load(f"{base}/s_3.npy")[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv

    s_4 = np.load(f"{base}/s_4.npy")[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_4_before = np.load(f"{base}/s_4.npy")[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv

    s_5 = np.load(f"{base}/s_5.npy")[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_5_before = np.load(f"{base}/s_5.npy")[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv

    s_6 = np.load(f"{base}/s_6.npy")[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_6_before = np.load(f"{base}/s_6.npy")[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv

    Volume = 27.43**3.0

    # -----------------------------------------------------------------------------
    # Build directed graph with edge attributes
    # -----------------------------------------------------------------------------
    G = nx.DiGraph()
    G.add_nodes_from(unique_values)

    i = 0
    for pair in Connections:
        if int(V[i] + E_offset) != 666 and (V[i] - E_1[i] > 0) and (V[i] - E_2[i] > 0):
            G.add_edge(
                pair[0],
                pair[1],
                energy_barrier=V[i],
                energy_asymmetry=D[i],
                E_1=E_1[i],
                E_2=E_2[i],
                energy_barrier_TLS=V[i] - 0.5 * (E_1[i] + E_2[i]),
                g_xx=s_1[sample_index, i],
                g_yy=s_2[sample_index, i],
                g_zz=s_3[sample_index, i],
                g_xy=s_4[sample_index, i],
                g_xz=s_5[sample_index, i],
                g_yz=s_6[sample_index, i],
                g_xx_b=s_1_before[sample_index, i],
                g_yy_b=s_2_before[sample_index, i],
                g_zz_b=s_3_before[sample_index, i],
                g_xy_b=s_4_before[sample_index, i],
                g_xz_b=s_5_before[sample_index, i],
                g_yz_b=s_6_before[sample_index, i],
            )
        i += 1

    # Keep largest connected component
    S = [G.subgraph(c).copy() for c in nx.connected_components(G.to_undirected())]
    node_number = [len(sg.nodes()) for sg in S]
    max_graph_idx = int(np.argmax(node_number))
    G = S[max_graph_idx]

    # Use undirected graph for attribute extraction and rates
    J = G.to_undirected()
    N_subgraph = len(list(J.nodes))
    print(N_subgraph)

    # -----------------------------------------------------------------------------
    # Extract edge attributes
    # -----------------------------------------------------------------------------
    V_dict = nx.get_edge_attributes(J, "energy_barrier")
    E1_dict = nx.get_edge_attributes(J, "E_1")
    E2_dict = nx.get_edge_attributes(J, "E_2")

    g_xx_dict = nx.get_edge_attributes(J, "g_xx")
    g_yy_dict = nx.get_edge_attributes(J, "g_yy")
    g_zz_dict = nx.get_edge_attributes(J, "g_zz")
    g_xy_dict = nx.get_edge_attributes(J, "g_xy")
    g_xz_dict = nx.get_edge_attributes(J, "g_xz")
    g_yz_dict = nx.get_edge_attributes(J, "g_yz")

    g_xx_b_dict = nx.get_edge_attributes(J, "g_xx_b")
    g_yy_b_dict = nx.get_edge_attributes(J, "g_yy_b")
    g_zz_b_dict = nx.get_edge_attributes(J, "g_zz_b")
    g_xy_b_dict = nx.get_edge_attributes(J, "g_xy_b")
    g_xz_b_dict = nx.get_edge_attributes(J, "g_xz_b")
    g_yz_b_dict = nx.get_edge_attributes(J, "g_yz_b")

    Nodes = np.array(J.nodes)

    # -----------------------------------------------------------------------------
    # Allocate arrays
    # -----------------------------------------------------------------------------
    Q = np.zeros((N_omega, N_T))
    Q_L = np.zeros((N_omega, N_T))
    Q_T = np.zeros((N_omega, N_T))

    Q_eigen = np.zeros((N_subgraph, N_omega, N_T))
    frequency = np.zeros((N_omega, N_T))
    T_vec = np.zeros((N_omega, N_T))

    Energy = np.ones(N_subgraph)
    Vij = np.zeros((N_subgraph, N_subgraph))
    dE = np.zeros((N_subgraph, N_subgraph))

    Gamma_ij = np.zeros((N_subgraph, N_subgraph))
    V_TLS = []
    D_TLS = []

    # Deformation potential vector/tensor used to form Gamma_L_ij and Gamma_T_ij
    Gamma_vec = np.ones(N_subgraph)
    Gamma_tens = np.zeros((6, N_subgraph))

    # -----------------------------------------------------------------------------
    # Populate energies, Vij, and deformation-potential tensors
    # -----------------------------------------------------------------------------
    for i in range(N_subgraph):
        Node_1 = Nodes[i]
        for j in range(N_subgraph):
            Node_2 = Nodes[j]

            if (Node_1, Node_2) in V_dict:
                V_edge = V_dict.get((Node_1, Node_2))
                E1_temp = E1_dict.get((Node_1, Node_2))
                E2_temp = E2_dict.get((Node_1, Node_2))

                Vij[i, j] = V_edge
                Vij[j, i] = V_edge
                Energy[j] = 1.0 * E2_temp

                V_TLS.append(V_edge - 0.5 * (E2_temp + E1_temp))
                D_TLS.append(E2_temp - E1_temp)

                g_xx = g_xx_dict.get((Node_1, Node_2)) - g_xx_b_dict.get((Node_1, Node_2))
                g_yy = g_yy_dict.get((Node_1, Node_2)) - g_yy_b_dict.get((Node_1, Node_2))
                g_zz = g_zz_dict.get((Node_1, Node_2)) - g_zz_b_dict.get((Node_1, Node_2))
                g_xy = g_xy_dict.get((Node_1, Node_2)) - g_xy_b_dict.get((Node_1, Node_2))
                g_xz = g_xz_dict.get((Node_1, Node_2)) - g_xz_b_dict.get((Node_1, Node_2))
                g_yz = g_yz_dict.get((Node_1, Node_2)) - g_yz_b_dict.get((Node_1, Node_2))

                Gamma_ij[i, j] = (
                    1
                    - (1.0 / 5.0) * (g_xx * g_xx + g_yy * g_yy + g_zz * g_zz)
                    - (2.0 / 15.0) * (g_xx * g_yy + g_xx * g_yy + g_xx * g_zz)
                    - (4.0 / 15.0) * (g_xy * g_xy + g_xz * g_xz + g_yz * g_yz)
                )
                Gamma_ij[j, i] = 1.0 * Gamma_ij[i, j]

                # Vector/tensor population
                Gamma_vec[j] = 1.0 * g_xx
                Gamma_vec[i] = -1.0 * g_xx

                Gamma_tens[0, j] = 1.0 * g_xx
                Gamma_tens[0, i] = -1.0 * g_xx
                Gamma_tens[1, j] = 1.0 * g_yy
                Gamma_tens[1, i] = -1.0 * g_yy
                Gamma_tens[2, j] = 1.0 * g_zz
                Gamma_tens[2, i] = -1.0 * g_zz
                Gamma_tens[3, j] = 1.0 * g_xy
                Gamma_tens[3, i] = -1.0 * g_xy
                Gamma_tens[4, j] = 1.0 * g_xz
                Gamma_tens[4, i] = -1.0 * g_xz
                Gamma_tens[5, j] = 1.0 * g_yz
                Gamma_tens[5, i] = -1.0 * g_yz

            elif (Node_2, Node_1) in V_dict:
                V_edge = V_dict.get((Node_2, Node_1))
                E1_temp = E1_dict.get((Node_2, Node_1))
                E2_temp = E2_dict.get((Node_2, Node_1))

                Vij[i, j] = V_edge
                Vij[j, i] = V_edge
                Energy[j] = 1.0 * E1_temp

                V_TLS.append(V_edge - 0.5 * (E2_temp + E1_temp))
                D_TLS.append(E2_temp - E1_temp)

                g_xx = g_xx_dict.get((Node_2, Node_1)) - g_xx_b_dict.get((Node_2, Node_1))
                g_yy = g_yy_dict.get((Node_2, Node_1)) - g_yy_b_dict.get((Node_2, Node_1))
                g_zz = g_zz_dict.get((Node_2, Node_1)) - g_zz_b_dict.get((Node_2, Node_1))
                g_xy = g_xy_dict.get((Node_2, Node_1)) - g_xy_b_dict.get((Node_2, Node_1))
                g_xz = g_xz_dict.get((Node_2, Node_1)) - g_xz_b_dict.get((Node_2, Node_1))
                g_yz = g_yz_dict.get((Node_2, Node_1)) - g_yz_b_dict.get((Node_2, Node_1))

                Gamma_ij[i, j] = (
                    1
                    - (1.0 / 5.0) * (g_xx * g_xx + g_yy * g_yy + g_zz * g_zz)
                    - (2.0 / 15.0) * (g_xx * g_yy + g_xx * g_yy + g_xx * g_zz)
                    - (4.0 / 15.0) * (g_xy * g_xy + g_xz * g_xz + g_yz * g_yz)
                )
                Gamma_ij[j, i] = 1.0 * Gamma_ij[i, j]

                Gamma_tens[0, j] = 1.0 * g_xx
                Gamma_tens[0, i] = -1.0 * g_xx
                Gamma_tens[1, j] = 1.0 * g_yy
                Gamma_tens[1, i] = -1.0 * g_yy
                Gamma_tens[2, j] = 1.0 * g_zz
                Gamma_tens[2, i] = -1.0 * g_zz
                Gamma_tens[3, j] = 1.0 * g_xy
                Gamma_tens[3, i] = -1.0 * g_xy
                Gamma_tens[4, j] = 1.0 * g_xz
                Gamma_tens[4, i] = -1.0 * g_xz
                Gamma_tens[5, j] = 1.0 * g_yz
                Gamma_tens[5, i] = -1.0 * g_yz

    # Overwrite Gamma_ij using Gamma_vec outer product
    Gamma_ij = np.einsum("i,j->ij", Gamma_vec, Gamma_vec)

    # Longitudinal / transverse couplings built from Gamma_tens
    Gamma_tens_ij = np.einsum("ij,kl->ikjl", Gamma_tens, Gamma_tens)
    Gamma_L_ij = (
        (1.0 / 5.0) * (Gamma_tens_ij[0, 0, :, :] + Gamma_tens_ij[1, 1, :, :] + Gamma_tens_ij[2, 2, :, :])
        + (2.0 / 15.0) * (Gamma_tens_ij[0, 1, :, :] + Gamma_tens_ij[0, 2, :, :] + Gamma_tens_ij[1, 2, :, :])
        + (4.0 / 15.0) * (Gamma_tens_ij[3, 3, :, :] + Gamma_tens_ij[4, 4, :, :] + Gamma_tens_ij[5, 5, :, :])
    )
    Gamma_T_ij = (
        (1.0 / 15.0) * (Gamma_tens_ij[0, 0, :, :] + Gamma_tens_ij[1, 1, :, :] + Gamma_tens_ij[2, 2, :, :])
        - (1.0 / 15.0) * (Gamma_tens_ij[0, 1, :, :] + Gamma_tens_ij[0, 2, :, :] + Gamma_tens_ij[1, 2, :, :])
        + (3.0 / 15.0) * (Gamma_tens_ij[3, 3, :, :] + Gamma_tens_ij[4, 4, :, :] + Gamma_tens_ij[5, 5, :, :])
    )

    V_TLS = np.array(V_TLS)
    D_TLS = np.array(D_TLS)

    # -----------------------------------------------------------------------------
    # Rate matrix, eigen-decomposition, and dissipation vs omega
    # -----------------------------------------------------------------------------
    for T_index in range(0, N_T):
        # With N_T=1 this does nothing, allows for multiple T calculations
        T = T - 100 * T_index
        beta = 1.0 / (k_B * T)

        R = np.zeros((N_subgraph, N_subgraph)).astype(np.longdouble)
        dV = np.zeros((N_subgraph, N_subgraph))

        for i in range(N_subgraph):
            Node_1 = Nodes[i]
            for j in range(N_subgraph):
                Node_2 = Nodes[j]

                if (Node_1, Node_2) in V_dict:
                    V_edge = np.longdouble(V_dict.get((Node_1, Node_2)))
                    E1_temp = np.longdouble(E1_dict.get((Node_1, Node_2)))
                    E2_temp = np.longdouble(E2_dict.get((Node_1, Node_2)))

                    dE[i, j] = E1_temp - E2_temp
                    dE[j, i] = E2_temp - E1_temp

                    dV[i, j] = (V_edge - E2_temp)
                    dV[j, i] = (V_edge - E1_temp)

                    R[i, j] = np.longdouble(k0 * (np.exp(-beta * (V_edge - E2_temp))))
                    R[j, i] = np.longdouble(k0 * (np.exp(-beta * (V_edge - E1_temp))))

                elif (Node_2, Node_1) in V_dict:
                    V_edge = np.longdouble(V_dict.get((Node_2, Node_1)))
                    E1_temp = np.longdouble(E1_dict.get((Node_2, Node_1)))
                    E2_temp = np.longdouble(E2_dict.get((Node_2, Node_1)))

                    dV[i, j] = (V_edge - E1_temp)
                    dV[j, i] = (V_edge - E2_temp)

                    dE[i, j] = E1_temp - E2_temp
                    dE[j, i] = E2_temp - E1_temp

                    R[i, j] = np.longdouble(k0 * (np.exp(-beta * (V_edge - E1_temp))))
                    R[j, i] = np.longdouble(k0 * (np.exp(-beta * (V_edge - E2_temp))))

        # Diagonal terms enforce column-sum zero
        for i in range(N_subgraph):
            R[i, i] = -np.sum(np.longdouble(R[:, i]))

        # Eigen-decomposition
        R_inv = pinv(R.astype(np.double))
        eigenvalues, eigenvectors = eig(R, left=False)

        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real

        P_0 = np.exp(-beta * Energy) / np.sum(np.exp(-beta * Energy))

        eigenvectors_inv = pinv(eigenvectors)

        for Omega_index in range(0, N_omega):
            omega = 1.25 ** (-Omega_index) * 1e13 * 1e-12 / np.pi

            D_mode = np.zeros(N_subgraph)
            D_slow = np.zeros(N_subgraph)

            for i in range(0, len(eigenvalues)):
                if eigenvalues[i] < -1e-30:
                    D_mode[i] = (2 * np.pi * omega * eigenvalues[i]) / (eigenvalues[i] ** 2 + (2 * np.pi * omega) ** 2)
                    D_slow[i] = eigenvalues[i] ** (-1.0)

            T_ij = -np.pi * beta * Gamma**2.0 * np.einsum(
                "ij,j,jk,k->ik", eigenvectors, D_mode, eigenvectors_inv, P_0, optimize="optimal"
            )

            Q[Omega_index, T_index] = np.einsum("ij,ij->", T_ij, Gamma_ij, optimize="optimal")
            Q_L[Omega_index, T_index] = np.einsum("ij,ij->", Gamma_L_ij, T_ij, optimize="optimal")
            Q_T[Omega_index, T_index] = np.einsum("ij,ij->", Gamma_T_ij, T_ij, optimize="optimal")

            Q_eigen[:, Omega_index, T_index] = -np.pi * beta * Gamma**2.0 * np.einsum(
                "ij,j,jk,k,ik->j", eigenvectors, D_mode, eigenvectors_inv, P_0, Gamma_ij, optimize="optimal"
            )

            frequency[Omega_index, T_index] = omega * 1e12
            T_vec[Omega_index, T_index] = 1.0 * T

        # -----------------------------------------------------------------------------
        # TLS-style dissipation sum
        # -----------------------------------------------------------------------------
        t_0 = 1e-13  # s

        V_TLS = list(nx.get_edge_attributes(J, "energy_barrier_TLS").values())
        D_TLS = list(nx.get_edge_attributes(J, "energy_asymmetry").values())

        E1_TLS = np.array(list(nx.get_edge_attributes(J, "E_1").values()))
        E2_TLS = np.array(list(nx.get_edge_attributes(J, "E_2").values()))

        Z_TLS = np.sum(np.exp(-beta * E1_TLS)) + np.sum(np.exp(-beta * E2_TLS))
        P_TLS_correction = (np.exp(-beta * E1_TLS) + np.exp(-beta * E2_TLS)) / Z_TLS

        g_xx_TLS = np.array(list(nx.get_edge_attributes(J, "g_xx").values()))
        g_yy_TLS = np.array(list(nx.get_edge_attributes(J, "g_yy").values()))
        g_zz_TLS = np.array(list(nx.get_edge_attributes(J, "g_zz").values()))
        g_xy_TLS = np.array(list(nx.get_edge_attributes(J, "g_xy").values()))
        g_xz_TLS = np.array(list(nx.get_edge_attributes(J, "g_xz").values()))
        g_yz_TLS = np.array(list(nx.get_edge_attributes(J, "g_yz").values()))

        g_xx_b_TLS = np.array(list(nx.get_edge_attributes(J, "g_xx_b").values()))
        g_yy_b_TLS = np.array(list(nx.get_edge_attributes(J, "g_yy_b").values()))
        g_zz_b_TLS = np.array(list(nx.get_edge_attributes(J, "g_zz_b").values()))
        g_xy_b_TLS = np.array(list(nx.get_edge_attributes(J, "g_xy_b").values()))
        g_xz_b_TLS = np.array(list(nx.get_edge_attributes(J, "g_xz_b").values()))
        g_yz_b_TLS = np.array(list(nx.get_edge_attributes(J, "g_yz_b").values()))

        g_xx_TLS -= g_xx_b_TLS
        g_yy_TLS -= g_yy_b_TLS
        g_zz_TLS -= g_zz_b_TLS
        g_xy_TLS -= g_xy_b_TLS
        g_xz_TLS -= g_xz_b_TLS
        g_yz_TLS -= g_yz_b_TLS

        gamma_TLS = (
            (1.0 / 5.0) * (g_xx_TLS**2.0 + g_yy_TLS**2.0 + g_zz_TLS**2.0)
            + (2.0 / 15.0) * (g_xx_TLS * g_yy_TLS + g_xx_TLS * g_zz_TLS + g_yy_TLS * g_zz_TLS)
            + (4.0 / 15.0) * (g_xy_TLS**2.0 + g_xz_TLS**2.0 + g_yz_TLS**2.0)
        )

        w = np.logspace(-6, 13, 2001)
        T_TLS = T

        Q_inv = np.zeros((len(D_TLS), len(w)))
        for i in range(0, len(D_TLS)):
            t_TLS = t_0 * (np.cosh(D_TLS[i] / (2.0 * k_B * T_TLS))) ** (-1.0) * np.exp(V_TLS[i] / (k_B * T_TLS)) / 2.0
            A = (1.0 / (4.0 * k_B * T_TLS)) * np.cosh(D_TLS[i] / (2.0 * k_B * T_TLS)) ** (-2.0)
            Q_inv[i, :] = A * ((w * t_TLS) / (1 + w**2.0 * t_TLS**2.0)) * gamma_TLS[i]

        Q_inv_corrected = np.einsum("ij,i->ij", Q_inv, P_TLS_correction)

        print("next sample")

        # -----------------------------------------------------------------------------
        # Plot results for this sample
        # -----------------------------------------------------------------------------
        plt.figure(10)
        plt.loglog(2 * np.pi * frequency[:, -1], Q_eigen.shape[0] * Q[:, 0] / (8 * Gamma**2.0 * np.pi), "r", alpha=0.5)
        plt.semilogx(2 * np.pi * frequency[:, -1], Q_eigen.shape[0] * Q_T[:, 0] / (8 * Gamma**2.0 * np.pi), "g", alpha=0.5)
        plt.semilogx(2 * np.pi * frequency[:, -1], Q_eigen.shape[0] * Q_L[:, 0] / (8 * Gamma**2.0 * np.pi), "b", alpha=0.5)

        plt.semilogx(w, np.sum(Q_inv, axis=0), "k-.", alpha=0.5)
        plt.grid()
        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel("Scaled Dissipation", fontsize=14)
        plt.tight_layout()

        # Optional saves
        # np.save("./Heat_Data/frequency_300K.npy", np.pi * frequency)
        # np.save(f"./Heat_Data/Q_L_300K_sample_{N_sample}.npy", Q_L)
        # np.save(f"./Heat_Data/Q_T_300K_sample_{N_sample}.npy", Q_T)
        # np.save(f"./Heat_Data/Heat_TLS_300K_sample_{N_sample}.npy", np.sum(Q_inv, axis=0))

plt.show()