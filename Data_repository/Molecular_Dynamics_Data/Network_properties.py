"""
Analyze network-derived energy-landscape data (a-Si) and compare simple structural
statistics (degree and cycle distributions) against TiO2 reference distributions.

This script:
  1) Loads a-Si transition/barrier/asymmetry data and stress-tensor components.
  2) Builds a directed NetworkX graph for each sample and extracts the largest
     weakly connected subgraph (then largest weakly connected component).
  3) For the first temperature index (T_index < 1; here N_T = 1), computes and
     stores:
        - basis-cycle size distribution (cycle_basis on undirected version)
        - degree distribution (on the generated/loaded graph)
        - energy / barrier / asymmetry distributions for later aggregation
  4) Loads TiO2 reference histograms from disk.
  5) Produces and saves:
        - degree_cycles_combined_binned.pdf
        - energy_distribution_combined.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import networkx as nx
from scipy import stats
from scipy.stats import binned_statistic

# --- Plot style ---
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# ===============================
# Physical / global parameters
# ===============================
Gamma = 0.01
k0 = 10                      # rate constant in ps^{-1}
k_B = 8.617e-5               # eV/K
T = 300                      # K
beta = (k_B * T) ** (-1.0)

# ===============================
# Analysis / sampling parameters
# ===============================
N_omega = 201
N_graph_max = 3000

# Base data directory
base = './a_Si/Data'

# Containers used later in final figures
degree_combined = []
cycles_combined = []
V_combined = []
E_combined = []
D_combined = []

# The following are built from the a-Si data and used later for binning/plotting
bins_cycles_1 = None
bins_degree_1 = None
hist_cycles_1 = None
hist_degree_1 = None

# ===============================
# Load and process a-Si samples
# ===============================
for sample_index in range(0, 10):

    N_subgraph_0 = N_graph_max
    N_subgraph = 1 * N_subgraph_0
    N_sample = 1 + sample_index

    # --- Load barrier/asymmetry/energy arrays and filter NaNs (pairs without barriers) ---
    V = np.load(base + '/V_monotonicP_wofirstlast_sample_' + str(N_sample) + '.npy')[sample_index, :, :]
    V = V[~np.isnan(V)]
    N_graph_max = len(V)

    Connections = np.load(base + '/Connections' + str(N_sample) + '.npy')[0:N_graph_max]

    D = np.load(base + '/D_asymm_sanity_monotonicP_wofirstlast_sample_' + str(N_sample) + '.npy')[sample_index, :, :]
    D = D[~np.isnan(D)]

    E_1 = np.load(base + '/E_initial_monotonicP_wofirstlast_sample_' + str(N_sample) + '.npy')[sample_index, :, :]
    E_2 = np.load(base + '/E_final_monotonicP_wofirstlast_sample_' + str(N_sample) + '.npy')[sample_index, :, :]

    E_1 = E_1[~np.isnan(E_1)][0:N_graph_max]
    E_offset = E_1.min()
    E_1 = E_1[~np.isnan(E_1)] - E_offset
    E_2 = E_2[~np.isnan(E_2)] - E_offset

    E_array = np.transpose(np.vstack((E_1, E_2)))
    V = V - E_offset

    # Determine unique node labels from Connections, excluding energies where NEB failed set to code number '666'
    C_flat = Connections.flatten()
    E_flat = E_array.flatten()
    unique_values, unique_indices = np.unique(
        C_flat[(E_flat + E_offset) != 666],
        return_index=True
    )
    E_unique = E_flat[(E_flat + E_offset) != 666][unique_indices]

    # --- Load stress tensor components and "before" components ---
    list_1 = np.loadtxt(base + '/list_' + str(sample_index + 1) + '.txt').astype(int)

    # Conversion factor used in the original script
    stress_conv = 6.242e-7  # eV/A^3

    s_1 = np.load(base + '/s_1.npy')[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_1_before = np.load(base + '/s_1.npy')[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv
    s_2 = np.load(base + '/s_2.npy')[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_2_before = np.load(base + '/s_2.npy')[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv
    s_3 = np.load(base + '/s_3.npy')[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_3_before = np.load(base + '/s_3.npy')[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv
    s_4 = np.load(base + '/s_4.npy')[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_4_before = np.load(base + '/s_4.npy')[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv
    s_5 = np.load(base + '/s_5.npy')[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_5_before = np.load(base + '/s_5.npy')[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv
    s_6 = np.load(base + '/s_6.npy')[:, list_1[:, 0], list_1[:, 1]] * stress_conv
    s_6_before = np.load(base + '/s_6.npy')[:, list_1[:, 0], list_1[:, 1] - 1] * stress_conv

    # --- Build directed graph G from connections, filtering out sentinel edges ---
    G = nx.DiGraph()
    G.add_nodes_from(unique_values)

    i = 0
    for pair in Connections:
        # Sentinel / validity checks
        if int(V[i] + E_offset) != 666 and V[i] - E_1[i] > 0 and V[i] - E_2[i] > 0:
            G.add_edge(
                pair[0], pair[1],
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
                g_yz_b=s_6_before[sample_index, i]
            )
        i += 1

    # Extract largest weakly connected subgraph, then largest weakly connected component
    S = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
    J = nx.subgraph(S[0], list(S[0].nodes)[0:N_subgraph])
    Jcc = sorted(nx.weakly_connected_components(J), key=len, reverse=True)
    J = J.subgraph(Jcc[0])
    N_subgraph = len(list(J.nodes))
    print(N_subgraph)

    # Edge-attribute dictionaries
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

    # Build per-node energies, barrier matrix, and TLS parameter lists for this subgraph
    Energy_sub = np.ones(N_subgraph)
    Vij = np.zeros((N_subgraph, N_subgraph))
    V_TLS = []
    D_TLS = []

    Gamma_vec = np.ones(N_subgraph)
    Gamma_tens = np.zeros((6, N_subgraph))

    for i in range(0, N_subgraph):
        Node_1 = Nodes[i]
        for j in range(0, N_subgraph):
            Node_2 = Nodes[j]

            # Handle forward edge and reverse edge cases
            if (Node_1, Node_2) in V_dict:
                V_edge = V_dict.get((Node_1, Node_2))
                E1_temp = E1_dict.get((Node_1, Node_2))
                E2_temp = E2_dict.get((Node_1, Node_2))

                Vij[i, j] = V_edge
                Vij[j, i] = V_edge
                Energy_sub[j] = 1.0 * E2_temp

                V_TLS.append(V_edge - 0.5 * (E2_temp + E1_temp))
                D_TLS.append(E2_temp - E1_temp)

                # Stress differences (after - before)
                g_xx_temp = g_xx_dict.get((Node_1, Node_2)) - g_xx_b_dict.get((Node_1, Node_2))
                g_yy_temp = g_yy_dict.get((Node_1, Node_2)) - g_yy_b_dict.get((Node_1, Node_2))
                g_zz_temp = g_zz_dict.get((Node_1, Node_2)) - g_zz_b_dict.get((Node_1, Node_2))
                g_xy_temp = g_xy_dict.get((Node_1, Node_2)) - g_xy_b_dict.get((Node_1, Node_2))
                g_xz_temp = g_xz_dict.get((Node_1, Node_2)) - g_xz_b_dict.get((Node_1, Node_2))
                g_yz_temp = g_yz_dict.get((Node_1, Node_2)) - g_yz_b_dict.get((Node_1, Node_2))

                # These Gamma_* tensors are used downstream
                Gamma_vec[j] = 1.0 * g_xx_temp
                Gamma_vec[i] = -1.0 * g_xx_temp

                Gamma_tens[0, j] = 1.0 * g_xx_temp
                Gamma_tens[0, i] = -1.0 * g_xx_temp
                Gamma_tens[1, j] = 1.0 * g_yy_temp
                Gamma_tens[1, i] = -1.0 * g_yy_temp
                Gamma_tens[2, j] = 1.0 * g_zz_temp
                Gamma_tens[2, i] = -1.0 * g_zz_temp
                Gamma_tens[3, j] = 1.0 * g_xy_temp
                Gamma_tens[3, i] = -1.0 * g_xy_temp
                Gamma_tens[4, j] = 1.0 * g_xz_temp
                Gamma_tens[4, i] = -1.0 * g_xz_temp
                Gamma_tens[5, j] = 1.0 * g_yz_temp
                Gamma_tens[5, i] = -1.0 * g_yz_temp

            elif (Node_2, Node_1) in V_dict:
                V_edge = V_dict.get((Node_2, Node_1))
                E1_temp = E1_dict.get((Node_2, Node_1))
                E2_temp = E2_dict.get((Node_2, Node_1))

                Vij[i, j] = V_edge
                Vij[j, i] = V_edge
                Energy_sub[j] = 1.0 * E1_temp

                V_TLS.append(V_edge - 0.5 * (E2_temp + E1_temp))
                D_TLS.append(E2_temp - E1_temp)

                g_xx_temp = g_xx_dict.get((Node_2, Node_1)) - g_xx_b_dict.get((Node_2, Node_1))
                g_yy_temp = g_yy_dict.get((Node_2, Node_1)) - g_yy_b_dict.get((Node_2, Node_1))
                g_zz_temp = g_zz_dict.get((Node_2, Node_1)) - g_zz_b_dict.get((Node_2, Node_1))
                g_xy_temp = g_xy_dict.get((Node_2, Node_1)) - g_xy_b_dict.get((Node_2, Node_1))
                g_xz_temp = g_xz_dict.get((Node_2, Node_1)) - g_xz_b_dict.get((Node_2, Node_1))
                g_yz_temp = g_yz_dict.get((Node_2, Node_1)) - g_yz_b_dict.get((Node_2, Node_1))

                Gamma_tens[0, j] = 1.0 * g_xx_temp
                Gamma_tens[0, i] = -1.0 * g_xx_temp
                Gamma_tens[1, j] = 1.0 * g_yy_temp
                Gamma_tens[1, i] = -1.0 * g_yy_temp
                Gamma_tens[2, j] = 1.0 * g_zz_temp
                Gamma_tens[2, i] = -1.0 * g_zz_temp
                Gamma_tens[3, j] = 1.0 * g_xy_temp
                Gamma_tens[3, i] = -1.0 * g_xy_temp
                Gamma_tens[4, j] = 1.0 * g_xz_temp
                Gamma_tens[4, i] = -1.0 * g_xz_temp
                Gamma_tens[5, j] = 1.0 * g_yz_temp
                Gamma_tens[5, i] = -1.0 * g_yz_temp

    # Construct longitudinal/transverse coupling matrices
    Gamma_tens_ij = np.einsum('ij,kl->ikjl', Gamma_tens, Gamma_tens)

    Gamma_L_ij = (1.0 / 5.0) * (Gamma_tens_ij[0, 0, :, :] + Gamma_tens_ij[1, 1, :, :] + Gamma_tens_ij[2, 2, :, :]) \
               + (2.0 / 15.0) * (Gamma_tens_ij[0, 1, :, :] + Gamma_tens_ij[0, 2, :, :] + Gamma_tens_ij[1, 2, :, :]) \
               + (4.0 / 15.0) * (Gamma_tens_ij[3, 3, :, :] + Gamma_tens_ij[4, 4, :, :] + Gamma_tens_ij[5, 5, :, :])

    Gamma_T_ij = (1.0 / 15.0) * (Gamma_tens_ij[0, 0, :, :] + Gamma_tens_ij[1, 1, :, :] + Gamma_tens_ij[2, 2, :, :]) \
               - (1.0 / 15.0) * (Gamma_tens_ij[0, 1, :, :] + Gamma_tens_ij[0, 2, :, :] + Gamma_tens_ij[1, 2, :, :]) \
               + (3.0 / 15.0) * (Gamma_tens_ij[3, 3, :, :] + Gamma_tens_ij[4, 4, :, :] + Gamma_tens_ij[5, 5, :, :])

    # -------------------------------
    # Compute and store distributions for plotting
    # -------------------------------

    # TLS distributions pulled from graph edges
    V_TLS_edges = list(nx.get_edge_attributes(J, "energy_barrier_TLS").values())
    D_TLS_edges = list(nx.get_edge_attributes(J, "energy_asymmetry").values())

    V_combined.append(V_TLS_edges)
    D_combined.append(D_TLS_edges)
    E_combined.append(Energy_sub)

    # --- Basis cycles in the undirected version of J ---
    cycles = list(nx.cycle_basis(J.to_undirected()))
    nodes_in_cycles = np.zeros(len(cycles))
    for ii in range(len(cycles)):
        nodes_in_cycles[ii] = len(cycles[ii])

    #plt.figure()
    bins = [x - 0.5 for x in range(3, 20 + 2)]
    hist_cycles_1, bins_cycles_1, _ = plt.hist(
        nodes_in_cycles,
        bins=bins,
        align='mid',
        density=False,
        log=False
    )
    hist_cycles_1 = hist_cycles_1 / N_subgraph
    cycles_combined.append(hist_cycles_1)

    plt.xlabel('Number of nodes in a cycle')
    plt.ylabel('Frequency')
    plt.title('$T_{\\rm search} = 600K$')
    plt.tight_layout()

    # --- Degree distribution for a-Si graph ---
    #plt.figure()
    degree_sequence = [d for n, d in (G.to_undirected()).degree()]
    hist_degree_1, bins_degree_1, _ = plt.hist(
        degree_sequence,
        bins=range(1, 100 + 1),
        density=True,
        alpha=0.75,
        log=True
    )
    degree_combined.append(hist_degree_1)

    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')

# ===============================
# Load TiO2 reference distributions
# ===============================
asym = []
barrier = []
energy = []
degree = []
cycle = []

for i in range(1, 11):
    asym.append(np.load('./a_TiO2/TiO2_data_out/asym_dist_TiO2_sample_' + str(i) + '.npy')[1, :])
    barrier.append(np.load('./a_TiO2/TiO2_data_out/barrier_dist_TiO2_sample_' + str(i) + '.npy')[1, :])
    energy.append(np.load('./a_TiO2/TiO2_data_out/Energy_dist_TiO2_sample_' + str(i) + '.npy')[1, :])
    degree.append(np.load('./a_TiO2/TiO2_data_out/degree_dist_TiO2_sample_' + str(i) + '.npy')[1, :])

    if i == 3:
        cycle.append(np.array([0]))
    else:
        cycle.append(np.load('./a_TiO2/TiO2_data_out/cycle_dist_TiO2_sample_' + str(i) + '.npy')[1, :])

# Pad degree arrays to common length
size_temp = 0
for i in range(0, 10):
    if degree[i].shape[0] > size_temp:
        size_temp = degree[i].shape[0]
for i in range(0, 10):
    degree[i] = np.pad(degree[i], (0, size_temp - degree[i].shape[0]))

# Pad cycle arrays to common length
size_temp = 0
for i in range(0, 10):
    if cycle[i].shape[0] > size_temp:
        size_temp = cycle[i].shape[0]
for i in range(0, 10):
    cycle[i] = np.pad(cycle[i], (0, size_temp - cycle[i].shape[0]))

asym = np.array(asym)
barrier = np.array(barrier)
energy = np.array(energy)
degree = np.array(degree)
cycle = np.array(cycle)

for i in range(0, 10):
    cycle[i] = cycle[i] / np.sum(energy[i, :])
cycle_bins = np.linspace(4, 3 + cycle[0].shape[0], cycle[0].shape[0])

# ===============================
# Figure 1: Degree + cycle distributions (a-Si vs TiO2)
# Saves: degree_cycles_combined_binned.pdf
# ===============================
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

# --- Degree: a-Si (binned mean + SEM) ---
x = bins_degree_1[1::] - 1
y_mean = np.mean(np.array(degree_combined), axis=0)

bins = np.logspace(np.log10(1), np.log10(100), num=20)
bin_means, bin_edges, binnumber = binned_statistic(
    bins_degree_1[1::] - 1,
    np.mean(np.array(degree_combined), axis=0),
    statistic='mean',
    bins=bins
)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
y_sem = stats.sem(np.array(degree_combined), axis=0)

# Combine SEMs within each bin
bin_indices = np.digitize(x, bins)
binned_sem = np.empty(len(bins) - 1)
binned_sem[:] = np.nan
for i in range(1, len(bins)):
    mask = bin_indices == i
    n = np.sum(mask)
    if n > 0:
        binned_sem[i - 1] = np.sqrt(np.sum(y_sem[mask] ** 2)) / n
    else:
        binned_sem[i - 1] = np.nan

# Horizontal error bars from bin widths
xerr_lower = bin_centers - bin_edges[:-1]
xerr_upper = bin_edges[1:] - bin_centers
xerr = [xerr_lower, xerr_upper]

axs[0].loglog(bin_centers, bin_means, 'ko')
axs[0].errorbar(bin_centers, bin_means, yerr=binned_sem, xerr=xerr, fmt='ko')

# Power-law fit line
x_fit = np.log(bins_degree_1[2::] - 1)
y_fit = np.log(np.mean(np.array(degree_combined), axis=0))[1::]
y_err = y_sem[1::] / np.mean(np.array(degree_combined), axis=0)[1::]

mask = np.isfinite(x_fit) & np.isfinite(y_fit) & np.isfinite(y_err)
x_clean = x_fit[mask]
y_clean = y_fit[mask]
y_err_clean = y_err[mask]

x_base = np.log(bins_degree_1[1::] - 1)
coeffs, cov = np.polyfit(x_clean, y_clean, 1, w=1 / y_err_clean, cov=True)

axs[0].loglog(np.exp(x_base), np.exp(x_base * coeffs[0] + coeffs[1]), linestyle='--', color='grey')

# --- Degree: TiO2 (binned mean + SEM) ---
x = np.linspace(2, 1 + degree.shape[1], degree.shape[1])
y_mean = np.mean(degree, axis=0)

bins = np.logspace(np.log10(1), np.log10(100), num=20)
bin_means, bin_edges, binnumber = binned_statistic(x, y_mean, statistic='mean', bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
y_sem = stats.sem(degree, axis=0)

bin_indices = np.digitize(x, bins)
binned_sem = np.empty(len(bins) - 1)
binned_sem[:] = np.nan
for i in range(1, len(bins)):
    mask = bin_indices == i
    n = np.sum(mask)
    if n > 0:
        binned_sem[i - 1] = np.sqrt(np.sum(y_sem[mask] ** 2)) / n
    else:
        binned_sem[i - 1] = np.nan

xerr_lower = bin_centers - bin_edges[:-1]
xerr_upper = bin_edges[1:] - bin_centers
xerr = [xerr_lower, xerr_upper]
axs[0].errorbar(bin_centers, bin_means, yerr=binned_sem, xerr=xerr, fmt='rv')

# --- Cycles: a-Si and TiO2 ---
axs[1].plot([0, 100], [0, 0], 'k:', alpha=0.5)

axs[1].plot(bins_cycles_1[1::] - 0.5, np.mean(np.array(cycles_combined), axis=0), 'ko')
sem = stats.sem(np.array(cycles_combined), axis=0)
axs[1].errorbar(
    bins_cycles_1[1::] - 0.5,
    np.mean(np.array(cycles_combined), axis=0),
    yerr=sem,
    fmt='ko',
    alpha=0.5
)

axs[1].plot(cycle_bins - 1, np.mean(cycle, axis=0), 'rv')
sem = stats.sem(cycle, axis=0)
axs[1].errorbar(cycle_bins - 1, np.mean(cycle, axis=0), yerr=sem, fmt='rv', alpha=0.5)

# Axes formatting
axs[0].set_xlabel('Degree', fontsize=14)
axs[0].set_ylabel('Density', fontsize=14)
axs[0].set_ylim(1e-6, 1)
axs[0].set_xlim(1, 100)
axs[0].set_xticks([1, 10, 100], ['$10^{0}$', '$10^{1}$', '$10^{2}$'], fontsize=12)
axs[0].set_yticks([1e-6, 1e-4, 1e-2, 1], ['$10^{-6}$', '$10^{-4}$', '$10^{-2}$', '$10^{0}$'], fontsize=12)

axs[1].set_xlabel('Basis Cycles', fontsize=14)
axs[1].set_ylabel('$N_{\\rm cycle}/N_{\\rm Network}\\times 100$\\%', fontsize=14)
axs[1].set_xlim(2.5, 10.5)
axs[1].set_ylim(-0.01, 0.15)
axs[1].set_xticks([4, 6, 8, 10], ['$4$', '$6$', '$8$', '$10$'], fontsize=12)
axs[1].set_yticks([0, 0.05, 0.1, 0.15], [0, 5, 10, 15], fontsize=12)

fig.tight_layout()
fig.savefig('degree_cycles_combined_binned.pdf')

# ===============================
# Figure 2: Energy/barrier/asymmetry distributions (a-Si aggregated vs TiO2 aggregated)
# Saves: energy_distribution_combined.pdf
# ===============================

# Shift each a-Si energy list by its own minimum
for i in range(0, 10):
    E_combined[i] += -E_combined[i].min()

# a-Si aggregated histograms (flatten across samples)
flattened = [item for sublist in E_combined for item in sublist]
plt.figure()
bins = np.linspace(0, 1, 51)
hist_energy_1, bins_energy_1, _ = plt.hist(flattened, bins=bins, align='mid', density=True, log=False)

flattened = [item for sublist in V_combined for item in sublist]
plt.figure()
bins = np.linspace(0, 1, 51)
hist_barrier_1, bins_barrier_1, _ = plt.hist(flattened, bins=bins, align='mid', density=True, log=False)

flattened = [item for sublist in D_combined for item in sublist]
plt.figure()
bins = np.linspace(0, 1, 51)
hist_asym_1, bins_asym_1, _ = plt.hist(flattened, bins=bins, align='mid', density=True, log=False)

# Final 2-panel figure
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

axs[0].bar(bins[1::] - 0.01, hist_energy_1, width=0.02, align='center', color='k', alpha=0.25)
axs[0].bar(bins[1::] - 0.01, hist_barrier_1, width=0.02, align='center', color='b', alpha=0.25)
axs[0].bar(bins[1::] - 0.01, hist_asym_1, width=0.02, align='center', color='r', alpha=0.25)

axs[0].plot(bins[1::] - 0.01, hist_energy_1, 'k', linewidth=2.0)
axs[0].plot(bins[1::] - 0.01, hist_barrier_1, 'b--', linewidth=2.0)
axs[0].plot(bins[1::] - 0.01, hist_asym_1, 'r:', linewidth=4.0)

axs[1].bar(bins[1::] - 0.01, np.sum(energy, axis=0) / np.sum(np.sum(energy, axis=0)) / 0.02,
           width=0.02, align='center', color='k', alpha=0.25)
axs[1].bar(bins[1::] - 0.01, np.sum(barrier, axis=0) / np.sum(np.sum(barrier, axis=0)) / 0.02,
           width=0.02, align='center', color='b', alpha=0.25)
axs[1].bar(bins[1::] - 0.01, np.sum(asym, axis=0) / np.sum(np.sum(asym, axis=0)) / 0.02,
           width=0.02, align='center', color='r', alpha=0.25)

axs[1].plot(bins[1::] - 0.01, np.sum(energy, axis=0) / np.sum(np.sum(energy, axis=0)) / 0.02, 'k', linewidth=2.0)
axs[1].plot(bins[1::] - 0.01, np.sum(barrier, axis=0) / np.sum(np.sum(barrier, axis=0)) / 0.02, 'b--', linewidth=2.0)
axs[1].plot(bins[1::] - 0.01, np.sum(asym, axis=0) / np.sum(np.sum(asym, axis=0)) / 0.02, 'r:', linewidth=4.0)

axs[0].set_xlabel('Energy (eV)', fontsize=14)
axs[0].set_ylabel('Density', fontsize=14)
axs[0].set_ylim(0, 15)
axs[0].set_xlim(0, 1)
axs[0].set_xticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], fontsize=12)
axs[0].set_yticks([0, 5, 10, 15], [0, 5, 10, 15], fontsize=12)

axs[1].set_xlabel('Energy (eV)', fontsize=14)
axs[1].set_ylabel('Density', fontsize=14)
axs[1].set_ylim(0, 15)
axs[1].set_xlim(0, 1)
axs[1].set_xticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], fontsize=12)
axs[1].set_yticks([0, 5, 10, 15], [0, 5, 10, 15], fontsize=12)

fig.tight_layout()
fig.savefig('energy_distribution_combined.pdf')

# plt.show()
