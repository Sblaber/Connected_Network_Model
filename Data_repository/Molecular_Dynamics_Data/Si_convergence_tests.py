"""
Convergence / subsampling analysis for an a-Si inherent-structure transition network.

This script:
  1) Loads barrier/asymmetry/energy data and a directed transition graph for each sample.
  2) Extracts the largest weakly-connected component.
  3) Subsamples progressively larger portions of the node list and measures:
       - percent connected (edge density) in the induced subgraph,
       - degree distribution histogram,
       - counts of basis cycles of various lengths (4, 6, 8, 10) normalized by subgraph size.
  4) Aggregates results across samples and produces a 2×3 panel figure saved as pdf.

Output:
  - Si_convergence_test_2.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import networkx as nx

rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# ===============================
# Physical / model parameters
# ===============================
Gamma = 0.01

k_B = 8.617e-5  # eV/K
T = 300  # K
beta = (k_B * T) ** (-1.0)

# ===============================
# Analysis / sampling parameters
# ===============================
N_graph_max = 3000
N_subsample = 50

base = './a_Si/Data'

# Storage for subsampling results
Connectivity = np.zeros((10, N_subsample))
degree_subsample = np.zeros((10, N_subsample, 17))
Network_fraction = np.zeros((10, N_subsample))
N_network_subsample = np.zeros((10, N_subsample))

N_3_cycle = np.zeros((10, N_subsample))
N_4_cycle = np.zeros((10, N_subsample))
N_5_cycle = np.zeros((10, N_subsample))
N_6_cycle = np.zeros((10, N_subsample))
N_7_cycle = np.zeros((10, N_subsample))
N_8_cycle = np.zeros((10, N_subsample))
N_10_cycle = np.zeros((10, N_subsample))

# Binning for cycle sizes and degree histograms
bins_cycles = np.linspace(1.5, 10.5, 10)
bins_degree = np.logspace(0, 2, 18)
N_subgraph_value = np.logspace(0, 4, N_subsample + 1)

# ===============================
# Main loop over samples
# ===============================
for sample_index in range(0, 10):
    N_sample = 1 + sample_index

    # NOTE: This script intentionally loads "sample_10" arrays and indexes with sample_index-1,
    # matching the provided code exactly.
    V = np.load(base + '/V_monotonicP_wofirstlast_sample_' + str(10) + '.npy')[sample_index - 1, :, :]
    V = V[~np.isnan(V)]
    N_graph_max = len(V)

    Connections = np.load(base + '/Connections' + str(10) + '.npy')[0:N_graph_max]

    D = np.load(base + '/D_asymm_sanity_monotonicP_wofirstlast_sample_' + str(10) + '.npy')[sample_index - 1, :, :]
    D = D[~np.isnan(D)][0:N_graph_max]

    E_1 = np.load(base + '/E_initial_monotonicP_wofirstlast_sample_' + str(10) + '.npy')[sample_index - 1, :, :]
    E_2 = np.load(base + '/E_final_monotonicP_wofirstlast_sample_' + str(10) + '.npy')[sample_index - 1, :, :]

    E_1 = E_1[~np.isnan(E_1)][0:N_graph_max]
    E_offset = E_1.min()

    # Shift energies to have minimum zero
    E_1 = E_1[~np.isnan(E_1)] - E_offset
    E_2 = E_2[~np.isnan(E_2)][0:N_graph_max] - E_offset

    V = V - E_offset

    # Build graph nodes from unique values in Connections excluding sentinel "666" values
    E_array = np.transpose(np.vstack((E_1, E_2)))
    C_flat = Connections.flatten()
    E_flat = E_array.flatten()
    unique_values, unique_indices = np.unique(
        C_flat[(E_flat + E_offset) != 666], return_index=True
    )

    # --- Build directed graph of transitions (edge attributes kept minimal here) ---
    G_full = nx.DiGraph()
    G_full.add_nodes_from(unique_values)

    for i, pair in enumerate(Connections):
        if int(V[i] + E_offset) != 666 and (V[i] - E_1[i] > 0) and (V[i] - E_2[i] > 0):
            G_full.add_edge(
                pair[0], pair[1],
                energy_barrier=V[i],
                energy_asymmetry=D[i],
                E_1=E_1[i],
                E_2=E_2[i],
                energy_barrier_TLS=V[i] - 0.5 * (E_1[i] + E_2[i]),
            )

    # --- Restrict to largest weakly-connected component ---
    S = [G_full.subgraph(c).copy() for c in nx.weakly_connected_components(G_full)]
    J = nx.subgraph(S[0], list(S[0].nodes)[0:N_graph_max])
    Jcc = sorted(nx.weakly_connected_components(J), key=len, reverse=True)
    G = J.subgraph(Jcc[0])
    N_nodes_total = len(list(G.nodes()))
    print(N_nodes_total)

    # Precompute cycle basis on the full graph (used only to check if any cycles exist)
    cycles_full = list(nx.cycle_basis(G.to_undirected()))

    # ===============================
    # Subsampling loop (increasing induced subgraph sizes)
    # ===============================
    for i in range(1, N_subsample + 1):
        N_nodes_subgraph = np.min([N_nodes_total, int(N_subgraph_value[i])])
        if N_nodes_subgraph > 1:
            G_subgraph = G.subgraph(list(G.nodes())[0:N_nodes_subgraph])

            # Edge density (%) for the undirected induced subgraph
            Connectivity[sample_index - 1, i - 1] = (
                100 * len(list(G_subgraph.to_undirected().edges()))
                / (len(list(G_subgraph.to_undirected().nodes())) ** 2.0)
            )

            # Fraction of nodes included in the induced subgraph
            Network_fraction[sample_index - 1, i - 1] = (
                len(list(G_subgraph.to_undirected().nodes())) / N_nodes_total
            )

            # Store absolute node count as well
            N_network_subsample[sample_index - 1, i - 1] = len(list(G_subgraph.to_undirected().nodes()))

            # Degree histogram for the induced subgraph (log-binned)
            degree_sequence = [d for _, d in G_subgraph.to_undirected().degree()]
            hist_degree_1, _, _ = plt.hist(
                degree_sequence, bins=bins_degree, density=True, alpha=0.75, log=True
            )
            degree_subsample[sample_index - 1, i - 1, :] = hist_degree_1

            # Cycle counting on the induced subgraph (basis cycles)
            if len(cycles_full) > 0:
                cycles_subgraph = list(nx.cycle_basis(G_subgraph.to_undirected()))
                if len(cycles_subgraph) > 0:
                    nodes_in_cycles_subgraph = np.zeros(len(cycles_subgraph))
                    for j in range(len(cycles_subgraph)):
                        nodes_in_cycles_subgraph[j] = len(cycles_subgraph[j])

                    plt.figure(1)
                    hist_cycles_1, _, _ = plt.hist(
                        nodes_in_cycles_subgraph, bins=bins_cycles, align='mid',
                        density=False, log=False
                    )

                    # Store counts in specific bins
                    N_3_cycle[sample_index - 1, i - 1] = hist_cycles_1[1]
                    N_4_cycle[sample_index - 1, i - 1] = hist_cycles_1[2]
                    N_5_cycle[sample_index - 1, i - 1] = hist_cycles_1[3]
                    N_6_cycle[sample_index - 1, i - 1] = hist_cycles_1[4]
                    N_7_cycle[sample_index - 1, i - 1] = hist_cycles_1[5]
                    N_8_cycle[sample_index - 1, i - 1] = hist_cycles_1[6]
                    N_10_cycle[sample_index - 1, i - 1] = hist_cycles_1[8]

                    plt.xlabel('Number of nodes in a cycle')
                    plt.ylabel('Frequency')
                    plt.title('$T_{\\rm search} = 600K$')
                    plt.tight_layout()

    # Degree distribution on the full graph (not used in final figure; kept because it is executed)
    plt.figure(2)
    degree_sequence = [d for _, d in G.to_undirected().degree()]
    plt.hist(
        degree_sequence,
        bins=range(min(degree_sequence), max(degree_sequence) + 1),
        density=True,
        alpha=0.75,
        log=True,
    )
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')

# ===============================
# Final figure: 2×3 panel plot
# ===============================
fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(12, 6),
    constrained_layout=True
)
lw = 2.0

# (0,0) Percent connected vs fraction of total network
for i in range(0, 10):
    axes[0, 0].plot(Network_fraction[i, :], Connectivity[i, :], 'k', linewidth=lw, alpha=0.3)
axes[0, 0].plot(np.mean(Network_fraction, axis=0), np.mean(Connectivity, axis=0), 'k--', linewidth=lw)

# (1,0) Degree histograms at selected subsample sizes
for i in range(5, int(N_subsample / 5)):
    axes[1, 0].loglog(
        bins_degree[1::],
        np.mean(degree_subsample, axis=0)[5 * i, :],
        'o',
        label=str(np.round(np.mean(Network_fraction, axis=0)[5 * i], 2)),
    )
axes[1, 0].legend(frameon=False)

# (0,1) 4-cycles per node (%)
axes[0, 1].plot(
    np.mean(Network_fraction, axis=0),
    (N_4_cycle / N_network_subsample).transpose() * 100,
    'k',
    linewidth=lw,
    alpha=0.3,
)
axes[0, 1].plot(
    np.mean(Network_fraction, axis=0),
    np.mean(N_4_cycle / N_network_subsample * 100, axis=0),
    'k--',
    linewidth=lw,
)

# (0,2) 6-cycles per node (%)
axes[0, 2].plot(
    np.mean(Network_fraction, axis=0),
    (N_6_cycle / N_network_subsample).transpose() * 100,
    'r',
    linewidth=lw,
    alpha=0.3,
)
axes[0, 2].plot(
    np.mean(Network_fraction, axis=0),
    np.mean(N_6_cycle / N_network_subsample, axis=0) * 100,
    'r--',
    linewidth=lw,
)

# (1,1) 8-cycles per node (%)
axes[1, 1].plot(
    np.mean(Network_fraction, axis=0),
    (N_8_cycle / N_network_subsample).transpose() * 100,
    'g',
    linewidth=lw,
    alpha=0.3,
)
axes[1, 1].plot(
    np.mean(Network_fraction, axis=0),
    np.mean(N_8_cycle / N_network_subsample, axis=0) * 100,
    'g--',
    linewidth=lw,
)

# (1,2) 10-cycles per node (%)
axes[1, 2].plot(
    np.mean(Network_fraction, axis=0),
    (N_10_cycle / N_network_subsample).transpose() * 100,
    'b',
    linewidth=lw,
    alpha=0.3,
)
axes[1, 2].plot(
    np.mean(Network_fraction, axis=0),
    np.mean(N_10_cycle / N_network_subsample, axis=0) * 100,
    'b--',
    linewidth=lw,
)

# --- Labels ---
axes[0, 0].set_ylabel('Percent Connected')
axes[1, 0].set_ylabel('Density')
axes[0, 1].set_ylabel('$N_{\\rm cycle}/N_{\\rm Network}\\times100$\\%')
axes[0, 2].set_ylabel('$N_{\\rm cycle}/N_{\\rm Network}\\times100$\\%')
axes[1, 1].set_ylabel('$N_{\\rm cycle}/N_{\\rm Network}\\times100$\\%')
axes[1, 2].set_ylabel('$N_{\\rm cycle}/N_{\\rm Network}\\times100$\\%')

axes[0, 0].set_xlabel('Fraction of Total Network')
axes[0, 1].set_xlabel('Fraction of Total Network')
axes[0, 2].set_xlabel('Fraction of Total Network')
axes[1, 1].set_xlabel('Fraction of Total Network')
axes[1, 2].set_xlabel('Fraction of Total Network')
axes[1, 0].set_xlabel('Degree')

# --- Limits ---
axes[0, 0].set_xlim(0, 1)
axes[1, 0].set_xlim(0, 100)
axes[0, 1].set_xlim(0, 1)
axes[0, 2].set_xlim(0, 1)
axes[1, 1].set_xlim(0, 1)
axes[1, 2].set_xlim(0, 1)

axes[0, 0].set_ylim(0, 1)
axes[1, 0].set_ylim(10**(-6.0), 1)
axes[0, 1].set_ylim(0, 15)
axes[0, 2].set_ylim(0, 15)
axes[1, 1].set_ylim(0, 15)
axes[1, 2].set_ylim(0, 15)

# --- Uniform tick/label sizing ---
axes_flat = axes.flatten()
for ax in axes_flat:
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

plt.tight_layout()
plt.savefig('Si_convergence_test_2.pdf')
plt.show()
