"""
Convergence / subsampling analysis for an a-TiO2 inherent-structure transition network.

This script:
  1) Loads barrier/asymmetry/energy data and a directed transition graph for each sample.
  2) Extracts the largest weakly-connected component.
  3) Subsamples progressively larger portions of the node list and measures:
       - percent connected (edge density) in the induced subgraph,
       - degree distribution histogram,
       - counts of basis cycles of various lengths (4, 6, 8, 10) normalized by subgraph size.
  4) Aggregates results across samples and produces a 2×3 panel figure saved as pdf.

Output:
  - TiO2_convergence_test_updated.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import networkx as nx

rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)

# =========================
# Convergence test settings
# =========================
N_subsample = 50  # number of subgraph sizes to test per network
N_samples = 10    # sample_index runs 1..10 in the original script

Connectivity = np.zeros((N_samples, N_subsample))
degree_subsample = np.zeros((N_samples, N_subsample, 17))
Network_fraction = np.zeros((N_samples, N_subsample))
N_network_subsample = np.zeros((N_samples, N_subsample))

N_3_cycle = np.zeros((N_samples, N_subsample))
N_4_cycle = np.zeros((N_samples, N_subsample))
N_5_cycle = np.zeros((N_samples, N_subsample))
N_6_cycle = np.zeros((N_samples, N_subsample))
N_7_cycle = np.zeros((N_samples, N_subsample))
N_8_cycle = np.zeros((N_samples, N_subsample))

bins_cycles = np.linspace(1.5, 10.5, 10)
bins_degree = np.logspace(0, 2, 18)
N_subgraph_value = np.logspace(0, 4, N_subsample + 1)

# =========================
# Load + build networks
# =========================
for sample_index in range(1, 11):
    # Load arrays
    V = np.load(f"./a_TiO2/TiO2_data/V_monotonicP_wofirstlast_sample_{sample_index}_traj_200.npy")[sample_index - 1, :, :]
    V = V[~np.isnan(V)]
    Connections = np.load(f"./a_TiO2/TiO2_data/Connections_Sample_{sample_index}_200.npy")

    N = min(V.shape[0], Connections.shape[0])
    Connections = Connections[0:N, :]
    V = V[0:N]

    D = np.load(f"./a_TiO2/TiO2_data/D_asymm_sanity_monotonicP_wofirstlast_sample_{sample_index}_traj_200.npy")[sample_index - 1, :, :]
    D = D[~np.isnan(D)]

    E_1 = np.load(f"./a_TiO2/TiO2_data/E_initial_monotonicP_wofirstlast_sample_{sample_index}_traj_200.npy")[sample_index - 1, :, :]
    E_2 = np.load(f"./a_TiO2/TiO2_data/E_final_monotonicP_wofirstlast_sample_{sample_index}_traj_200.npy")[sample_index - 1, :, :]
    E_1 = E_1[~np.isnan(E_1)]
    E_offset = E_1.min()

    E_1 = E_1[0:N] - E_offset
    E_2 = E_2[~np.isnan(E_2)][0:N] - E_offset
    V = V - E_offset

    # Build node set (as in original)
    E_array = np.transpose(np.vstack((E_2, E_1)))
    C_flat = Connections.flatten()
    E_flat = E_array.flatten()
    unique_values, unique_indices = np.unique(C_flat[(E_flat + E_offset) != 666], return_index=True)

    # Build directed graph
    G = nx.DiGraph()
    G.add_nodes_from(unique_values)

    for i, pair in enumerate(Connections):
        if (
            int(V[i] + E_offset) != 666
            and (pair[0] - pair[1] != 0)
            and (V[i] < 10)
            and (V[i] > E_1[i])
            and (V[i] > E_2[i])
        ):
            # keep original stdout
            print(pair[0])
            print(pair[1])

            G.add_edge(
                pair[0],
                pair[1],
                energy_barrier=V[i],
                energy_asymmetry=D[i],
                E_1=E_1[i],
                E_2=E_2[i],
                index=i,
            )

    # Extract largest connected component (undirected)
    components = [G.subgraph(c).copy() for c in nx.connected_components(G.to_undirected())]
    max_graph_index = np.argmax([len(c.nodes()) for c in components])
    G = components[max_graph_index].copy()

    # Cycles on full graph (basis cycles)
    cycles_full = list(nx.cycle_basis(G.to_undirected()))

    # =========================
    # Subsample convergence
    # =========================
    for i in range(1, N_subsample + 1):
        N_nodes = len(G.nodes())
        N_nodes_subgraph = min(N_nodes, int(N_subgraph_value[i]))
        if N_nodes_subgraph <= 1:
            continue

        G_sub = G.subgraph(list(G.nodes())[0:N_nodes_subgraph])
        Gu = G_sub.to_undirected()

        # Connectivity (edge density)
        Connectivity[sample_index - 1, i - 1] = 100.0 * len(Gu.edges()) / (len(Gu.nodes()) ** 2.0)
        Network_fraction[sample_index - 1, i - 1] = len(Gu.nodes()) / N_nodes
        N_network_subsample[sample_index - 1, i - 1] = len(Gu.nodes())

        # Degree histogram
        degree_sequence = [d for _, d in Gu.degree()]
        hist_degree, _, _ = plt.hist(degree_sequence, bins=bins_degree, density=True, alpha=0.75, log=True)
        degree_subsample[sample_index - 1, i - 1, :] = hist_degree

        # Cycle histogram for this induced subgraph
        if len(cycles_full) > 0:
            cycles_sub = list(nx.cycle_basis(Gu))
            if len(cycles_sub) > 0:
                nodes_in_cycles = np.array([len(c) for c in cycles_sub], dtype=float)
                hist_cycles, _, _ = plt.hist(nodes_in_cycles, bins=bins_cycles, align="mid", density=False, log=False)

                N_3_cycle[sample_index - 1, i - 1] = hist_cycles[1]
                N_4_cycle[sample_index - 1, i - 1] = hist_cycles[2]
                N_5_cycle[sample_index - 1, i - 1] = hist_cycles[3]
                N_6_cycle[sample_index - 1, i - 1] = hist_cycles[4]
                N_7_cycle[sample_index - 1, i - 1] = hist_cycles[5]
                N_8_cycle[sample_index - 1, i - 1] = hist_cycles[6]

    # Degree plot for full graph
    plt.figure(2)
    deg_full = [d for _, d in G.to_undirected().degree()]
    plt.hist(deg_full, bins=range(min(deg_full), max(deg_full) + 1), density=True, alpha=0.75, log=True)

# =========================
# Make final convergence plot (2 × 3)
# =========================
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), constrained_layout=True)
lw = 2.0

# Connectivity vs fraction
for i in range(0, N_samples):
    axes[0, 0].plot(Network_fraction[i, :], Connectivity[i, :], "k", linewidth=lw, alpha=0.3)
axes[0, 0].plot(np.mean(Network_fraction, axis=0), np.mean(Connectivity, axis=0), "k--", linewidth=lw)

# Degree distributions at selected fractions
for i in range(4, int(N_subsample / 5) - 1):
    axes[1, 0].loglog(
        bins_degree[1:],
        np.mean(degree_subsample, axis=0)[5 * i, :],
        "o",
        label=str(np.round(np.mean(Network_fraction, axis=0)[5 * i], 2)),
    )
axes[1, 0].legend(frameon=False)

# Cycle densities (percent)
axes[0, 1].plot(np.mean(Network_fraction, axis=0), (N_3_cycle / N_network_subsample).T * 100, "k", linewidth=lw, alpha=0.3)
axes[0, 1].plot(np.mean(Network_fraction, axis=0), np.mean(N_3_cycle / N_network_subsample * 100, axis=0), "k--", linewidth=lw)

axes[0, 2].plot(np.mean(Network_fraction, axis=0), (N_4_cycle / N_network_subsample).T * 100, "r", linewidth=lw, alpha=0.3)
axes[0, 2].plot(np.mean(Network_fraction, axis=0), np.mean(N_4_cycle / N_network_subsample, axis=0) * 100, "r--", linewidth=lw)

axes[1, 1].plot(np.mean(Network_fraction, axis=0), (N_5_cycle / N_network_subsample).T * 100, "g", linewidth=lw, alpha=0.3)
axes[1, 1].plot(np.mean(Network_fraction, axis=0), np.mean(N_5_cycle / N_network_subsample, axis=0) * 100, "g--", linewidth=lw)

axes[1, 2].plot(np.mean(Network_fraction, axis=0), (N_6_cycle / N_network_subsample).T * 100, "b", linewidth=lw, alpha=0.3)
axes[1, 2].plot(np.mean(Network_fraction, axis=0), np.mean(N_6_cycle / N_network_subsample, axis=0) * 100, "b--", linewidth=lw)

# Labels
axes[0, 0].set_ylabel("Percent Connected")
axes[1, 0].set_ylabel("Density")
axes[0, 1].set_ylabel(r"$N_{\rm cycle}/N_{\rm Network}\times100$\%")
axes[0, 2].set_ylabel(r"$N_{\rm cycle}/N_{\rm Network}\times100$\%")
axes[1, 1].set_ylabel(r"$N_{\rm cycle}/N_{\rm Network}\times100$\%")
axes[1, 2].set_ylabel(r"$N_{\rm cycle}/N_{\rm Network}\times100$\%")

axes[0, 0].set_xlabel("Fraction of Total Network")
axes[0, 1].set_xlabel("Fraction of Total Network")
axes[0, 2].set_xlabel("Fraction of Total Network")
axes[1, 1].set_xlabel("Fraction of Total Network")
axes[1, 2].set_xlabel("Fraction of Total Network")
axes[1, 0].set_xlabel("Degree")

# Limits
axes[0, 0].set_xlim(0, 1)
axes[1, 0].set_xlim(0, 100)
axes[0, 1].set_xlim(0, 1)
axes[0, 2].set_xlim(0, 1)
axes[1, 1].set_xlim(0, 1)
axes[1, 2].set_xlim(0, 1)

axes[0, 0].set_ylim(0, 40)
axes[1, 0].set_ylim(10 ** (-6.0), 1)
axes[0, 1].set_ylim(0, 40)
axes[0, 2].set_ylim(0, 40)
axes[1, 1].set_ylim(0, 40)
axes[1, 2].set_ylim(0, 40)

# Tick + label font sizes
for ax in axes.flatten():
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

plt.tight_layout()
plt.savefig("TiO2_convergence_test_updated.pdf")
plt.show()
