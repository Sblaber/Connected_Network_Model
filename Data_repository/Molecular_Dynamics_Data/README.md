# Network Structure and Mechanical Loss Analysis

This folder contains Python scripts used to analyze network structure, convergence properties, and mechanical loss spectra for amorphous materials, with a focus on **a-Si** and **TiO₂** energy-landscape networks. The scripts generate the figures used in the associated manuscript and rely on precomputed network and transition data stored in subdirectories.

---

## Contents

### 1. `Network_properties.py`

Analyzes **structural properties of energy-landscape networks**, including:

- Degree distributions
- Basis cycle statistics (cycle length distributions)
- Network connectivity as a function of subgraph size
- Comparison between **a-Si** and **TiO2** networks

**Primary outputs:**
- Degree, cycle, energy, statistics
- Network connectivity
- Saved figures (PDF)

---

### 2. `Mechanical_loss.py`

Computes **frequency-dependent mechanical loss** from both:

- Full configuration-network (CN) dynamics
- Independent two-level system (TLS) approximations

For both **a-Si** and **TiO2**, this script:
- Loads precomputed dissipation spectra
- Normalizes by elastic constants and volume
- Compares CN and TLS contributions over many decades in frequency

**Primary outputs:**
- Mechanical loss vs frequency plots
- Direct comparison of CN and TLS dissipation mechanisms

---

### 3. `Si_convergence_tests.py`

Performs **network convergence tests for amorphous silicon (a-Si)** networks.

This script:
- Builds energy-landscape graphs from a-Si transition data
- Computes connectivity, degree distributions, and cycle counts
- Evaluates how network observables converge as a function of retained network fraction

**Primary outputs:**
- Connectivity vs network fraction
- Degree distribution convergence
- Cycle density convergence
- Saved figure: `Si_convergence_test_2.pdf`

---

### 4. `TiO2_convergence_tests.py`

Performs **network convergence tests for TiO₂** energy-landscape networks, analogous to the a-Si analysis.

This script:
- Constructs TiO₂ transition networks
- Extracts the largest connected component
- Computes degree distributions and cycle statistics on induced subgraphs
- Quantifies convergence of network observables with increasing network size

**Primary outputs:**
- Connectivity and cycle convergence plots
- Saved figure: `TiO2_convergence_test_updated.pdf`

---

## Data Dependencies

These scripts rely on precomputed `.npy` and `.txt` files located in:

- `./a_Si/`
- `./a_TiO2/`

These data include:
- Transition energies and barriers
- Network connectivity lists
- Stress-coupling tensors
- Precomputed dissipation spectra

---

## Requirements

The scripts were developed and tested using:

- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib
- NetworkX

LaTeX must be available on the system for figure rendering (`usetex=True`).

---
