# Frequency-Dependent Dissipation in Minimal Energy-Landscape Models

This folder contains Python scripts used to generate the frequency-dependent dissipation curves shown in the associated figures. Both scripts implement closely related calculations of mechanical loss on small (four-state) energy landscapes and save PDF figures.

## Contents

- **`cycle_v_chain.py`**  
  Computes and plots the dissipation spectrum for a four-state cyclic network, comparing two topologies (chain and cycle) that differ in the barrier \(V_{14}\).  
  Output:  
  - `Q_cycle_v_chain.pdf`

- **`Mountain.py`**  
  Computes and plots the dissipation spectrum for a four-state “mountain” landscape, comparing three values of the energy splitting \(\Delta E\).  
  Output:  
  - `Q_mountain.pdf`

## What the scripts do

Each script:
1. Constructs a continuous-time Markov generator for a four-state energy landscape with specified energies and barriers.
2. Diagonalizes the generator and evaluates the linear-response dissipation as a function of driving frequency.
3. Computes an independent two-level-system (TLS) sum for comparison.
4. Plots the resulting dissipation spectra on log–log axes and saves the figure as a PDF.

The scripts are self-contained and do not rely on external data files.

## Requirements

- Python 3.x  
- NumPy  
- SciPy  
- Matplotlib (with LaTeX support enabled for labels)

A typical scientific Python environment (e.g. Anaconda) is sufficient.

## How to run

From this directory, simply execute:

```bash
python cycle_v_chain.py
python Mountain.py
