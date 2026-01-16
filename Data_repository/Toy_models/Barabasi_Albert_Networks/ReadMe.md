# Dissipation Spectra of Random Network Energy Landscapes

This folder contains Python scripts used to generate dissipation spectra for large random energy landscapes represented as Barabasi-Albert networks. The calculations average over many random graph and energy realizations to extract robust, frequency-dependent dissipation behavior.

## Contents

- **`Figure_3.py`**  
  Computes dissipation spectra for Barabási–Albert networks with varying connectivity (controlled by the attachment parameter \(m\)). Results are averaged over many random realizations and compared to an independent two-level-system (TLS) sum.  
  This script produces the data and plot shown in Figure 3.

- **`Figure_5.py`**  
  Computes dissipation spectra for Barabási–Albert networks while varying the maximum node energy scale. Results are averaged over many random realizations and compared to a TLS sum.  
  This script produces the data and plot shown in Figure 5.

## What the scripts do

Each script:
1. Generates large random networks (Barabási–Albert graphs) representing energy landscapes.
2. Assigns random node energies and random energy barriers to edges.
3. Constructs a continuous-time Markov generator for thermally activated transitions.
4. Diagonalizes the generator and evaluates the linear-response dissipation as a function of driving frequency.
5. Averages dissipation spectra over many independent realizations.
6. Computes an independent TLS sum for comparison.
7. Produces log–log plots of dissipation versus frequency.

The scripts are fully self-contained and do not require external data files.

## Requirements

- Python 3.x  
- NumPy  
- SciPy  
- NetworkX  
- Matplotlib (with LaTeX support enabled for labels)

A standard scientific Python distribution (e.g. Anaconda) is sufficient.

## How to run

From this directory, run:

```bash
python Figure_3.py
python Figure_5.py
