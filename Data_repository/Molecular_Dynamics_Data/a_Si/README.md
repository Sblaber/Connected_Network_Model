# Mechanical Loss / Dissipation Calculation (a-Si)

This folder contains a Python script that computes frequency-dependent dissipation for a configuration-network (CN) model of amorphous silicon (a-Si), and compares it to a baseline TLS estimate.

---

## What this script does

For each of 10 samples:

1. **Loads network/landscape data**
   - Barrier energies `V`
   - Energy asymmetries `D`
   - Minimum energies `E_1`, `E_2`
   - Node connectivity list `Connections`
   - Stress components `s_1`–`s_6` and “before” values (used to compute deformation-potential-like couplings)

2. **Builds a graph and selects the largest connected component**
   - Constructs a directed graph with edge attributes (barrier, asymmetry, endpoint energies, stress components).
   - Keeps the largest connected component for analysis.

3. **Builds a rate matrix `R` and diagonalizes it**
   - Transition rates are Arrhenius in the barrier height relative to each state energy.
   - The rate matrix is constructed to have **column-sum zero** (diagonal entries are minus the sum of outgoing rates in that column).
   - Computes eigenvalues/eigenvectors of `R`.

4. **Computes CN dissipation vs frequency**
   - For each driving frequency `omega`, constructs a frequency-response factor using the eigenvalues.
   - Computes:
     - `Q(omega)` (sum all eigenmodes)
     - `Q_L(omega)` (longitudinal component)
     - `Q_T(omega)` (transverse component)

5. **Computes a TLS-style dissipation curve for comparison**
   - Uses the same edge-level barrier/asymmetry data to compute a standard TLS relaxation form, summed over transitions.

6. **Plots curves for each sample**
   - Plots scaled CN dissipation (total, longitudinal, transverse) and the TLS sum.

---

## Inputs and expected folder structure

The script expects the following relative paths (as written in the code):

- `./Data/`
  - `V_monotonicP_wofirstlast_sample_10.npy`
  - `Connections10.npy`
  - `D_asymm_sanity_monotonicP_wofirstlast_sample_10.npy`
  - `E_initial_monotonicP_wofirstlast_sample_10.npy`
  - `E_final_monotonicP_wofirstlast_sample_10.npy`
  - `list_1.txt` … `list_10.txt`
  - `s_1.npy` … `s_6.npy`

---

## Outputs

- A Matplotlib figure window containing the overlaid curves for all samples.
- The script includes commented-out `np.save(...)` calls you can re-enable to write arrays to disk (frequency, Q curves, TLS curves).

No files are written by default.

---

## How to run

From this folder:

```bash
python Calculate_Mechanical_Loss.py
