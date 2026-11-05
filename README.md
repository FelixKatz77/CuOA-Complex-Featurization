# CuOA-Complex-Featurization

Scripts for the computational featurization of Cu(II) oxidative-addition complexes with steric and
electronic descriptors. Descriptors are computed for each conformer and Boltzmann-averaged at 298.15 K.

## Workflow

1. **Conformer search**: GOAT (ORCA 6.0.1, GFN2-xTB) starting from MMFF-preorganised square-planar templates.
2. **Optimisation and frequencies**: UKS-TPSS-D3BJ/def2-SVP (def2-TZVP on Cu). Conformers are kept only if they converged, have no imaginary modes and keep the template connectivity.
3. **Property calculations** on the validated conformers: TPSS/def2-TZVP single points, NBO/Hirshfeld/CM5 population analysis (Gaussian 16 + NBO7), and vertical one-electron oxidation/reduction for E_ox/E_red.
4. **Descriptor extraction**: steric descriptors (DBSTEP, morfeus) and electronic descriptors (ORCA, NBO, Multiwfn, xTB Fukui indices), followed by statistics across the ligand library.

The charge and multiplicity of each ligand/complex are read from `data/Ligands.csv`. All jobs are submitted through SLURM.

## Repository contents

| File | Description |
|---|---|
| `scripts/goat.py` | Writes ORCA GOAT conformer-search inputs |
| `scripts/ensemble_optimization.py` | Parses the final GOAT ensemble and writes DFT Opt+Freq inputs for each conformer |
| `scripts/check_optim_jobs.py` | Checks SCF/optimisation convergence, imaginary frequencies and connectivity |
| `scripts/single_points.py` | TPSS/def2-TZVP single points on the optimised conformers |
| `scripts/nbo.py` | Gaussian 16 inputs for NBO, Hirshfeld and CM5 population analysis |
| `scripts/e_ox_red.py` | Inputs for the oxidised and reduced complexes (vertical, at the optimised geometry) |
| `scripts/steric_descriptors.py` | Sterimol parameters, %V_bur, bite angle, Cu–L bond lengths and dispersion descriptors; Boltzmann averaging |
| `scripts/electronic_descriptors.py` | Frontier-orbital-derived descriptors, Fukui indices, quadrupole moment, NPA/Hirshfeld/CM5 charges, Laplacian bond orders, E_ox/E_red |
| `scripts/descriptor_statistics.py` | Merges the descriptors of all ligands, assigns ligand classes and reports mean/min/max |
| `scripts/solvents.py` | Looks up Hansen solubility parameters (δd, δp, δh) for a solvent from its SMILES |
| `scripts/utils.py` | Shared helpers: ORCA/Gaussian input writing, output parsing, job checks, atom identification |
| `scripts/submit.batch`, `scripts/gaussian.batch` | SLURM templates for ORCA and Gaussian jobs |
| `scripts/steric_descriptors.batch`, `scripts/elec_descriptors.batch` | SLURM wrappers for descriptor extraction |
