'''
This script reads the final ensemble from GOAT and writes input files for each conformer to be optimized with ORCA.
'''

from pathlib import Path
import pandas as pd
from utils import create_dirs, write_orca_input, write_batch_file, hartree_to_kcal_mol, write_xyz_file

def parse_conformers(input_text):
    """Parse conformers from input text and return list of (energy, atoms, coordinates)"""
    conformers = []
    lines = input_text.strip().split('\n')

    i = 0
    while i < len(lines):
        # Get number of atoms
        n_atoms = int(lines[i])

        # Get energy from next line (assumes format: "-43.6943482057 converged=true")
        energy = float(lines[i + 1].split()[0])

        # Get atomic coordinates
        atoms = []
        coords = []
        for j in range(n_atoms):
            line = lines[i + 2 + j].split()
            atoms.append(line[0])
            coords.append([float(x) for x in line[1:4]])

        conformers.append((energy, atoms, coords))
        i += n_atoms + 2  # Move to next conformer

    return conformers


def main():
    ligands_info = pd.read_csv("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/Ligands.csv")
    jobs_dir = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")
    for ligand_dir in jobs_dir.iterdir():
        ligand_name = ligand_dir.name
        ligand_info = ligands_info[ligands_info["name"] == ligand_name]
        charge = ligand_info["charge"].values[0]


        ensemble_dir = create_dirs(ligand_name, "Ensemble_Opt", "CuOA", jobs_dir, output=False)
        ensemble_name = ensemble_dir.stem

        goat_output = jobs_dir.joinpath(ligand_name, 'GOAT', f"{ligand_name}_CuOA_GOAT", "output", 'input.finalensemble.xyz')
        input_text = goat_output.read_text()

        conformers = parse_conformers(input_text)
        # TODO: Calculate relative energies

        for i, (energy, atoms, coords) in enumerate(conformers, 1):
            energy_kcal = hartree_to_kcal_mol(energy)

            # create conformer directory
            conf_dir = ensemble_dir.joinpath(f"conformer_{i}")
            try:
                if conf_dir.joinpath("output").exists():
                    raise FileExistsError
                else:
                    Path.mkdir(conf_dir, exist_ok=True)

                    # write xyz file with energy
                    xyz_path = conf_dir.joinpath(f"conf{i}_input.xyz")
                    write_xyz_file(xyz_path, energy_kcal, atoms, coords)

                    # write orca input and batch file
                    job_name = f"{ensemble_name}_conf{i}"
                    write_orca_input(conf_dir, job_name, scf_type="UKS", functional="TPSS", basis_set="def2-SVP",
                                     basis_options='NewGTO Cu "def2-TZVP"', sp_basis_set="def2-TZVP",
                                     dispersion_correction="D3BJ",
                                     runtype="Opt Freq", options="SARC/J RI", miscellaneous="KeepDens TightSCF",
                                     charge=charge,
                                     multiplicity=2, elprop="Quadrupole True,Polar 1", nbo="NBO", single_point=True,
                                     memory="1000")
                    write_batch_file(job_name, conf_dir, xyz_path, memory="16GB", tasks_per_node=12)

                    # create output directory
                    Path.mkdir(conf_dir.joinpath("output"), exist_ok=True)
                    print(f"Written conformer {i} with energy {energy_kcal:.2f} kcal/mol to {xyz_path}")

            except FileExistsError:
                print(f"Conformer {i} already exists in {conf_dir}")
                continue



if __name__ == "__main__":
    main()