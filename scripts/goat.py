'''
This script is used to generate the input files for the GOAT calculations.
'''

import shutil
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolFromSmiles

from utils import create_dirs, write_orca_input, write_batch_file, LIGANDS_CSV, JOBS_DIR, GEOMETRY_INPUT_DIR

species = 'CuOA'
jobs_path = JOBS_DIR

ligands_info = pd.read_csv(LIGANDS_CSV)
for xyz_file in GEOMETRY_INPUT_DIR.glob("*.xyz"):
    ligand_name = xyz_file.stem
    ligand_info = ligands_info[ligands_info["name"] == ligand_name]
    if species == "Ligand":
        charge = ligand_info["charge_ligand"].values[0]
        multiplicity = ligand_info["multiplicity_ligand"].values[0]
    else:
        charge = ligand_info["charge_complex"].values[0]
        multiplicity = ligand_info["multiplicity_complex"].values[0]

    job_directory = create_dirs(ligand_name, "GOAT", species, jobs_path=jobs_path)
    job_name = job_directory.stem

    xyz_path =  job_directory.joinpath(f"{ligand_name}.xyz")
    if species == "Ligand":
        mol = MolFromSmiles(ligand_info[f"smiles_ligand"].values[0])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.MolToXYZFile(mol, str(xyz_path))
    else:
        shutil.copy(xyz_file, xyz_path)

    write_orca_input(job_directory, job_directory.stem,
                        method="XTB GOAT",
                        charge=charge,
                        multiplicity=multiplicity,
                        goat_options="MAXEN 3.5",
                        n_procs='12',
                        memory='1000')

    write_batch_file(job_name, job_directory, xyz_path, memory="16GB", tasks_per_node=12)
