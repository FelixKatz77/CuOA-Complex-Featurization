'''
This script collects the electronic and steric descriptors of all ligands, 
adds mean/max/min rows and ligand classes, and saves them to combined CSV files.
'''

import pandas as pd
from pathlib import Path

from utils import JOBS_DIR

jobs_dir = JOBS_DIR

ligand_classes = {
    "Symmetric Diamide": ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L31"],
    "Asymmetric Diamide": ["L9", "L10"],
    "Phenanthrolinē": ["L11", "L12", "L13", "L14"],
    "Dipyrrazole": ["L15", "L16", "L17"],
    "6-Hydroxy Picolinohydrazide": ["L18", "L19", "L20", "L21"],
    "N,N-Diarylbenzene-1,2-diamine": ["L22", "L23", "L24"],
    "Amino-2-oxoacetic acid": ["L25", "L26", "L27"],
    "6‑Hydroxypicolinamide": ["L28", "L29", "L30", "L32"]
}
ligand_to_class = {lig: cls for cls, ligs in ligand_classes.items() for lig in ligs}

def get_ligand_class(ligand_name):
    """
    Get the class of a ligand based on its name.

    Parameters:
    - ligand_name (str): Name of the ligand.

    Returns:
    - str: Class of the ligand or 'Unknown' if not found.
    """
    return ligand_to_class.get(ligand_name, 'Unknown')

for i, ligand_dir in enumerate(jobs_dir.iterdir()):
    if ligand_dir.is_dir():
        ligand_name = ligand_dir.name
        elec_csv = ligand_dir.joinpath("electronic_descriptors", f"{ligand_name}_CuOA_electronic_descriptors.csv")
        steric_csv = ligand_dir.joinpath("steric_descriptors", f"{ligand_name}_CuOA_steric_descriptors.csv")
        curr_elec_df = pd.read_csv(elec_csv, index_col=0)
        curr_steric_df = pd.read_csv(steric_csv, index_col=0)
        # Add ligand before each index of the current DataFrame
        curr_elec_df.index = [f"{ligand_name}_{i}" for i in curr_elec_df.index]
        curr_steric_df.index = [f"{ligand_name}_{i}" for i in curr_steric_df.index]

        if i == 1:
            electronic_df = curr_elec_df
            steric_df = curr_steric_df
        else:
            electronic_df = pd.concat([electronic_df, curr_elec_df])
            steric_df = pd.concat([steric_df, curr_steric_df])

mean_elec_values = electronic_df.mean()
max_elec_values = electronic_df.max()
min_elec_values = electronic_df.min()
electronic_df = pd.concat([electronic_df, mean_elec_values.to_frame(name="Mean").T, max_elec_values.to_frame(name="Max").T, min_elec_values.to_frame(name="Min").T])

#Add ligand classes to the electronic descriptors DataFrame
electronic_df['Ligand Class'] = electronic_df.index.to_series().apply(lambda x: get_ligand_class(x.split('_')[0]))
electronic_df.to_csv(JOBS_DIR / "electronic_descriptors_synthesized.csv")

mean_steric_values = steric_df.mean()
max_steric_values = steric_df.max()
min_steric_values = steric_df.min()
steric_df = pd.concat([steric_df, mean_steric_values.to_frame(name="Mean").T, max_steric_values.to_frame(name="Max").T, min_steric_values.to_frame(name="Min").T])

#Add ligand classes to the electronic descriptors DataFrame
steric_df['Ligand Class'] = steric_df.index.to_series().apply(lambda x: get_ligand_class(x.split('_')[0]))
steric_df.to_csv(JOBS_DIR / "steric_descriptors_synthesized.csv")
pass

