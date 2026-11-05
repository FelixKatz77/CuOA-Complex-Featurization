'''
This script calculates steric descriptors from ORCA output files and saves them to a CSV file.
'''


import json
import sys
from pathlib import Path

import dbstep.Dbstep as db
import numpy as np
import pandas as pd
from morfeus import BiteAngle, read_xyz, Dispersion
from pymatgen.io.xyz import XYZ

from utils import read_energy, check_orca_success, check_optimization_success, check_neg_frequencies, \
    find_coordinating_atom_ids, find_atom_id, check_optimization_status, GEOMETRY_INPUT_DIR, JOBS_DIR


def calculate_boltzmann_average(df, energy_col='Energy', T=298.15):
    '''Calculate the Boltzmann average of a DataFrame'''

    # Constants
    R = 0.001987204  # Gas constant in kcal/mol⋅K

    # Calculate Boltzmann weights
    energies = df[energy_col]
    min_energy = energies.min()
    rel_energies = energies - min_energy
    weights = np.exp(-rel_energies / (R * T))
    weights = weights / weights.sum()

    # Calculate weighted average for all columns except energy
    other_cols = [col for col in df.columns if col != energy_col]
    boltz_averages = {}

    for col in other_cols:
        boltz_averages[col] = np.sum(df[col] * weights)

    return pd.Series(boltz_averages), weights


def calculate_lig_cu_bond_length(xyz_file, cu_id, coord_ids):
    '''Calculate the bond length between Cu and a coordinating ligand atom'''

    coord_id_1 = coord_ids[0]
    coord_id_2 = coord_ids[1]
    elements, coordinates = read_xyz(xyz_file)
    bond_length_1 = np.linalg.norm(coordinates[cu_id] - coordinates[coord_id_1])
    bond_length_2 = np.linalg.norm(coordinates[cu_id] - coordinates[coord_id_2])
    longer_bond = max(bond_length_1, bond_length_2)
    shorter_bond = min(bond_length_1, bond_length_2)
    avg_bond_length = (bond_length_1 + bond_length_2) / 2
    return {'Cu-L_Bond_length_1': shorter_bond, 'Cu-L_Bond_length_2': longer_bond,
            'Cu-L_Bond_length_avg': avg_bond_length}

def calculate_steric_descriptors(xyz_file, cu_id, br_id, coord_id1, coord_id2):
    '''Calculate sterimol, buried volume and bite angle descriptors for a Cu-Br complex'''

    cu_id += 1
    br_id += 1
    coord_id1 += 1
    coord_id2 += 1
    elements, coordinates = read_xyz(xyz_file)
    sterimol_L = db.dbstep(str(xyz_file), atom1=cu_id, atom2=br_id, commandline=True, verbose=True, sterimol=True,
                           measure='classic')
    sterimol_B = db.dbstep(str(xyz_file), atom1=br_id, atom2=cu_id, commandline=True, verbose=True, sterimol=True,
                           measure='classic')
    bur_v = db.dbstep(str(xyz_file), atom1=cu_id, commandline=True, verbose=True, volume=True, measure='classic')
    bite_angle = BiteAngle(coordinates, cu_id, coord_id1, coord_id2).angle
    # TODO: Add Pyramidalization
    return {'L': sterimol_L.L, 'Bmin': sterimol_B.Bmin, 'Bmax': sterimol_B.Bmax, 'BurVol': bur_v.bur_vol,
            'BiteAngle': bite_angle}

def calculate_dispersion(xyz_file):
    '''Calculate the dispersion descriptor'''

    elements, coordinates = read_xyz(xyz_file)
    disp = Dispersion(elements, coordinates, compute_coefficients=False)
    disp.compute_coefficients(model="gd4")
    #disp.surface_from_cube("corannulene.cub")
    disp.compute_p_int()
    p_int = disp.p_int
    p_min = disp.p_min
    p_max = disp.p_max
    return {'p_int': p_int, 'p_min': p_min, 'p_max': p_max}

if __name__ == '__main__':
    template_dir = GEOMETRY_INPUT_DIR
    jobs_dir = JOBS_DIR
    species = "CuOA"
    for ligand_dir in jobs_dir.iterdir():
        if ligand_dir.is_dir() and ligand_dir.name == sys.argv[1]: #and ligand_dir.name == sys.argv[1]:
            ligand_name = ligand_dir.name  # ligand_name of the ligand
            ensemble_path = jobs_dir.joinpath(ligand_name, "Ensemble_Opt", f"{ligand_name}_{species}_Ensemble_Opt")  # Path to ensemble directory

            steric_descriptors = {}

            # Loop over all conformers in the ensemble
            for conformer_dir in ensemble_path.glob('conformer_*'):
                sp_path = ligand_dir.joinpath("Single_Point",
                                              f"{ligand_name}_{species}_Single_Point", conformer_dir.name)

                orca_output = conformer_dir.joinpath("output", f"output_{ligand_name}_CuOA_Ensemble_Opt_conf{conformer_dir.name.split('_')[-1]}.out")
                xyz_path = conformer_dir.joinpath("output", 'input.xyz')  # Path to XYZ file
                temp_xyz_file = template_dir.joinpath(f"{ligand_name}.xyz")

                if check_optimization_status(orca_output, xyz_path, temp_xyz_file):

                    opt_json = conformer_dir.joinpath("output", 'input.property.json')
                    sp_json = sp_path.joinpath("output", 'input.property.json')
                    energy = read_energy(opt_json, sp_json)

                    cu_id = find_atom_id(xyz_path, 'Cu')  # Find atom IDs of Cu and Br
                    br_id = find_atom_id(xyz_path, 'Br')
                    coord_ids = find_coordinating_atom_ids(cu_id, xyz_path)  # Find atom IDs of coordinating atoms

                    sterics = calculate_steric_descriptors(xyz_path, cu_id, br_id,
                                                           coord_ids[0], coord_ids[1]) # Calculate steric descriptors
                    dispersion_desc = calculate_dispersion(xyz_path) # Calculate dispersion descriptors
                    sterics.update(dispersion_desc)
                    cu_l_bond_desc = calculate_lig_cu_bond_length(xyz_path, cu_id, coord_ids)
                    sterics.update(cu_l_bond_desc)
                    sterics['Energy'] = energy # Add energy to dictionary
                    steric_descriptors[conformer_dir.name] = sterics # Add to descriptor dictionary

            desc_df = pd.DataFrame(steric_descriptors).T # Convert dictionary to DataFrame

            boltz_avg, weights = calculate_boltzmann_average(desc_df) # Calculate Boltzmann average
            min_values = desc_df.min() # Get minimum values
            max_values = desc_df.max() # Get maximum values

            stat = pd.concat([min_values, max_values, boltz_avg], axis=1, keys=['Min', 'Max', 'Boltzmann Average']) # Concat averages, min and max
            desc_df = pd.concat([desc_df, stat.T], axis=0) # Add values to DataFrame

            # Save DataFrame to CSV
            desc_dir = jobs_dir.joinpath(ligand_name, 'steric_descriptors')
            Path.mkdir(desc_dir, exist_ok=False)
            desc_df.to_csv(desc_dir.joinpath(f"{ligand_name}_{species}_steric_descriptors.csv"))

