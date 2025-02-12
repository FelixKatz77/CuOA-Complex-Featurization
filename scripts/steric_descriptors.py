import json
from pathlib import Path

import dbstep.Dbstep as db
import numpy as np
import pandas as pd
from morfeus import BiteAngle, read_xyz
from pymatgen.io.xyz import XYZ

jobs_path = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/")


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


def read_energy(filename):
    '''Read the energy from an ORCA output file and add thermal corrections'''

    orca_output = json.loads(filename.read_text())
    for geometry in orca_output.keys():
        if 'THERMOCHEMISTRY_Energies' in orca_output[geometry]:
            final_eel = orca_output[geometry]['DFT_Energy']['FINALEN']
            gibb_e = orca_output[geometry]['THERMOCHEMISTRY_Energies']['FREEENERGYG']
            eel = orca_output[geometry]['THERMOCHEMISTRY_Energies']['ELENERGY']
            thermal_corr_gibbs = gibb_e - eel
            return final_eel + thermal_corr_gibbs
    return None


def find_atom_ids(xyz_path, element1, element2):
    '''Find the atom IDs of two elements in an XYZ file'''

    input_text = xyz_path.read_text()
    lines = input_text.strip().split('\n')
    br_id = None
    cu_id = None
    if element1 not in input_text or element2 not in input_text:
        raise ValueError(f"Atom {element1} or {element2} not found in {xyz_path}")
    # Skip the first two lines (header)
    for i, line in enumerate(lines[2:], start=0):
        if line.strip().startswith(element1):
            atom1_id = i
        elif line.strip().startswith(element2):
            atom2_id = i

    return atom1_id, atom2_id


def find_coordinating_atom_ids(xyz_path, cu_id):
    '''Find the atom IDs of the two N or O atoms closest to a given atom in an XYZ file'''

    mol = XYZ.from_file(xyz_path).molecule
    cu_coords = mol.sites[cu_id].coords
    neighbors = []
    for i, site in enumerate(mol.sites):
        if i != cu_id:  # Skip self
            if site.species_string in ['O', 'N']:  # Only look at O and N
                dist = np.linalg.norm(cu_coords - site.coords)
                neighbors.append((i, site.species_string, dist))

    # Sort by distance and get two closest
    neighbors.sort(key=lambda x: x[2])
    closest_two = neighbors[:2]
    ids = [neighbor[0] for neighbor in neighbors]
    return ids


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
    bur_v = db.dbstep(str(xyz_file), atom1=br_id, commandline=True, verbose=True, volume=True, measure='classic')
    bite_angle = BiteAngle(coordinates, cu_id, coord_id1, coord_id2).angle
    # TODO: Add Pyramidalization
    return {'L': sterimol_L.L, 'Bmin': sterimol_B.Bmin, 'Bmax': sterimol_B.Bmax, 'BurVol': bur_v.bur_vol,
            'BiteAngle': bite_angle}


if __name__ == '__main__':
    name = "L1"  # name of the ligand
    species = "CuBr"  # species
    job_type = "Ensemble_Opt"  # job type
    ensemble_path = jobs_path.joinpath(name, job_type, f"{name}_{species}_{job_type}")  # Path to ensemble directory

    steric_descriptors = {}

    # Loop over all conformers in the ensemble
    for conformer_dir in ensemble_path.glob('conformer_*'):
        energy = read_energy(conformer_dir.joinpath("output", 'input.property.json'))  # Read energy from ORCA output
        xyz_path = conformer_dir.joinpath("output", 'input.xyz')  # Path to XYZ file

        cu_id, br_id = find_atom_ids(xyz_path, 'Cu', 'Br')  # Find atom IDs of Cu and Br
        coord_ids = find_coordinating_atom_ids(xyz_path, cu_id)  # Find atom IDs of coordinating atoms

        sterics = calculate_steric_descriptors(conformer_dir.joinpath('output', 'input.xyz'), cu_id, br_id,
                                               coord_ids[0], coord_ids[1]) # Calculate steric descriptors
        sterics['Energy'] = energy # Add energy to dictionary
        steric_descriptors[conformer_dir.name] = sterics # Add to descriptor dictionary

    desc_df = pd.DataFrame(steric_descriptors).T # Convert dictionary to DataFrame

    boltz_avg, weights = calculate_boltzmann_average(desc_df) # Calculate Boltzmann average
    min_values = desc_df.min() # Get minimum values
    max_values = desc_df.max() # Get maximum values

    stat = pd.concat([min_values, max_values, boltz_avg], axis=1, keys=['Min', 'Max', 'Boltzmann Average']) # Concat averages, min and max
    desc_df = pd.concat([desc_df, stat.T], axis=0) # Add values to DataFrame

    # Save DataFrame to CSV
    desc_dir = jobs_path.joinpath(name, 'steric_descriptors')
    Path.mkdir(desc_dir, exist_ok=False)
    desc_df.to_csv(desc_dir.joinpath(f"{name}_{species}_steric_descriptors.csv"))

