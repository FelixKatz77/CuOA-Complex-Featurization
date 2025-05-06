'''
This script extracts electronic descriptors from ORCA output files and saves them to a CSV file.
'''

from morfeus import read_xyz
from morfeus import XTB
import pandas as pd
import json
from pathlib import Path
import re
from pymatgen.io.xyz import XYZ
from utils import find_atom_id, find_coordinating_atom_ids
import numpy as np


def get_charge(filepath):

    '''
    Get charge from ORCA input file
    '''

    with open(filepath, 'r') as file:
        for line in file:
            if "*xyzfile" in line:
                charge = int(line.split(' ')[1])
                return charge
    return None

def parse_orbital_energies(filepath):
    """
    Parse orbital energies section from an ORCA output file.
    """
    orbital_data = []
    reading_orbitals = False

    with open(filepath, 'r') as file:
        for line in file:
            # Start reading when we find the header
            if "ORBITAL ENERGIES" in line:
                reading_orbitals = True
                # Skip the header lines
                next(file)  # Skip the dashes
                next(file)  # Skip the column headers
                continue

            # Stop reading if we hit an empty line after starting
            if reading_orbitals and not line.strip():
                break

            # Parse orbital data
            if reading_orbitals:
                # Skip any lines with asterisks (footnotes)
                if '*' in line:
                    continue

                if 'NO' in line:
                    continue

                # Split the line and convert to appropriate types
                parts = line.split()
                if len(parts) == 4:  # Ensure we have all columns
                    orbital_data.append({
                        'NO': int(parts[0]),
                        'OCC': float(parts[1]),
                        'E(Eh)': float(parts[2]),
                        'E(eV)': float(parts[3])
                    })

    # Create DataFrame
    df = pd.DataFrame(orbital_data)
    return df

def calc_front_orb_derived_descriptors(e_homo, e_lumo):
    """Calculate front molecular orbital derived descriptors"""

    homo_lumo_gap = e_lumo - e_homo
    molecular_electronegativity = (e_lumo + e_homo) / 2
    global_electrophilicity = molecular_electronegativity ** 2 / 2 * homo_lumo_gap
    global_nucleophilicity = 1 / global_electrophilicity
    return {'HOMO-LUMO gap': homo_lumo_gap, 'Molecular electronegativity': molecular_electronegativity,
            'Global electrophilicity parameter': global_electrophilicity,
            'Global nucleophilicity parameter': global_nucleophilicity}


def get_front_orb_derived_descriptors(filepath):
    """Get front molecular orbital derived descriptors"""

    orb_df = parse_orbital_energies(filepath)
    homo_idx = orb_df[orb_df['OCC'] > 0]['NO'].max()
    lumo_idx = homo_idx + 1

    e_homo = orb_df[orb_df['NO'] == homo_idx]['E(eV)'].iloc[0]
    e_lumo = orb_df[orb_df['NO'] == lumo_idx]['E(eV)'].iloc[0]
    orb_desc = calc_front_orb_derived_descriptors(e_homo, e_lumo)
    return orb_desc

def calc_fukui_descriptors(xyz_path, charge=0):
    """Calculate fukui function derived descriptors"""

    elements, coordinates = read_xyz(xyz_path)
    xtb = XTB(elements, coordinates, charge=charge)
    f_elec = xtb.get_fukui("electrophilicity")
    f_nuc = xtb.get_fukui("nucleophilicity")
    return {'Fukui electrophilicity': f_elec, 'Fukui nucleophilicity': f_nuc}


def extract_quadrupole_traceless(filepath):
    """Extract the traceless form quadrupole moment values from an ORCA output file."""

    with open(filepath, 'r') as file:
        content = file.read()

    # Look for the quadrupole moment section
    quad_section = re.search(
        r'-----------------\s*'
        r'QUADRUPOLE MOMENT\s*'
        r'-----------------.*?'
        r'traceless form in Buckingham\)\s*'
        r'\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)',
        content, re.DOTALL)

    if quad_section:
        # Extract the three traceless form values
        values = [float(val) for val in quad_section.groups()]
        quad_amplitude = sum(val**2 for val in values) ** 0.5
        quad_values = {
            'Quad_amp': quad_amplitude,
            'Quad_xx': values[0],
            'Quad_yy': values[1],
            'Quad_zz': values[2],
        }
        return quad_values
    else:
        print("Warning: Quadrupole moment section not found in file")
        return None

def get_mol_desc(properties_dict):
    '''Extract molecular descriptors from ORCA output'''

    dipole_moment = properties_dict['Dipole_Moment']['DIPOLEMAGNITUDE']
    polarizability = properties_dict['Polarizability']['ISOTROPICPOLAR']
    return {'Dipole': dipole_moment, 'Polarizability': polarizability}

def identify_phenyl_carbon(xyz_path):
    cu_id = find_atom_id(xyz_path, 'Cu')
    mol = XYZ.from_file(xyz_path).molecule
    cu_coords = mol.sites[cu_id].coords
    neighbors = []
    for i, site in enumerate(mol.sites):
        if i != cu_id:  # Skip self
            if site.species_string == 'C':  # Only look at O and N
                dist = np.linalg.norm(cu_coords - site.coords)
                neighbors.append((i, site.species_string, dist))

    # Sort by distance and get closest
    neighbors.sort(key=lambda x: x[2])
    closest = neighbors[:1]
    return closest[0]

def get_charge_desc(properties_dict):
    '''Extract charge descriptors for Cu and Br from ORCA output'''

    for i, component in enumerate(properties_dict['Mayer_Population_Analysis']['COMPONENTS']):
        if 29 in component and 35 in component:
            mayer_index = i
    mayer_bo_ord = properties_dict['Mayer_Population_Analysis']['BONDORDERS'][mayer_index][0] #Mayer bond order between Cu and Br
    for j, no in enumerate(properties_dict['Mulliken_Population_Analysis']['ATNO']):
        if 35 in no:
            br_index = j
        if 29 in no:
            cu_index = j
    mulliken_charge_br = properties_dict['Mulliken_Population_Analysis']['ATOMICCHARGES'][br_index]
    mulliken_charge_cu = properties_dict['Mulliken_Population_Analysis']['ATOMICCHARGES'][cu_index]
    return {'Mayer Bond Order Cu-Br': mayer_bo_ord, 'Muliken Charge Br': mulliken_charge_br,
            'Mulliken Charge Cu': mulliken_charge_cu}


def extract_npa_table(file_path):
    """
    Extract the Natural Population Analysis table from an orca output file and convert it to a pandas DataFrame.
    """
    # Initialize variables
    table_found = False
    table_ended = False
    table_lines = []

    # Read the file and extract table lines
    with open(file_path, 'r') as file:
        for line in file:
            # Check for the start of the table
            if "Summary of Natural Population Analysis:" in line:
                table_found = True
                continue

            # Skip header lines
            if table_found and "Natural Population" in line:
                continue
            if table_found and "Natural" in line and "Core" in line:
                continue
            if table_found and "---" in line:
                continue

            # Collect data lines
            if table_found and not table_ended and line.strip():
                # Check if we've reached the end of the table
                if "====" in line:
                    table_ended = True
                    continue

                # Skip the Total line
                if "* Total *" in line:
                    continue

                # Add the line to our collection
                table_lines.append(line.strip())

    # If no table was found, return empty DataFrame
    if not table_found or not table_lines:
        return pd.DataFrame()

    # Parse the table lines into a list of dictionaries
    data = []
    for line in table_lines:
        # Use regex to extract values from each line
        match = re.search(r'(\w+)\s+(\d+)\s+([-]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match:
            atom, atom_no, charge, core, valence, rydberg, total = match.groups()
            data.append({
                'Atom': atom,
                'No': int(atom_no),
                'Natural Charge': float(charge),
                'Core': float(core),
                'Valence': float(valence),
                'Rydberg': float(rydberg),
                'Total': float(total)
            })

    # Create DataFrame from the parsed data
    df = pd.DataFrame(data)

    return df

def get_nbo_energies(output_file, xyz_file):
    '''Extract NBO energies from ORCA output'''

    npa_df = extract_npa_table(output_file)
    cu_id = find_atom_id(xyz_file, 'Cu')
    phenyl_c_id = identify_phenyl_carbon(xyz_file)
    coord1, coord2 = find_coordinating_atom_ids(cu_id, xyz_file)
    cu_natural_charge = npa_df[npa_df['Atom'] == 'Cu']['Natural Charge'].values[0]
    br_natural_charge = npa_df[npa_df['Atom'] == 'Br']['Natural Charge'].values[0]
    c_natural_charge = npa_df[npa_df['No'] == phenyl_c_id+1]['Natural Charge'].values[0]
    coord1_natural_charge = npa_df[npa_df['No'] == coord1+1]['Natural Charge'].values[0]
    coord2_natural_charge = npa_df[npa_df['No'] == coord2+1]['Natural Charge'].values[0]
    return {'Cu Natural Charge': cu_natural_charge, 'Br Natural Charge': br_natural_charge,
            'C_Phenyl Natural Charge': c_natural_charge, 'Coord1_Natural_Charge': coord1_natural_charge,
            'Coord2_Natural_Charge': coord2_natural_charge}

if __name__ == '__main__':
    jobs_dir = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")
    species = "CuOA"
    for ligand_dir in jobs_dir.iterdir():
        ligand_name = ligand_dir.name  # ligand_name of the ligand
        ensemble_path = jobs_dir.joinpath(ligand_name, "Ensemble_Opt",
                                           f"{ligand_name}_{species}_Ensemble_Opt")  # Path to ensemble directory
        for conformer_dir in ensemble_path.glob('conformer_*'):
            conf_no = conformer_dir.name.split('_')[-1]
            input_path = conformer_dir.joinpath("output", 'input.inp')
            txt_path = conformer_dir.joinpath("output", f'output_{ligand_name}_{species}_Ensemble_Opt_conf{conf_no}.out')
            json_path = conformer_dir.joinpath("output", 'input.property.json')
            orca_dict = json.loads(json_path.read_text())
            property_dict = orca_dict[f'Geometry_{len(orca_dict)-1}']
            charge = get_charge(input_path)
            orb_desc = get_front_orb_derived_descriptors(txt_path)
            fukui_desc = calc_fukui_descriptors(ligand_dir.joinpath('NBO', f"{ligand_name}_{species}_NBO", f"conformer_{conf_no}", 'output', 'input.log'), charge=charge)
            mol_desc = get_mol_desc(property_dict)
            charge_desc = get_charge_desc(property_dict)
            nbo_desc = get_nbo_energies(txt_path, conformer_dir.joinpath('output', 'input.xyz'))
            elec_desc = {**orb_desc, **fukui_desc, **mol_desc, **charge_desc}
            desc_df = pd.DataFrame(elec_desc).T

            # Save DataFrame to CSV
            desc_dir = jobs_dir.joinpath(ligand_name, 'electronic_descriptors')
            Path.mkdir(desc_dir, exist_ok=False)
            desc_df.to_csv(desc_dir.joinpath(f"{ligand_name}_{species}_electronic_descriptors.csv"))

