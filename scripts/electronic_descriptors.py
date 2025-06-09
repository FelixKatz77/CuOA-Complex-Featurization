'''
This script extracts electronic descriptors from ORCA output files and saves them to a CSV file.
'''
import subprocess

import resource
from morfeus import read_xyz
from morfeus import XTB
import pandas as pd
import json
from pathlib import Path
import sys
from pymatgen.io.xyz import XYZ

from steric_descriptors import calculate_boltzmann_average
from utils import find_atom_id, find_coordinating_atom_ids, run_command, extract_floats, read_energy
import numpy as np
import os
import re

MULTIWFN_PATH = "/home/employee/f_katz01/Software/Multiwfn/Multiwfn_noGUI"

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

def calc_fukui_descriptors(xyz_file, charge, cu_id, br_id, phenyl_c_id):
    """Calculate fukui function derived descriptors"""

    cu_id += 1
    br_id += 1
    phenyl_c_id += 1
    elements, coordinates = read_xyz(xyz_file)
    xtb = XTB(elements, coordinates, charge=charge)
    f_elec = xtb.get_fukui("electrophilicity")
    f_nuc = xtb.get_fukui("nucleophilicity")
    return {'Fukui electrophilicity_Cu': f_elec[cu_id], 'Fukui nucleophilicity_Cu': f_nuc[cu_id],
            'Fukui electrophilicity_Br': f_elec[br_id], 'Fukui nucleophilicity_Br': f_nuc[br_id],
            'Fukui electrophilicity_C_phen': f_elec[phenyl_c_id], 'Fukui nucleophilicity_C_phen': f_nuc[phenyl_c_id]}


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

def identify_phenyl_carbon(xyz_path, cu_id):
    '''Identify the phenyl carbon atom in the complex'''
    mol = XYZ.from_file(xyz_path).molecule
    cu_coords = mol.sites[cu_id].coords
    neighbors = []
    for i, site in enumerate(mol.sites):
        if i != cu_id:  # Skip self
            if site.species_string == 'C':
                dist = np.linalg.norm(cu_coords - site.coords)
                neighbors.append((i, site.species_string, dist))

    # Sort by distance and get closest
    neighbors.sort(key=lambda x: x[2])
    closest = neighbors[:1]
    return closest[0][0]

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
    mulliken_charge_br = properties_dict['Mulliken_Population_Analysis']['ATOMICCHARGES'][br_index][0]
    mulliken_charge_cu = properties_dict['Mulliken_Population_Analysis']['ATOMICCHARGES'][cu_index][0]
    return {'Mayer Bond Order Cu-Br': mayer_bo_ord, 'Muliken Charge Br': mulliken_charge_br,
            'Mulliken Charge Cu': mulliken_charge_cu}


def extract_last_npa_table(file_path):
    """
    Extract the last Natural Population Analysis table from a Gaussian or ORCA output file
    and convert it to a pandas DataFrame.
    """
    # Read the entire file into memory
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find all indices where NPA tables start
    start_indices = [i for i, line in enumerate(lines) if "Summary of Natural Population Analysis:" in line]

    # If none found, return empty DataFrame
    if not start_indices:
        return pd.DataFrame()

    # Start from the last occurrence
    start = start_indices[-1]
    table_lines = []
    table_found = False
    table_ended = False

    for line in lines[start + 1:]:
        if "Natural Population" in line or ("Natural" in line and "Core" in line) or "---" in line:
            continue
        if "====" in line:
            break
        if "* Total *" in line:
            continue
        if line.strip():  # skip empty lines
            table_lines.append(line.strip())

    if not table_lines:
        return pd.DataFrame()

    # Parse the table lines into a list of dictionaries
    data = []
    for line in table_lines:
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

    return pd.DataFrame(data)

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

def get_nbo_energies(output_file, xyz_file, cu_id, phenyl_c_id):
    '''Extract NBO energies from ORCA output'''

    npa_df = extract_last_npa_table(output_file)

    coord1, coord2 = find_coordinating_atom_ids(cu_id, xyz_file)
    cu_natural_charge = npa_df[npa_df['Atom'] == 'Cu']['Natural Charge'].values[0]
    br_natural_charge = npa_df[npa_df['Atom'] == 'Br']['Natural Charge'].values[0]
    c_natural_charge = npa_df[npa_df['No'] == phenyl_c_id+1]['Natural Charge'].values[0]
    coord1_natural_charge = npa_df[npa_df['No'] == coord1+1]['Natural Charge'].values[0]
    coord2_natural_charge = npa_df[npa_df['No'] == coord2+1]['Natural Charge'].values[0]
    return {'Cu Natural Charge': cu_natural_charge, 'Br Natural Charge': br_natural_charge,
            'C_Phenyl Natural Charge': c_natural_charge, 'Coord1_Natural_Charge': coord1_natural_charge,
            'Coord2_Natural_Charge': coord2_natural_charge}

def extract_hirshfeld_table(file_path):
    """
    Extract the Hirshfeld table from a Gaussian output file and return as a pandas DataFrame.

    Parameters:
        file_path (str): Path to the Gaussian output file.

    Returns:
        pd.DataFrame: DataFrame containing atom index, element, and associated data.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the line with the table header
    start_idx = None
    for i, line in enumerate(lines):
        if "Hirshfeld charges, spin densities, dipoles, and CM5 charges" in line:
            start_idx = i + 2  # Table data starts 2 lines below
            break

    if start_idx is None:
        raise ValueError("Hirshfeld table not found in the file.")

    # Read table lines until "Tot" row or empty line
    table_data = []
    for line in lines[start_idx:]:
        if re.match(r'\s*Tot', line) or not line.strip():
            break
        parts = line.strip().split()
        atom_index = int(parts[0])
        element = parts[1]
        values = list(map(float, parts[2:]))
        table_data.append([atom_index, element] + values)

    # Define column names
    columns = ['Atom', 'Element', 'Q-H', 'S-H', 'Dx', 'Dy', 'Dz', 'Q-CM5']

    df = pd.DataFrame(table_data, columns=columns)
    return df

def get_hirshfeld_cm5_charges(output_file, xyz_file, cu_id, phenyl_c_id):
    """
    Extract Hirshfeld and CM5 charges for selected atoms from a Gaussian output.

    Parameters:
        output_file (str): Path to the Gaussian output file.
        xyz_file (str): Path to the XYZ structure file.
        cu_id (int): Index of Cu atom (0-based).
        phenyl_c_id (int): Index of the carbon atom on the phenyl ring (0-based).

    Returns:
        dict: Charges for Cu, Br, C_Phenyl, Coord1, and Coord2 atoms.
    """
    # Load Hirshfeld table
    df = extract_hirshfeld_table(output_file)

    # Use your existing coordination logic
    coord1, coord2 = find_coordinating_atom_ids(cu_id, xyz_file)

    # Extract charges using 1-based atom indices
    cu_hirshfeld = df[df['Element'] == 'Cu']['Q-H'].values[0]
    cu_cm5 = df[df['Element'] == 'Cu']['Q-CM5'].values[0]

    br_hirshfeld = df[df['Element'] == 'Br']['Q-H'].values[0]
    br_cm5 = df[df['Element'] == 'Br']['Q-CM5'].values[0]

    c_hirshfeld = df[df['Atom'] == phenyl_c_id + 1]['Q-H'].values[0]
    c_cm5 = df[df['Atom'] == phenyl_c_id + 1]['Q-CM5'].values[0]

    coord1_hirshfeld = df[df['Atom'] == coord1 + 1]['Q-H'].values[0]
    coord2_hirshfeld = df[df['Atom'] == coord2 + 1]['Q-H'].values[0]
    coord1_cm5 = df[df['Atom'] == coord1 + 1]['Q-CM5'].values[0]
    coord2_cm5 = df[df['Atom'] == coord2 + 1]['Q-CM5'].values[0]

    return {
        'Cu Hirshfeld Charge': cu_hirshfeld, 'Cu CM5 Charge': cu_cm5,
        'Br Hirshfeld Charge': br_hirshfeld, 'Br CM5 Charge': br_cm5,
        'C_Phenyl Hirshfeld Charge': c_hirshfeld, 'C_Phenyl CM5 Charge': c_cm5,
        'Coord1 Hirshfeld Charge': coord1_hirshfeld, 'Coord2 Hirshfeld Charge': coord2_hirshfeld,
        'Coord1 CM5 Charge': coord1_cm5, 'Coord2 CM5 Charge': coord2_cm5
    }

def calculate_laplacian_bond_order(molden_filepath, xyz_filepath, cu_id, br_id, phenyl_c_id, threads=6):
    """Calculate the Laplacian Bond Order (LBO) for Cu-C, Cu-Br, and Cu-ligand bonds."""

    # Get relevant indexes
    xyz_path = Path(xyz_filepath)
    mol = XYZ.from_file(xyz_path).molecule
    natoms = len(mol)

    # Prepare environment and stack limit
    def set_stack_unlimited():
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    env = os.environ.copy()
    env['OMP_STACKSIZE'] = '200M'

    # Construct Multiwfn command
    multiwfn_input = '9\n8\ny\n0\nq\nq\n'

    proc = subprocess.run(
        [MULTIWFN_PATH, molden_filepath, '-nt', str(threads)],
        input=multiwfn_input,
        text=True,
        env=env,
        preexec_fn=set_stack_unlimited,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Multiwfn failed:\n{proc.stderr}")

    multiwfn_out = proc.stdout.splitlines()

    # Check if file exists
    if not os.path.exists("bndmat.txt"):
        raise FileNotFoundError("The bndmat.txt file was not created. Check the Multiwfn output.")

    # Parse bond order matrix
    with open("bndmat.txt", "r") as f:
        lines = f.readlines()

    lbo_mat = np.zeros((natoms, natoms))
    row = 0
    col = 0

    for line in lines:
        floats = extract_floats(line)
        if not floats:
            continue
        lbo_mat[row, col:col + len(floats)] = floats
        row += 1
        if row == natoms:
            col += len(floats)
            row = 0

    if not np.allclose(lbo_mat, lbo_mat.T):
        raise ValueError("The LBO matrix is not symmetric. Error when parsing matrix.")

    cu_br_lbo = lbo_mat[cu_id, br_id]
    cu_c_lbo = lbo_mat[cu_id, phenyl_c_id]
    cu_x_lbo = lbo_mat[cu_id, :]

    top_indexes = np.argsort(cu_x_lbo)[-5:]
    largest_bond_orders_dict = {i: cu_x_lbo[i] for i in top_indexes}

    for key in [phenyl_c_id, br_id, cu_id]:
        largest_bond_orders_dict.pop(key, None)

    coordinating_atoms_n_o = all(
        mol.atomic_numbers[i] in [7, 8] for i in largest_bond_orders_dict
    )

    if not coordinating_atoms_n_o:
        raise ValueError("The largest bond orders do not correspond to N or O atoms. Check input structure.")

    total_ligand_lbo = sum(largest_bond_orders_dict.values())
    average_ligand_lbo = np.mean(list(largest_bond_orders_dict.values()))

    os.remove("bndmat.txt")

    return {
        "Cu-C LBO": cu_c_lbo,
        "Cu-Br LBO": cu_br_lbo,
        "Total Cu-ligand LBO": total_ligand_lbo,
        "Average Cu-ligand LBO": average_ligand_lbo,
        #"Largest Bond Orders": [largest_bond_orders_dict]
    }

def calculate_e_ox_red(freq_base, sp_base, freq_ox, sp_ox, freq_red,sp_red):
    """Calculate the E_ox and E_red descriptors"""
    e_comp_base = read_energy(freq_base, sp_base)
    e_comp_ox = read_energy(freq_ox, sp_ox)
    e_comp_red = read_energy(freq_red, sp_red)
    e_ox = e_comp_ox - e_comp_base
    e_red = e_comp_red - e_comp_base
    return {"E_ox": e_ox, "E_red": e_red}



if __name__ == '__main__':
    jobs_dir = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")
    species = "CuOA"
    for ligand_dir in jobs_dir.iterdir():
        if ligand_dir.is_dir() and ligand_dir.name == sys.argv[1]:
            ligand_name = ligand_dir.name  # ligand_name of the ligand
            sp_path = ligand_dir.joinpath("Single_Point",
                                               f"{ligand_name}_{species}_Single_Point")
            opt_path = ligand_dir.joinpath("Ensemble_Opt", f"{ligand_name}_{species}_Ensemble_Opt")
            ox_path = ligand_dir.joinpath("E_ox", f"{ligand_name}_{species}_E_ox")
            red_path = ligand_dir.joinpath("E_red", f"{ligand_name}_{species}_E_red")

            elec_desc = {}
            for conformer_dir in sp_path.glob('conformer_*'):
                opt_conf_dir = opt_path.joinpath(conformer_dir.name)
                opt_json = opt_conf_dir.joinpath("output", 'input.property.json')
                sp_json = conformer_dir.joinpath("output", 'input.property.json')
                energy = read_energy(opt_json, sp_json)
                conf_no = conformer_dir.name.split('_')[-1]
                input_path = conformer_dir.joinpath("output", 'input.inp')
                xyz_path = conformer_dir.joinpath("output", 'input.xyz')
                txt_path = conformer_dir.joinpath("output", f'output_{ligand_name}_{species}_Single_Point_conf{conf_no}.out')
                nbo_output = ligand_dir.joinpath('NBO', f"{ligand_name}_{species}_NBO", f"conformer_{conf_no}", 'output', 'input.log')

                json_path = conformer_dir.joinpath("output", 'input.property.json')
                orca_dict = json.loads(json_path.read_text())
                property_dict = orca_dict[f'Geometry_{len(orca_dict)-1}']
                charge = get_charge(input_path)

                cu_id = find_atom_id(xyz_path, 'Cu')
                br_id = find_atom_id(xyz_path, 'Br')
                phenyl_c_id = identify_phenyl_carbon(xyz_path, cu_id)

                orb_desc = get_front_orb_derived_descriptors(txt_path)
                fukui_desc = calc_fukui_descriptors(xyz_path, charge, cu_id, br_id, phenyl_c_id)
                mol_desc = get_mol_desc(property_dict)
                charge_desc = get_charge_desc(property_dict)
                nbo_desc = get_nbo_energies(nbo_output, xyz_path, cu_id, phenyl_c_id)
                hirsh_cm5_desc = get_hirshfeld_cm5_charges(nbo_output, xyz_path, cu_id, phenyl_c_id)
                lbo_desc = calculate_laplacian_bond_order(conformer_dir.joinpath('output', 'input.molden.input'),
                                                conformer_dir.joinpath('output', 'input.xyz'), cu_id, br_id, phenyl_c_id)

                ox_json_freq = ox_path.joinpath(conformer_dir.name, 'output', 'input.property.json')
                ox_json_sp = ox_path.joinpath(conformer_dir.name, 'output', 'input_job2.property.json')
                red_json_freq = red_path.joinpath(conformer_dir.name, 'output', 'input.property.json')
                red_json_sp = red_path.joinpath(conformer_dir.name, 'output', 'input_job2.property.json')
                e_ox_red_desc = calculate_e_ox_red(opt_json, sp_json, ox_json_freq, ox_json_sp, red_json_freq, red_json_sp)

                electronics = {**orb_desc, **fukui_desc, **mol_desc, **charge_desc, **nbo_desc, **hirsh_cm5_desc, **lbo_desc, **e_ox_red_desc}
                electronics['Energy'] = energy
                elec_desc[conformer_dir.name] = electronics


            desc_df = pd.DataFrame(elec_desc).T  # Convert dictionary to DataFrame

            boltz_avg, weights = calculate_boltzmann_average(desc_df)  # Calculate Boltzmann average
            min_values = desc_df.min()  # Get minimum values
            max_values = desc_df.max()  # Get maximum values

            stat = pd.concat([min_values, max_values, boltz_avg], axis=1,
                             keys=['Min', 'Max', 'Boltzmann Average'])  # Concat averages, min and max
            desc_df = pd.concat([desc_df, stat.T], axis=0)

            # Save DataFrame to CSV
            desc_dir = jobs_dir.joinpath(ligand_name, 'electronic_descriptors')
            Path.mkdir(desc_dir, exist_ok=True)
            desc_df.to_csv(desc_dir.joinpath(f"{ligand_name}_{species}_electronic_descriptors.csv"))

