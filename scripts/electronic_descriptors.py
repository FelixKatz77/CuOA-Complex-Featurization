from morfeus import read_xyz
from morfeus import XTB
import pandas as pd
import json
from pathlib import Path
import re

jobs_path = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/")

def get_charge(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            if "*xyzfile" in line:
                charge = int(line.split(' ')[1])
                return charge
    return None

def parse_orbital_energies(filepath):
    """
    Parse orbital energies section from an ORCA output file.

    Parameters:
    filepath (str): Path to the ORCA output file

    Returns:
    pd.DataFrame: DataFrame containing orbital energies data
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

if __name__ == '__main__':
    name = "L1"
    species = "CuBr"
    job_type = "Ensemble_Opt"
    ensemble_path = jobs_path.joinpath(name, job_type, f"{name}_{species}_{job_type}")
    steric_descriptors = {}
    for conformer_dir in ensemble_path.glob('conformer_*'):
        conf_no = conformer_dir.name.split('_')[-1]
        input_path = conformer_dir.joinpath("output", 'input.inp')
        txt_path = conformer_dir.joinpath("output", f'output_{name}_{species}_{job_type}_conf{conf_no}.out')
        json_path = conformer_dir.joinpath("output", 'input.property.json')
        orca_dict = json.loads(json_path.read_text())
        property_dict = orca_dict[f'Geometry_{len(orca_dict)-1}']
        charge = get_charge(input_path)
        orb_desc = get_front_orb_derived_descriptors(txt_path)
        fukui_desc = calc_fukui_descriptors(conformer_dir.joinpath('output', 'input.xyz'), charge=charge)
        mol_desc = get_mol_desc(property_dict)
        charge_desc = get_charge_desc(property_dict)
        elec_desc = {**orb_desc, **fukui_desc, **mol_desc, **charge_desc}
        desc_df = pd.DataFrame(elec_desc).T
        pass

