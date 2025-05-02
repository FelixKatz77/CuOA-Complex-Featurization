from pathlib import Path
from rdkit import Chem
import json
import pandas as pd
import subprocess
import re

def create_dirs(name, job_type, species, jobs_path, output=True):

    jobs_path = Path(jobs_path)
    job_name = f"{name}_{species}_{job_type}"
    ligand_directory = jobs_path.joinpath(name)
    job_directory = ligand_directory.joinpath(job_type, job_name)
    out_directory = job_directory.joinpath("output")

    if not ligand_directory.exists():
        Path.mkdir(ligand_directory, exist_ok=False)
    else:
        if job_directory.exists():
            print(f"Directory {job_directory} already exists.")
    if not job_directory.exists():
        Path.mkdir(job_directory, exist_ok=False, parents=True)
        if output:
            Path.mkdir(out_directory, exist_ok=False)
    else:
        print(f"Directory {job_directory} already exists.")
    return job_directory

def write_orca_input(dir, job_name, **kwargs):
    """Write ORCA input file for geometry optimization"""

    scf_type = kwargs.get('scf_type', '')
    method = kwargs.get('method', '')
    functional = kwargs.get('functional', '')
    basis_set = kwargs.get('basis_set', '')
    basis_options = kwargs.get('basis_options', '')
    dispersoin_correction = kwargs.get('dispersion_correction', '')
    runtype = kwargs.get('runtype', '')
    algo = kwargs.get('options', '')
    options = kwargs.get('miscellaneous', '')
    charge = kwargs.get('charge', '0')
    multiplicity = kwargs.get('multiplicity', '1')
    elprop = kwargs.get('elprop', '')
    nbo = kwargs.get('nbo', '')
    memory = kwargs.get('memory', '')
    n_procs = kwargs.get('n_procs', '12')
    single_point = kwargs.get('single_point', False)
    sp_basis_set = kwargs.get('sp_basis_set', basis_set)

    orca_input = f'! {scf_type} {method} {functional} {basis_set} {dispersoin_correction} {runtype} {algo} {options} {nbo}'
    if basis_options:
        orca_input += f'\n\n%basis\n  {basis_options} end\nend'
    if elprop:
        elprop = f'\n  '.join(elprop.split(','))
        orca_input += f'\n\n%elprop\n  {elprop}\nend'  # write elprop section
    orca_input += f'\n\n*xyzfile {charge} {multiplicity} /tmp/{job_name}/input.xyz'
    if memory:
        orca_input += f'\n\n%maxcore {memory}'
    orca_input += f'\n\n%pal nprocs {n_procs} end'
    orca_input += f'\n\n%Method\n  WriteJSONPropertyfile True\nend' # Write JSON file with properties
    if single_point:
        orca_input += '\n\n$new_job'
        orca_input += f'\n\n! {scf_type} {method} {functional} {sp_basis_set} {dispersoin_correction} {algo} {options}'
        orca_input += f'\n\n*xyzfile {charge} {multiplicity} /tmp/{job_name}/input.xyz'
    if memory:
        orca_input += f'\n\n%maxcore {memory}'
    orca_input += f'\n\n%pal nprocs {n_procs} end'
    orca_input += f'\n\n%Method\n  WriteJSONPropertyfile True\nend'

    with open(dir.joinpath("input.inp"), "w") as f:
        f.write(orca_input)

def write_batch_file(job_name, job_directory, xyz_path, memory='4GB', tasks_per_node=12):
    batch_file = Path('submit.batch').read_text()
    batch_file = batch_file.replace('*job_name*', str(job_name))
    batch_file = batch_file.replace('*job_directory*', str(job_directory))
    batch_file = batch_file.replace('*xyz_path*', str(xyz_path))
    batch_file = batch_file.replace('*memory*', memory)
    batch_file = batch_file.replace('*tasks_per_node*', str(tasks_per_node))
    Path(job_directory.joinpath("submit.batch")).write_text(batch_file)

def is_transition_metal(at):

    '''check if an atom is a transition metal'''

    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)

def set_dative_bonds(mol, fromAtoms=(7,8)):

    """ convert some bonds to dative

    Replaces some single bonds between metals and atoms with atomic numbers in fromAtoms
    with dative bonds. The replacement is only done if the atom has "too many" bonds.

    Returns the modified molecule."""

    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if nbr.GetAtomicNum() in fromAtoms and \
               nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
               rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
    return rwmol

def hartree_to_kcal_mol(hartree):
    """Convert energy from Hartree to kcal/mol"""
    return hartree * 627.509  # Standard conversion factor

def write_xyz_file(path, energy_kcal, atoms, coords):
    """Write XYZ file with energy in comment line"""
    path.write_text(
        f"{len(atoms)}\n"
        f"Energy (kcal/mol) = {energy_kcal:.2f}\n"
        + "\n".join(
            f"{atom:2s} {x:15.10f} {y:15.10f} {z:15.10f}"
            for atom, (x, y, z) in zip(atoms, coords)
        )
    )

def find_atom_id(xyz_path, element):
    '''Find the atom IDs of two elements in an XYZ file'''

    input_text = xyz_path.read_text()
    lines = input_text.strip().split('\n')
    atom_id = None
    if element not in input_text:
        raise ValueError(f"Atom {element} not found in {xyz_path}")
    # Skip the first two lines (header)
    for i, line in enumerate(lines[2:], start=0):
        if line.strip().startswith(element):
            atom1_id = i

    return atom1_id

def read_energy(opt_out, single_out):
    '''Read the energy from an ORCA output file and add thermal corrections'''

    opt_output = json.loads(opt_out.read_text())
    single_output = json.loads(single_out.read_text())
    for geometry in opt_output.keys():
        if 'THERMOCHEMISTRY_Energies' in opt_output[geometry]:
            gibb_e = opt_output[geometry]['THERMOCHEMISTRY_Energies']['FREEENERGYG']
            eel = opt_output[geometry]['THERMOCHEMISTRY_Energies']['ELENERGY']
            thermal_corr_gibbs = gibb_e - eel
    final_eel = single_output['Geometry_1']['DFT_Energy']['FINALEN']
    return hartree_to_kcal_mol(final_eel + thermal_corr_gibbs)


def check_neg_frequencies(filename):
    """
    Reads vibrational frequency data from an output file and checks for negative frequencies.
    """
    frequencies = {}
    reading_freqs = False

    with open(filename, 'r') as file:
        for line in file:
            # Check if we've reached the frequency section
            if '-----------------------' in line and 'VIBRATIONAL FREQUENCIES' in file.readline():
                reading_freqs = True
                # Skip the next line (dashed line)
                next(file)
                # Skip the scaling factor line
                next(file)
                continue

            # If we're in the frequency section and line has expected format
            if reading_freqs and ':' in line and 'cm**-1' in line:
                try:
                    # Parse line to extract index and frequency value
                    parts = line.strip().split(':')
                    index = int(parts[0])
                    freq_value = float(parts[1].split('cm**-1')[0].strip())

                    frequencies[index] = freq_value


                except (ValueError, IndexError):
                    # If line doesn't match expected format, we might be out of the frequencies section
                    reading_freqs = False

    freq_df = pd.DataFrame.from_dict(frequencies, orient='index', columns=['Frequency (cm^-1)'])
    negative_freqs = freq_df[freq_df['Frequency (cm^-1)'] < 0]
    has_negative_freqs = not negative_freqs.empty

    return has_negative_freqs

def check_orca_success(filename):
    """
    Reads the ORCA output file to check if the job was successful.
    """

    with open(filename, 'r') as file:
        content = file.read()
        if 'ORCA TERMINATED NORMALLY' in content:
            return True
        else:
            return False

def check_optimization_success(filename):
    """
    Reads the ORCA output file to check if the optimization was successful.
    """

    with open(filename, 'r') as file:
        content = file.read()
        if 'The optimization did not converge' in content:
            return False
        else:
            return True
        
def run_command(cmd, cwd = None, shell = True):
    """
    Run a command in the shell and return the output.
    """
    result = subprocess.run(
        cmd,
        shell=shell,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr

def extract_floats(line):
    """In a line of text, extract all numbers that include a decimal"""
    float_pattern = r"[-+]?\d*\.\d+"
    matches = re.findall(float_pattern, line)
    return [float(match) for match in matches]
