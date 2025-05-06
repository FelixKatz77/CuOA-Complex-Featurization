import contextlib
import os
import sys
from pathlib import Path
from tabnanny import verbose

import networkx as nx
import numpy as np
from rdkit import Chem
import json
import pandas as pd
from pymatgen.io.xyz import XYZ
from openbabel import pybel
from openbabel import openbabel

def dir_empty(dir_path):
    path = Path(dir_path)
    has_next = next(path.iterdir(), None)
    if has_next is None:
        return True
    return False

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
    goat_options = kwargs.get('goat_options', '')

    orca_input = f'! {scf_type} {method} {functional} {basis_set} {dispersoin_correction} {runtype} {algo} {options} {nbo}'
    if basis_options:
        orca_input += f'\n\n%basis\n  {basis_options} end\nend'
    if goat_options:
        orca_input += f'\n\n%GOAT\n  {goat_options}\nend'
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

def write_batch_file(job_name, job_directory, xyz_path, memory='4GB', tasks_per_node=12, template='submit.batch'):
    batch_file = Path(template).read_text()
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
    '''Find the atom ID for an element in an XYZ file'''

    input_text = xyz_path.read_text()
    lines = input_text.strip().split('\n')
    atom_id = None
    if element not in input_text:
        raise ValueError(f"Atom {element} not found in {xyz_path}")
    # Skip the first two lines (header)
    for i, line in enumerate(lines[2:], start=0):
        if line.strip().startswith(element):
            atom_id = i

    return atom_id

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


def check_neg_frequencies(filename, verbose=False):
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
    if verbose:
        if has_negative_freqs:
            print(f"Negative frequencies found in {filename}:")
    return has_negative_freqs

def check_orca_success(filename, verbose=False):
    """
    Reads the ORCA output file to check if the job was successful.
    """

    with open(filename, 'r') as file:
        content = file.read()
        if 'ORCA TERMINATED NORMALLY' in content:
            return True
        else:
            if verbose:
                print(f"ORCA job failed for {filename}.")
            return False

def check_optimization_success(filename, verbose=False):
    """
    Reads the ORCA output file to check if the optimization was successful.
    """

    with open(filename, 'r') as file:
        content = file.read()
        if 'The optimization did not converge' in content:
            return False
        else:
            if verbose:
                print(f"Optimization failed for {filename}.")
            return True

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
    # Sort by element type
    closest_two.sort(key=lambda x: x[1])
    ids = [neighbor[0] for neighbor in closest_two]
    return ids

@contextlib.contextmanager
def suppress_obabel_warnings():
    # Save the original stderr file descriptor
    old_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(old_stderr_fd)

    # Redirect stderr to /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, old_stderr_fd)
    os.close(devnull)

    try:
        yield
    finally:
        # Restore original stderr
        os.dup2(saved_stderr_fd, old_stderr_fd)
        os.close(saved_stderr_fd)

def molecule_to_graph(mol):
    G = nx.Graph()
    for atom in mol:
        G.add_node(atom.idx, element=atom.type)
    for atom in mol:
        ob_atom = atom.OBAtom
        for ob_bond in openbabel.OBAtomBondIter(ob_atom):
            begin_idx = ob_bond.GetBeginAtomIdx()
            end_idx = ob_bond.GetEndAtomIdx()
            G.add_edge(begin_idx, end_idx)
    return G

def check_connectivity(xyz_opt, xyz_temp, verbose=False):
    with suppress_obabel_warnings():
      mol_temp = next(pybel.readfile("xyz", str(xyz_temp)))
      mol_temp.OBMol.ConnectTheDots()
      mol_temp.OBMol.PerceiveBondOrders()

    with suppress_obabel_warnings():
      mol_opt = next(pybel.readfile("xyz", str(xyz_opt)))
      mol_opt.OBMol.ConnectTheDots()
      mol_opt.OBMol.PerceiveBondOrders()
    graph_temp = molecule_to_graph(mol_temp)
    graph_opt = molecule_to_graph(mol_opt)
    res = nx.is_isomorphic(graph_temp, graph_opt)
    # if not res:
    #     pos1 = nx.spring_layout(graph_temp, seed=42)  # Seed for reproducibility
    #     pos2 = nx.spring_layout(graph_opt, seed=42)
    #
    #     labels_temp = nx.get_node_attributes(graph_temp, 'element')
    #     labels_opt = nx.get_node_attributes(graph_opt, 'element')
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #
    #     # Plot G1
    #     nx.draw(graph_temp, pos1, labels=labels_temp, ax=axs[0], with_labels=True, node_color='skyblue', node_size=200, font_size=8)
    #     axs[0].set_title("Graph Input")
    #
    #     # Plot G2
    #     nx.draw(graph_opt, pos2, labels=labels_opt, ax=axs[1], with_labels=True, node_color='lightgreen', node_size=200, font_size=8)
    #     axs[1].set_title("Graph Opt")
    #
    #     plt.tight_layout()
    #     plt.show()
    if verbose:
        if not res:
            print(f"Connectivity check failed for {xyz_opt}.")
    return res

def extract_elements_and_coordinates(xyz_file):
    with open(xyz_file, 'r') as f:
        input_text = f.read()
    lines = input_text.strip().split('\n')

    # Identify element lines by whitespace at the beginning
    element_lines = []
    for line in lines:
        # Check if line starts with whitespace followed by an element
        if line.startswith('  '):  # Two spaces before elements in the input
            element_lines.append(line)

    # Extract elements and coordinates
    elements_and_coordinates = []
    for line in element_lines:
        # Strip leading whitespace and split by multiple spaces
        parts = line.strip().split()

        # First part is the element, next three are coordinates
        element = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

        elements_and_coordinates.append((element, x, y, z))

    formatted_lines = []
    for element, x, y, z in elements_and_coordinates:
        formatted_lines.append(f"{element}   {x:.14f}   {y:.14f}   {z:.14f}")

    return '\n'.join(formatted_lines)


def write_gaussian_input(dir, job_name, xyz_path, charge):
    """Write Gaussian input file for NBO analysis"""
    coordinates = extract_elements_and_coordinates(xyz_path)
    gaussian_input = f'%mem=10000MB\n%nprocshared=12\n#p TPSSTPSS/def2TZVP pop=nbo\n\n{job_name}\n\n{charge} 2\n'
    gaussian_input += f'{coordinates}\n\n\n'
    with open(dir.joinpath("input.gjf"), "w") as f:
        f.write(gaussian_input)

def check_optimization_status(orca_output_file, opt_xyz_file, temp_xyz_file, verbose=False):
    try:
        orca_success = check_orca_success(orca_output_file, verbose=verbose)
        opt_success = check_optimization_success(orca_output_file, verbose=verbose)
        neg_frequencies = check_neg_frequencies(orca_output_file, verbose=verbose)
    except FileNotFoundError:
        print(f"File {orca_output_file} not found.")
        return False
    try:
        connectivity = check_connectivity(opt_xyz_file, temp_xyz_file, verbose=verbose)
    except FileNotFoundError:
        print(f"File {opt_xyz_file} not found.")
        return False
    if orca_success and opt_success and not neg_frequencies and connectivity:
        return True
    else:
        print(f"Optimization status not successful for {orca_output_file}.")
        return False