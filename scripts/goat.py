from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
from morfeus import read_xyz
from morfeus import XTB
from pathlib import Path


jobs_path = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/")

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

name = "L1" # name of the ligand
species = "CuBr" # species
job_type = "GOAT" # job type
job_name = f"{name}_{species}_{job_type}"
smiles = 'Br[Cu]1[N]2=CC=CC3=C2C([N]1=CC=C4)=C4C=C3'

ligand_directory = jobs_path.joinpath(name)
Path.mkdir(ligand_directory, exist_ok=False) # create the ligand directory if it doesn't exist

# create the job directory if it doesn't exist
job_directory = ligand_directory.joinpath(job_type, job_name)
Path.mkdir(job_directory, exist_ok=False, parents=True)

Path.mkdir(job_directory.joinpath("output"), exist_ok=False) # create the output directory if it doesn't exist

m = Chem.MolFromSmiles(smiles, sanitize=False) # create a molecule from the SMILES string
m = set_dative_bonds(m) # set dative bonds
m = Chem.AddHs(m)  # add hydrogens

# write the molecule to an XYZ file
AllChem.EmbedMolecule(m)
xyz_path = job_directory.joinpath(f'{name}_rdkit.xyz')
Chem.MolToXYZFile(m, xyz_path)

goat_input = f"!XTB GOAT #XTB version 6.4.0\n*xyzfile 0 1 /tmp/{job_name}/input.xyz\n%pal nprocs 12 end" # create the GOAT input

with open(job_directory.joinpath("input.inp"), "w") as f: # write the GOAT input to a file
    f.write(goat_input)

batch_file = Path('submit.batch').read_text() # read the batch file template

# replace the placeholders in the batch file template
batch_file = batch_file.replace('*job_name*', job_name)
batch_file = batch_file.replace('*job_directory*', str(job_directory))
batch_file = batch_file.replace('*xyz_path*', str(xyz_path))

Path(job_directory.joinpath("submit.batch")).write_text(batch_file) # write the batch file




pass