'''
This script is used to generate the input files for the GOAT calculations.
'''

import shutil
import pandas as pd
from pathlib import Path

from utils import create_dirs, write_orca_input, write_batch_file

jobs_path = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")

ligands_info = pd.read_csv("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/Ligands.csv")
for xyz_file in Path("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/geometries/input/").glob("*.xyz"):
    ligand_name = xyz_file.stem
    ligand_info = ligands_info[ligands_info["name"] == ligand_name]
    charge = ligand_info["charge"].values[0]

    job_directory = create_dirs(ligand_name, "GOAT", "CuOA", jobs_path=jobs_path)
    job_name = job_directory.stem

    xyz_path =  job_directory.joinpath(f"{ligand_name}.xyz")
    shutil.copy(xyz_file, xyz_path)

    write_orca_input(job_directory, job_directory.stem,
                     method="XTB GOAT",
                     charge=charge,
                     multiplicity=2)

    write_batch_file(job_name, job_directory, xyz_path, memory="8GB", tasks_per_node=6)