import shutil
from pathlib import Path

import pandas as pd

from utils import create_dirs, dir_empty, write_orca_input, write_batch_file, check_optimization_status

if __name__ == "__main__":

    ligands_info = pd.read_csv("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/Ligands.csv")
    jobs_dir = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")
    template_dir = Path("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/geometries/input/")
    for ligand_dir in jobs_dir.iterdir():
        ligand_name = ligand_dir.name
        ligand_info = ligands_info[ligands_info["name"] == ligand_name]
        charge = ligand_info["charge_complex"].values[0]
        multiplicity = ligand_info["multiplicity_complex"].values[0]
        ensemble_dir = ligand_dir.joinpath("Ensemble_Opt", f"{ligand_name}_CuOA_Ensemble_Opt")
        for conf_dir in ensemble_dir.glob('conformer_*'):
            num = int(conf_dir.name.split("_")[-1])
            opt_xyz_file = conf_dir.joinpath("output", 'input.xyz')
            temp_xyz_file = template_dir.joinpath(f"{ligand_name}.xyz")
            orca_output = conf_dir.joinpath("output",
                                             f"output_{ligand_name}_CuOA_Ensemble_Opt_conf{num}.out")
            if check_optimization_status(orca_output, opt_xyz_file, temp_xyz_file):
                lowest_conf_dir = conf_dir
                break
            else:
                continue

        ox_dir = create_dirs(ligand_name, "CuOA", "E_ox", jobs_dir, output=True)
        red_dir = create_dirs(ligand_name, "CuOA", "E_red", jobs_dir, output=True)
        xyz_path_conf = lowest_conf_dir.joinpath('output', 'input.xyz')
        job_name_ox = f"{ligand_name}_CuOA_Eox"
        job_name_red = f"{ligand_name}_CuOA_Ered"
        if ox_dir.joinpath("output").exists() and not dir_empty(ox_dir.joinpath("output")):
            raise FileExistsError
        else:
            # write orca input and batch file for oxidation
            # copy xyz file to oxidation directory
            xyz_path_ox = ox_dir.joinpath(f"{ligand_name}_CuOA_Eox_input.xyz")
            shutil.copy(xyz_path_conf, xyz_path_ox)
            write_orca_input(ox_dir, job_name_ox, scf_type="RKS", functional="TPSS", basis_set="def2-TZVP",
                             dispersion_correction="D3BJ", options="SARC/J RI", miscellaneous="KeepDens TightSCF",
                             charge=charge + 1,
                             multiplicity=1, memory="1000")
            write_batch_file(job_name_ox, ox_dir, xyz_path_ox, memory="16GB", tasks_per_node=12)
            print(f"Written E_oxidation input for {ligand_name} in {ox_dir}")

        if red_dir.joinpath("output").exists() and not dir_empty(red_dir.joinpath("output")):
            raise FileExistsError
        else:
            # write orca input and batch file for reduction
            # copy xyz file to reduction directory
            xyz_path_red = red_dir.joinpath(f"{ligand_name}_CuOA_Ered_input.xyz")
            shutil.copy(xyz_path_conf, xyz_path_red)
            write_orca_input(red_dir, job_name_red, scf_type="RKS", functional="TPSS", basis_set="def2-TZVP",
                             dispersion_correction="D3BJ", options="SARC/J RI", miscellaneous="KeepDens TightSCF",
                             charge=charge - 1,
                             multiplicity=1, memory="1000")
            write_batch_file(job_name_red, red_dir, xyz_path_red, memory="16GB", tasks_per_node=12)
            print(f"Written E_reduction input for {ligand_name} in {red_dir}")