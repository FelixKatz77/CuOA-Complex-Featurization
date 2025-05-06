from pathlib import Path

import pandas as pd
from sympy.physics.mechanics.tests.test_system import coordinates

from utils import create_dirs, write_batch_file, dir_empty, write_gaussian_input, check_optimization_status

if __name__ == '__main__':
    jobs_dir = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")
    template_dir = Path("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/geometries/input/")
    ligands_info = pd.read_csv("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/Ligands.csv")
    species = "CuOA"
    for ligand_dir in jobs_dir.iterdir():
        if ligand_dir.is_dir():
            ligand_name = ligand_dir.name  # ligand_name of the ligand
        else:
            continue
        ligand_info = ligands_info[ligands_info["name"] == ligand_name]
        charge = ligands_info["charge_complex"].values[0]
        ensemble_path = jobs_dir.joinpath(ligand_name, "Ensemble_Opt", f"{ligand_name}_{species}_Ensemble_Opt")  # Path to ensemble directory

        nbo_dir = create_dirs(ligand_name, "NBO", "CuOA", jobs_dir, output=False)

        # Loop over all conformers in the ensemble
        for conf_opt_dir in ensemble_path.glob('conformer_*'):
            conf_name = conf_opt_dir.name
            num = int(conf_name.split("_")[-1])
            job_name = f"{ligand_name}_{conf_name}_NBO"
            conf_nbo_dir = nbo_dir.joinpath(conf_name)
            try:
                if conf_nbo_dir.joinpath("output").exists() and not dir_empty(conf_nbo_dir.joinpath("output")):
                    raise FileExistsError
                else:
                    orca_output = conf_opt_dir.joinpath("output",
                                                         f"output_{ligand_name}_CuOA_Ensemble_Opt_conf{num}.out")
                    opt_xyz_file =conf_opt_dir.joinpath("output", 'input.xyz')
                    temp_xyz_file = template_dir.joinpath(f"{ligand_name}.xyz")
                    try:
                        if check_optimization_status(orca_output, opt_xyz_file, temp_xyz_file):
                            Path.mkdir(conf_nbo_dir, exist_ok=True)
                            write_gaussian_input(conf_nbo_dir, job_name, opt_xyz_file, charge=charge)
                            write_batch_file(job_name, conf_nbo_dir, opt_xyz_file, memory="16GB", tasks_per_node=12, template="gaussian.batch")

                            # create output directory
                            Path.mkdir(conf_nbo_dir.joinpath("output"), exist_ok=True)
                            print(f"Written NBO input for {ligand_name} {conf_name} in {conf_nbo_dir}")
                        else:
                            print(f'Optimization not successful for {ligand_name}_{conf_nbo_dir}.')
                            continue
                    except FileNotFoundError:
                        print(f"File {orca_output} not found.")

            except FileExistsError:
                print(f"{conf_name} already exists in {conf_nbo_dir}")
                continue

