import shutil
from pathlib import Path
import pandas as pd
from sphinxcontrib.htmlhelp import template_dir

from fod_analysis import xyz_path
from utils import create_dirs, write_orca_input, write_batch_file, dir_empty, check_optimization_status


def main():
    n_jobs = 0
    species = 'CuOA'
    ligands_info = pd.read_csv("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/Ligands.csv")
    template_dir = Path("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/geometries/input/")
    jobs_dir = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")

    for ligand_dir in jobs_dir.iterdir():
        if not ligand_dir.is_dir():
            continue

        ligand_name = ligand_dir.name
        ligand_info = ligands_info[ligands_info["name"] == ligand_name]
        charge = ligand_info[f"charge_complex"].values[0]
        multiplicity = ligand_info[f"multiplicity_complex"].values[0]

        ensemble_path = jobs_dir.joinpath(ligand_name, "Ensemble_Opt", f"{ligand_name}_{species}_Ensemble_Opt")
        sp_dir = create_dirs(ligand_name, "Single_Point", species, jobs_dir, output=False)

        for opt_conf_dir in ensemble_path.glob('conformer_*'):
            conf_name = opt_conf_dir.name
            num = int(conf_name.split("_")[-1])
            job_name = f"{ligand_name}_{conf_name}_Single_Point"
            conf_dir = sp_dir.joinpath(conf_name)
            orca_output = opt_conf_dir.joinpath("output",
                                                 f"output_{ligand_name}_CuOA_Ensemble_Opt_conf{num}.out")



            try:
                if conf_dir.joinpath("output").exists() and not dir_empty(conf_dir.joinpath("output")):
                    raise FileExistsError
                else:
                    opt_xyz_file = opt_conf_dir.joinpath("output", 'input.xyz')
                    temp_xyz_file = template_dir.joinpath(f"{ligand_name}.xyz")
                    if check_optimization_status(orca_output, opt_xyz_file, temp_xyz_file):
                        Path.mkdir(conf_dir, exist_ok=True)

                        new_xyz_path = conf_dir.joinpath(f"conf{num}_input.xyz")
                        shutil.copy(opt_xyz_file, new_xyz_path)

                        write_orca_input(conf_dir, job_name, scf_type="UKS", functional="TPSS", basis_set="def2-TZVP",
                                         dispersion_correction="D3BJ",
                                         options="SARC/J RI", miscellaneous="KeepDens TightSCF",
                                         charge=charge, multiplicity=multiplicity,
                                         elprop="Quadrupole True,Polar 1",
                                         memory="1000")

                        write_batch_file(job_name, conf_dir, new_xyz_path, memory="16GB", tasks_per_node=12)
                        Path.mkdir(conf_dir.joinpath("output"), exist_ok=True)

                        print(f"Written single-point job for {ligand_name} to {conf_dir}")
                        n_jobs += 1

            except FileExistsError:
                print(f"Single-point job already exists in {conf_dir}")
                continue
            except FileNotFoundError:
                print(f"File {orca_output} not found.")

    print('Generated', n_jobs, 'single-point jobs.')

if __name__ == "__main__":
    main()
