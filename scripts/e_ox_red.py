import shutil
from pathlib import Path

import pandas as pd

from utils import create_dirs, dir_empty, write_orca_input, write_batch_file, check_optimization_status

if __name__ == "__main__":

    ligands_info = pd.read_csv("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/Ligands.csv")
    jobs_dir = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")
    template_dir = Path("/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/geometries/input/")
    for ligand_dir in jobs_dir.iterdir():
        if not ligand_dir.is_dir() or ligand_dir.name not in ["L31", "L32"]:
            continue
        ligand_name = ligand_dir.name
        ligand_info = ligands_info[ligands_info["name"] == ligand_name]
        charge = ligand_info["charge_complex"].values[0]
        multiplicity = ligand_info["multiplicity_complex"].values[0]
        ensemble_dir = ligand_dir.joinpath("Ensemble_Opt", f"{ligand_name}_CuOA_Ensemble_Opt")
        ox_dir = create_dirs(ligand_name, "E_ox", "CuOA", jobs_dir, output=False)
        red_dir = create_dirs(ligand_name, "E_red", "CuOA", jobs_dir, output=False)

        for conf_dir in ensemble_dir.glob('conformer_*'):
            conf_name = conf_dir.name
            num = int(conf_name.split("_")[-1])
            try:

                opt_xyz_file = conf_dir.joinpath("output", 'input.xyz')
                temp_xyz_file = template_dir.joinpath(f"{ligand_name}.xyz")
                orca_output = conf_dir.joinpath("output",
                                                 f"output_{ligand_name}_CuOA_Ensemble_Opt_conf{num}.out")
                if check_optimization_status(orca_output, opt_xyz_file, temp_xyz_file):



                    xyz_path_conf = conf_dir.joinpath('output', 'input.xyz')
                    conf_ox_dir = ox_dir.joinpath(conf_name)
                    conf_red_dir = red_dir.joinpath(conf_name)
                    job_name_ox = f"{ligand_name}_CuOA_E_ox"
                    job_name_red = f"{ligand_name}_CuOA_E_red"
                    if conf_ox_dir.joinpath("output").exists() and not dir_empty(conf_ox_dir.joinpath("output")):
                        raise FileExistsError
                    else:
                        # write orca input and batch file for oxidation
                        # copy xyz file to oxidation directory
                        Path.mkdir(conf_ox_dir, exist_ok=True)
                        xyz_path_ox = conf_ox_dir.joinpath(f"{ligand_name}_CuOA_E_ox_input.xyz")
                        shutil.copy(xyz_path_conf, xyz_path_ox)
                        write_orca_input(conf_ox_dir, job_name_ox, scf_type="RKS", functional="TPSS", basis_set="def2-SVP",
                                         basis_options='NewGTO Cu "def2-TZVP"',
                                         sp_basis_set="def2-TZVP",
                                         dispersion_correction="D3BJ",
                                         runtype="Freq", options="SARC/J RI", miscellaneous="KeepDens TightSCF",
                                         charge=charge + 1,
                                         multiplicity=1, single_point=True, memory="1000")
                        write_batch_file(job_name_ox, conf_ox_dir, xyz_path_ox, memory="16GB", tasks_per_node=12)
                        # create output directory
                        Path.mkdir(conf_ox_dir.joinpath("output"), exist_ok=True)
                        print(f"Written E_oxidation input for {ligand_name} in {conf_ox_dir}")

                    if conf_red_dir.joinpath("output").exists() and not dir_empty(conf_red_dir.joinpath("output")):
                        raise FileExistsError
                    else:
                        # write orca input and batch file for reduction
                        # copy xyz file to reduction directory
                        Path.mkdir(conf_red_dir, exist_ok=True)
                        xyz_path_red = conf_red_dir.joinpath(f"{ligand_name}_CuOA_Ered_input.xyz")
                        shutil.copy(xyz_path_conf, xyz_path_red)
                        write_orca_input(conf_red_dir, job_name_red, scf_type="RKS", functional="TPSS", basis_set="def2-SVP",
                                         basis_options='NewGTO Cu "def2-TZVP"',
                                         sp_basis_set="def2-TZVP",
                                         dispersion_correction="D3BJ",
                                         runtype="Freq", options="SARC/J RI", miscellaneous="KeepDens TightSCF",
                                         charge=charge - 1,
                                         multiplicity=1, single_point=True, memory="1000")
                        write_batch_file(job_name_red, conf_red_dir, xyz_path_red, memory="16GB", tasks_per_node=12, molden=True)

                        # create output directory
                        Path.mkdir(conf_red_dir.joinpath("output"), exist_ok=True)
                        print(f"Written E_reduction input for {ligand_name} in {conf_red_dir}")
                else:
                    print(f"Optimization not successful for {ligand_name}_{conf_name}.")
                    continue
            except FileExistsError:
                print(f"Job already ran for {ligand_name} {conf_name}.")
                continue