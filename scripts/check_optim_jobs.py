from pathlib import Path
from utils import check_orca_success, check_optimization_success, check_neg_frequencies, read_energy, dir_empty, \
    check_connectivity




jobs_dir = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/featurization/")
species = "CuOA"
for ligand_dir in jobs_dir.iterdir():
    if ligand_dir.is_dir(): # and ligand_dir.name in ["L31", "L32"]:
        ligand_name = ligand_dir.name  # ligand_name of the ligand
        ensemble_path = jobs_dir.joinpath(ligand_name, "Ensemble_Opt",
                                           f"{ligand_name}_{species}_Ensemble_Opt")  # Path to ensemble directory
        for conformer_dir in ensemble_path.glob('conformer_*'):
            if not dir_empty(conformer_dir.joinpath("output")):
                name = conformer_dir.name
                num = int(name.split("_")[-1])  # Extract number from conformer name
                orca_output = conformer_dir.joinpath("output", f"output_{ligand_name}_CuOA_Ensemble_Opt_conf{num}.out")
                xyz_path_opt = conformer_dir.joinpath("output", 'input.xyz')
                xyz_path_temp = Path(f"/home/employee/f_katz01/Research/PlateOpt/ComplexFeaturization/data/geometries/input/{ligand_name}.xyz")
                if not check_orca_success(orca_output):
                    #print(f"ORCA job failed for conformer {num} of {ligand_name}.")
                    pass
                if not check_optimization_success(orca_output):
                    #print(f"Optimization failed for conformer {num} of {ligand_name}.")
                    pass
                if check_neg_frequencies(orca_output):
                    print(f"Negative frequencies found for conformer {num} of {ligand_name}.")
                    pass
                if not check_connectivity(xyz_path_opt, xyz_path_temp):
                    #print(f"Conformer {num} of {ligand_name} has changed connectivity.")
                    pass

            else:
                pass
