from pathlib import Path


jobs_path = Path("/home/employee/f_katz01/Research/PlateOpt/jobs/")

def hartree_to_kcal_mol(hartree):
    """Convert energy from Hartree to kcal/mol"""
    return hartree * 627.509  # Standard conversion factor


def parse_conformers(input_text):
    """Parse conformers from input text and return list of (energy, atoms, coordinates)"""
    conformers = []
    lines = input_text.strip().split('\n')

    i = 0
    while i < len(lines):
        # Get number of atoms
        n_atoms = int(lines[i])

        # Get energy from next line (assumes format: "-43.6943482057 converged=true")
        energy = float(lines[i + 1].split()[0])

        # Get atomic coordinates
        atoms = []
        coords = []
        for j in range(n_atoms):
            line = lines[i + 2 + j].split()
            atoms.append(line[0])
            coords.append([float(x) for x in line[1:4]])

        conformers.append((energy, atoms, coords))
        i += n_atoms + 2  # Move to next conformer

    return conformers


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

def write_orca_input(conf_dir, job_name):
    """Write ORCA input file for geometry optimization"""
    orca_input = '! B3LYP D3 ZORA ZORA-def2-TZVP SARC/J RIJCOSX Opt Freq KeepDens TightSCF'
    orca_input += '\n\n%elprop\n  Quadrupole True\n  Polar 1\nend'  # Quadropole and Polarizability
    orca_input += f'\n\n*xyzfile 0 1 /tmp/{job_name}/input.xyz\n\n%pal nprocs 12 end'
    orca_input += '\n\n%Method\n  WriteJSONPropertyfile True\nend' # Write JSON file with properties
    with open(conf_dir.joinpath("input.inp"), "w") as f:
        f.write(orca_input)

def write_batch_file(job_name, job_directory, xyz_path):
    batch_file = Path('submit.batch').read_text()
    batch_file = batch_file.replace('*job_name*', job_name)
    batch_file = batch_file.replace('*job_directory*', str(job_directory))
    batch_file = batch_file.replace('*xyz_path*', str(xyz_path))
    Path(job_directory.joinpath("submit.batch")).write_text(batch_file)


def main():
    # Create output directory if it doesn't exist
    name = "L1" # name of the ligand
    species = "CuBr" # species
    job_type = "Ensemble_Opt" # job type
    ensemble_name = f"{name}_{species}_{job_type}"
    ensemble_dir = jobs_path.joinpath(name, job_type, f"{ensemble_name}")
    Path.mkdir(ensemble_dir, exist_ok=False, parents=True)

    # Read input file
    goat_output = jobs_path.joinpath(name, 'GOAT', f"{name}_{species}_GOAT", "output", 'input.finalensemble.xyz')
    input_text = goat_output.read_text()

    # Parse conformers
    conformers = parse_conformers(input_text)
    #TODO: Calculate relative energies

    # Prepare Orca input for each conformer
    for i, (energy, atoms, coords) in enumerate(conformers, 1):
        energy_kcal = hartree_to_kcal_mol(energy)

        # create conformer directory
        conf_dir = ensemble_dir.joinpath(f"conformer_{i}")
        Path.mkdir(conf_dir, exist_ok=False)

        # write xyz file with energy
        xyz_path = conf_dir.joinpath(f"conf{i}_input.xyz")
        write_xyz_file(xyz_path, energy_kcal, atoms, coords)

        # write orca input and batch file
        job_name = f"{ensemble_name}_conf{i}"
        write_orca_input(conf_dir, f"{job_name}")
        write_batch_file(job_name, conf_dir, xyz_path)

        # create output directory
        Path.mkdir(conf_dir.joinpath("output"), exist_ok=False)
        print(f"Written conformer {i} with energy {energy_kcal:.2f} kcal/mol to {xyz_path}")


if __name__ == "__main__":
    main()