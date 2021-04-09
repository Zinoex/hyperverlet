from argparse import ArgumentParser
import os

from scripts.md2h5 import read_file


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--num-configurations', type=int, required=True, help='Number of MD simulations to create')
    parser.add_argument('--output-dir', type=str, required=True, help='Name of the output directory')
    return parser.parse_args()


def create_md_dataset(num_config, output_dir):
    start_seed = 11111
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_config):
        #Create MD simulation using Lammps
        print("Creating configuration: {}".format(i))
        seed = start_seed + i
        run_cmd = "singularity run --nv -B$(pwd):/host_pwd --pwd /host_pwd lammps_29Oct2020.sif mpirun -n 1 lmp -var seed {} -var output_dir {} -k on g 1 -sf kk -pk kokkos cuda/aware on neigh full comm device binsize 2.8 -pk kokkos newton on -in /host_pwd/scripts/in.KA".format(seed, output_dir)
        print("Running cmd: {}".format(run_cmd))
        os.system(run_cmd)

        #Converte MD simulation to h5py
        lammps_file = os.path.join(output_dir, f"md{seed}.lammpstrj")
        output_path = os.path.join(output_dir, f"md{seed}.h5")
        print(f"Converting {lammps_file} to h5")
        h5_cmd = "singularity run glassy_dynamics.sif scripts/md2h5.py {} {}".format(lammps_file, output_path)
        os.system(h5_cmd)
        
        #Clearn up
        print(f"{lammps_file} was successfully converted to h5, proceeding to cleanup")
        os.remove(lammps_file)
        print(f"{lammps_file} was successfully removed")
    print("All {} is now created".format(num_config))


def main():
    args = parse_arguments()
    create_md_dataset(args.num_configurations, args.output_dir)


if __name__ == '__main__':
    main()



