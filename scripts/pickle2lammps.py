import pickle
from tqdm import trange
import numpy as np


def read_pickle(path):
    with open(path, 'rb') as f:
        rollout_data = pickle.load(f)
    return rollout_data


def dump2lammps(in_path, out_path):
    data = read_pickle(in_path)

    with open(out_path, 'w') as f:
        particle_types = data['particle_types']
        positions = np.concatenate((data['initial_positions'], data['predicted_rollout']), axis=1)

        num_atoms = particle_types.shape[0]
        num_timesteps = positions.shape[1]

        bbox = data['bbox']
        bbox = bbox - np.expand_dims(bbox[:, 0], axis=1) - np.expand_dims(bbox[:, 1] - bbox[:, 0], axis=1) / 2

        for timestep in trange(num_timesteps):
            write_timestep(f, timestep)
            write_num_atoms(f, num_atoms)
            write_bbox(f, bbox)
            write_positions_and_type(f, timestep, num_atoms, particle_types, positions)


def write_timestep(f, timestep):
    f.write(f"ITEM: TIMESTEP\n{timestep}\n")


def write_num_atoms(f, num_atoms):
    f.write(f"ITEM: NUMBER OF ATOMS\n{num_atoms}\n")


def write_bbox(f, bbox):
    f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
    for (upper, lower) in bbox:
        f.write("{:e} {:e}\n".format(upper, lower))


def write_positions_and_type(f, timestep, num_atoms, particle_types, positions):
    f.write("ITEM: ATOMS id type x y z\n")

    for i in range(num_atoms):
        position = positions[i, timestep]
        f.write("{} {} {:f} {:f} {:f}\n".format(i, particle_types[i], *position))


if __name__ == '__main__':
    in_path = "output/test/rollout_valid_0.pkl"
    out_path = "lammps_out.lammpstrj"
    dump2lammps(in_path, out_path)
