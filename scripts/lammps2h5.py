import subprocess
import sys

import h5py
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_file(in_path, out_path, include_velocities=False):
    num_timesteps = fetch_num_timesteps(in_path)
    batch_size = 500

    with open(in_path, 'r') as lammpstr_file, h5py.File(out_path, 'w', libver='latest', rdcc_nbytes=200 * 1024 ** 2) as h5_file:  # 200 MB cache size
        timestep, bbox, num_atoms = read_header(lammpstr_file)
        lammpstr_file.seek(0, 0)

        h5_file.create_dataset('timestep', data=np.arange(num_timesteps))
        position_ds = h5_file.create_dataset('position', (num_timesteps, num_atoms, len(bbox)), dtype=np.float32, shuffle=True, compression="gzip", compression_opts=9)
        type_ds = h5_file.create_dataset('particle_type', (num_atoms,), dtype=np.int16)
        if include_velocities:
            velocity_ds = h5_file.create_dataset('velocity', (num_timesteps, num_atoms, len(bbox)), dtype=np.float32, shuffle=True, compression="gzip", compression_opts=9)
        else:
            velocity_ds = None

        np_bbox = np.float32(np.array(bbox)[:, 1:])
        bbox_size = np_bbox[:, 1] - np_bbox[:, 0]
        shift = - np_bbox[:, 0] - bbox_size / 2

        # STATES:
        # headers - 6 + len(bbox)
        # atom - num_atoms

        header_length = 6 + len(bbox)
        state = 'header'
        it_in_state = 0
        timestep = -1
        batch_idx = 0
        batch_it = 0

        total = (header_length + num_atoms) * num_timesteps

        positions = np.empty((batch_size, num_atoms, len(bbox)), dtype=np.float32)
        if include_velocities:
            velocities = np.empty((batch_size, num_atoms, len(bbox)), dtype=np.float32)
        else:
            velocities = None

        for line in tqdm(lammpstr_file, total=total):
            if state == 'atom':
                read_atom(timestep, batch_it, line, positions, velocities, type_ds, include_velocities=include_velocities)

            if 'TIMESTEP' in line:
                timestep += 1

                if timestep == num_timesteps:
                    break

            it_in_state += 1
            if state == 'header' and it_in_state == header_length:
                state = 'atom'
                it_in_state = 0
            elif state == 'atom' and it_in_state == num_atoms:
                if batch_it == batch_size - 1:
                    position_ds[batch_idx * batch_size:(batch_idx + 1) * batch_size] = positions + shift
                    if include_velocities:
                        velocity_ds[batch_idx * batch_size:(batch_idx + 1) * batch_size] = velocities

                    positions = np.empty((batch_size, num_atoms, len(bbox)), dtype=np.float32)
                    if include_velocities:
                        velocities = np.empty((batch_size, num_atoms, len(bbox)), dtype=np.float32)
                    else:
                        velocities = None

                    batch_it = 0
                    batch_idx += 1
                else:
                    batch_it += 1

                state = 'header'
                it_in_state = 0

        position_ds[batch_idx * batch_size:] = positions[:batch_it] + shift
        if include_velocities:
            velocity_ds[batch_idx * batch_size:] = velocities[:batch_it]

        # for i in trange(num_timesteps):
        #     timestep, bbox, num_atoms, types, positions, velocities = read_timestep(lammpstr_file, include_velocities)
        #
        #     if i == 0:
        #         h5_file.create_dataset('particle_type', data=types)
        #
        #     # Shift all positions to make (0,0) in the center of the bbox
        #     positions = positions + shift
        #
        #     position_ds[i] = positions
        #
        #     if include_velocities:
        #         velocity_ds[i] = velocities

    # plot(all_positions, velocity, acceleration)


def fetch_num_timesteps(in_path):
    return int(subprocess.run(['grep', '-c', 'TIMESTEP', in_path], capture_output=True).stdout)


def plot(all_positions, velocity, acceleration):
    speed = np.linalg.norm(velocity[:10], axis=2).flatten()
    print("Computing histogram")
    print(np.histogram(speed))

    print("Starting to render")
    sns.displot(x=speed)
    plt.show()


def read_header(f):
    f.readline()  # ITEM: TIMESTEP
    text = f.readline()
    timestep = int(text)

    f.readline()  # ITEM: NUMBER OF ATOMS
    text = f.readline()
    num_atoms = int(text)

    text = f.readline()  # ITEM: BOX BOUNDS
    boundary_conditions = text.replace('ITEM: BOX BOUNDS', "").strip().split(' ')
    bbox = []

    for i in range(len(boundary_conditions)):
        text = f.readline()
        lower, upper = text.split()

        bbox.append((boundary_conditions[i], float(lower), float(upper)))

    return timestep, bbox, num_atoms


def read_atom(timestep, batch_it, line, position_ds, velocity_ds, type_ds, include_velocities=False):

    if include_velocities:
        aid, atype, x, y, z, vx, vy, vz = line.split(' ')
    else:
        aid, atype, x, y, z = line.split(' ')

    aid = int(aid) - 1

    position_ds[batch_it, aid] = np.array([x, y, z]).astype(np.float32)

    if timestep == 0:
        type_ds[aid] = int(atype)

    if include_velocities:
        velocity_ds[batch_it, aid] = np.array([vx, vy, vz]).astype(np.float32)


if __name__ == '__main__':
    read_file(sys.argv[1], sys.argv[2], include_velocities=True)

#
# in_path = "data/md_lammps/md.lammpstrj"
# out_path = "data/md_lammps/md_compressed.h5"
#
# read_file(in_path, out_path)
