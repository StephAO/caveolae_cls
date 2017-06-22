import numpy as np
import  os
import scipy.io as sio
import sys

from caveolae_cls.data_handler import DataHandler

def blob_to_projections(blob_filename, target_dir):
    proj_dims = 600;
    proj = np.zeros(proj_dims, proj_dims, 3)

    center = proj_dims / 2.

    blob = sio.loadmat(blob_filename)["blob"]
    coords = np.rint(center + blob - np.mean(blob, axis=0))

    for x, y, z in range(coords):
        proj[x][y][0] = 1
        proj[x][z][1] = 1
        proj[y][z][2] = 1

    proj.dump(os.path.join(target_dir, blob_filename.split('/')[-1]))

    return proj

def convert_all_blobs_to_projections(blob_dir, target_dir):
    for f in DataHandler.get_data_files(blob_dir, target_dir):
        blob_to_projections(f)


def main():
    blob_dir = sys.argv[1]
    target_dir = sys.argv[2]
    convert_all_blobs_to_projections(blob_dir, target_dir)


if __name__ == "__main__":
    main()