import numpy as np
import os
import scipy.io as sio
import sys

from caveolae_cls.data_handler import DataHandler
from caveolae_cls.pointnet.pointnet_data_handler import PointNetDataHandler
from caveolae_cls.cnn.cnn_model import CNN


def upsample(blob_dir, target_dir, num_new_samples=None):
    files = DataHandler.get_data_files(blob_dir)
    np.random.shuffle(files)
    if num_new_samples is None:
        num_new_samples = len(files)

    i = 0
    new_samples = 0
    while new_samples < num_new_samples:
        data = sio.loadmat(files[i])
        blob = data["blob"]
        orig_shape = blob.shape
        blob = np.expand_dims(blob, 0)
        blob = PointNetDataHandler.rotate_point_cloud(blob)
        blob = PointNetDataHandler.jitter_point_cloud(blob)
        blob = np.squeeze(blob)
        assert blob.shape == orig_shape
        proj = blob_to_projections(blob)
        data["Img3Ch"] = proj
        del data["blob"]
        new_file_path = os.path.join(target_dir, 'synthetic_' + str(new_samples) + '_' +
                                     files[i].split('/')[-1].split('_')[-1])
        sio.savemat(new_file_path, data)
        new_samples += 1
        i = (i + 1) % len(files)

def blob_to_projections(blob):
    proj_dim = int(CNN.proj_dim)
    proj = np.zeros([proj_dim, proj_dim, 3])

    center = proj_dim / 2.
    coords = np.rint(center + blob - np.mean(blob, axis=0))

    num_excluded_points = 0

    for x, y, z in coords.astype(int):
        if 0 <= x < proj_dim and 0 <= y < proj_dim and 0 <= z < proj_dim:
            proj[x][y][0] = 1
            proj[x][z][1] = 1
            proj[y][z][2] = 1
        else:
            num_excluded_points += 1

    if num_excluded_points > 0:
        print num_excluded_points, "excluded of a total of", len(coords)

    return proj


def convert(blob_dir, target_dir):
    for f in DataHandler.get_data_files(blob_dir):
        data = sio.loadmat(f)
        blob = data["blob"]
        proj = blob_to_projections(blob)
        data["Img3Ch"] = proj
        del data["blob"]
        new_file_path = os.path.join(target_dir, f.split('/')[-1])
        sio.savemat(new_file_path, data)


def main():
    blob_dir = sys.argv[1]
    target_dir = sys.argv[2]
    if len(sys.argv) >= 4:
        fn_flag = sys.argv[3]
        if fn_flag == "convert":
            fn = convert
        else:
            fn = upsample
    else:
        fn = upsample

    fn(blob_dir, target_dir)

if __name__ == '__main__':
    main()




