import numpy as np
import os
import scipy.io as sio
import shutil
import sys

from caveolae_cls.data_handler import DataHandler
from caveolae_cls.pointnet.pointnet_data_handler import PointNetDataHandler

def upsample(files, target_dir, num_new_samples=None):
    """Given a directory of 3D point clouds, upsamples"""
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
        sio.savemat(new_file_path, data, do_compression=True)
        new_samples += 1
        i = (i + 1) % len(files)


def convert(files, target_dir):
    num_skipped = 0
    num_convert = 0
    for f in files:
        data = sio.loadmat(f)
        blob = data["blob"]
        if len(data["blob"]) < 60:
            num_skipped += 1
            continue
        proj = blob_to_projections(blob)
        data["Img3Ch"] = proj
        del data["blob"]
        new_file_path = os.path.join(target_dir, f.split('/')[-1])
        sio.savemat(new_file_path, data, do_compression=True)
        num_convert += 1
    print "%d skipped, %d converted" % (num_skipped, num_convert)


def blob_to_projections(blob):
    proj_dim = int(DataHandler.proj_dim)
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


def create_training_validation(blob_dir, target_dir, input_type, validation_ratio=0.1, num_new_samples=None):
    training_dir = os.path.join(target_dir, "training")
    validation_dir = os.path.join(target_dir, "validation")
    test_dir = os.path.join(target_dir, "test")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    files = DataHandler.get_data_files(blob_dir)
    np.random.shuffle(files)

    num_val = int(validation_ratio * len(files))

    if input_type == "projection":
        convert(files[:num_val], test_dir)
        convert(files[num_val:num_val * 2], validation_dir)
        convert(files[num_val * 2:], training_dir)
        upsample(files[num_val * 2:], training_dir, num_new_samples=num_new_samples)
    elif input_type == "pointcloud":
        for f in files[:num_val]:
            shutil.copy2(f, test_dir)
        for f in files[num_val:num_val * 2]:
            shutil.copy2(f, validation_dir)
        for f in files[num_val * 2:]:
            shutil.copy2(f, training_dir)


def main():
    input_type = sys.argv[1]
    blob_dir = sys.argv[2]
    target_dir = sys.argv[3]
    validation_ratio = 0.1 if len(sys.argv) < 5 else float(sys.argv[4])
    num_new_samples = 0 if len(sys.argv) < 6 else int(sys.argv[5])

    create_training_validation(blob_dir, target_dir, input_type,
                               validation_ratio=validation_ratio, num_new_samples=num_new_samples)

if __name__ == '__main__':
    main()




