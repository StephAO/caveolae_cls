import numpy as np
import os
import scipy.io as sio

from caveolae_cls.data_handler import DataHandler


def generate_cube(num_points=1024):
    proj_dim = DataHandler.proj_dim
    center = proj_dim / 2
    blob = np.zeros([num_points, 3])
    half_side_length = np.random.randint(25, 125)
    for i in xrange(num_points):
        point = np.array([np.random.randint(-half_side_length, half_side_length),
                          np.random.randint(-half_side_length, half_side_length),
                          np.random.randint(-half_side_length, half_side_length)])
        blob[i] = center + point
    return blob


def generate_sphere(num_points=1024):
    proj_dim = DataHandler.proj_dim
    center = proj_dim / 2
    blob = np.zeros([num_points, 3])
    radius = np.random.randint(25, 175)
    i = 0
    while i < num_points:
        point = np.array([np.random.randint(-radius, radius),
                          np.random.randint(-radius, radius),
                          np.random.randint(-radius, radius)])
        if np.linalg.norm(point) > radius:
            continue
        else:
            blob[i] = center + point
            i += 1
    return blob


def generate_data(num_instances=2000, directory="/staff/2/sarocaou/data/simple_pointcloud"):
    if not os.path.exists(directory):
        os.mkdir(directory)
    for i in xrange(num_instances / 2):
        sphere = generate_sphere()
        cube = generate_cube()
        sio.savemat(os.path.join(directory, "sphere_" + str(i)), {"blob": sphere}, do_compression=True)
        sio.savemat(os.path.join(directory, "cube_" + str(i)), {"blob": cube}, do_compression=True)


def main():
    generate_data()


if __name__ == "__main__":
    main()
