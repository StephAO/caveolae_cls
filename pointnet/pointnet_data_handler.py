import numpy as np
import scipy.io as sio
import sys

from caveolae_cls.data_handler import DataHandler


class PointNetDataHandler(DataHandler):

    def __init__(self, use_softmax=False):
        super(PointNetDataHandler, self).__init__(use_softmax=use_softmax)

    def load_point_cloud(self, filename):
        """
        Load point cloud data and label given a file
        """
        f = sio.loadmat(filename)
        data = f['blob'][:]
        data -= np.mean(data, 0)
        data /= np.amax(abs(data))
        label = DataHandler.get_label_from_filename(filename)
        if self.use_softmax:
            l = np.zeros([2])
            l[label] = 1
            label = l
        return data, label

    def format_point_cloud(self, pc, num_points):
        """
        Appends or removes points in point cloud to match input size.
        :param pc: point cloud to pass in [N, 3]
        :param num_points: Expected number of points in point cloud
        :return: resize point cloud
        """
        resized_pc = np.zeros([num_points, 3])
        if len(pc) > num_points:
            # Randomly sample point cloud to reduce size
            resized_pc = pc[np.random.choice(pc.shape[0], num_points,
                                             replace=False), :]
        elif len(pc) < num_points:
            # Duplicate last point to fill point cloud. Because of the max function
            # the duplicated points will not affect the output
            resized_pc[:len(pc)] = pc
            resized_pc[len(pc):] = pc[-1]
        else:
            resized_pc = pc

        return resized_pc

    @staticmethod
    def jitter_point_cloud(data, sigma=1, clip=10):
        """
        Randomly jitter points. jittering is per point.
        :param batch_data: BxNx3 array, original batch of point clouds
        :return BxNx3 array, jittered batch of point clouds
        """
        B, N, C = data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
        jittered_data += data
        return jittered_data

    @staticmethod
    def rotate_point_cloud(data):
        """
        Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction
        :param batch_data: BxNx3 array, original batch of point clouds
        :return BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(data.shape, dtype=np.float32)
        for k in xrange(data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def get_batch(self, batch_shape, use='train', label=None, exp_cell_token=None):
        """
        Generator that will return batches
        :param files: List of data file names. Each file should contain a 1 element.
        :param max_ratio_n: Maximum ratio of negative data points in a single batch.
                            A value of 1 would mean that a batch containing only
                            negative elements would be acceptable.
        :param batch_shape: Expected shape of a single batch
        :return: Generates batches
        """
        self.batch_size = batch_shape[0]
        self.data = np.zeros(batch_shape)
        self.labels = np.zeros([self.batch_size, 2] if self.use_softmax else self.batch_size)

        files = self.get_data_files(use=use, label=label, exp_cell_token=exp_cell_token)

        random_file_idxs = np.arange(len(files))
        np.random.shuffle(random_file_idxs)

        i = 0
        # num_negatives = 0
        progress = 0
        for count, idx in enumerate(random_file_idxs):
            if float(count)/len(random_file_idxs) >= progress + 0.05:
                progress += 0.05
                print str(int(round(progress * 100))) + "%",
                sys.stdout.flush()
                if abs(progress - 0.95) <= 0.01:
                    print ""
            f = files[idx]
            d, l = self.load_point_cloud(f)
            d = self.format_point_cloud(d, batch_shape[1])
            self.data[i] = d
            self.labels[i] = l

            i += 1
            if i >= self.batch_size:
                # Augment batched point clouds by rotation and jittering
                self.data = PointNetDataHandler.rotate_point_cloud(self.data)
                self.data = PointNetDataHandler.jitter_point_cloud(self.data)
                # Yield batch
                yield self.data, self.labels
                i = 0
                # num_negatives = 0


