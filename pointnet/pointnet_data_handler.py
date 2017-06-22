import numpy as np
import scipy.io as sio
import sys

from caveolae_cls.data_handler import DataHandler


class PointNetDataHandler(DataHandler):

    def __init__(self, use_softmax=False):
        super(PointNetDataHandler, self).__init__()
        self.p_files = DataHandler.get_data_files(
            '/staff/2/sarocaou/data/pointcloud_positive')
        self.n_files = DataHandler.get_data_files(
            '/staff/2/sarocaou/data/pointcloud_negative')[:len(self.p_files)]
        self.p_train_files = self.p_files[:int(0.9 * len(self.p_files))]
        self.p_eval_files = self.p_files[int(0.9 * len(self.p_files)):]
        self.n_train_files = self.n_files[:int(0.9 * len(self.n_files))]
        self.n_eval_files = self.n_files[int(0.9 * len(self.n_files)):]
        self.use_softmax = use_softmax

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

    def jitter_point_cloud(self, sigma=0.01, clip=0.05):
        """
        Randomly jitter points. jittering is per point.
        :param batch_data: BxNx3 array, original batch of point clouds
        :return BxNx3 array, jittered batch of point clouds
        """
        B, N, C = self.data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
        jittered_data += self.data
        self.data = jittered_data

    def rotate_point_cloud(self):
        """
        Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        :param batch_data: BxNx3 array, original batch of point clouds
        :return BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(self.data.shape, dtype=np.float32)
        for k in xrange(self.data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = self.data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        self.data = rotated_data

    def get_batch(self, batch_shape, eval=False, type='mixed'):
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

        if eval:
            p_files = self.p_eval_files
            n_files = self.n_eval_files
        else:
            p_files = self.p_train_files
            n_files = self.n_train_files

        p_random_file_idxs = np.arange(len(p_files))
        np.random.shuffle(p_random_file_idxs)

        n_random_file_idxs = np.arange(len(n_files))
        np.random.shuffle(n_random_file_idxs)

        random_file_idxs = zip(p_random_file_idxs, n_random_file_idxs)

        i = 0
        # num_negatives = 0
        progress = 0
        for count, idxs in enumerate(random_file_idxs):
            if float(count)/len(random_file_idxs) >= progress + 0.05:
                progress += 0.05
                print str(int(round(progress * 100))) + "%",
                sys.stdout.flush()
                if abs(progress - 0.95) <= 0.01:
                    print ""
            p_idx, n_idx = idxs
            p_f = p_files[p_idx]
            d, l = self.load_point_cloud(p_f)
            # if l == 0:
            #     if num_negatives >= int(max_ratio_n * self.batch_size):
            #         continue
            #     num_negatives += 1
            d = self.format_point_cloud(d, batch_shape[1])
            self.data[i] = d
            self.labels[i] = l

            n_f = n_files[n_idx]
            d, l = self.load_point_cloud(n_f)
            d = self.format_point_cloud(d, batch_shape[1])
            self.data[i+1] = d
            self.labels[i+1] = l
            i += 2
            if i >= self.batch_size:
                # Augment batched point clouds by rotation and jittering
                # self.rotate_point_cloud()
                # self.jitter_point_cloud()
                # Yield batch
                yield self.data, self.labels
                i = 0
                # num_negatives = 0


