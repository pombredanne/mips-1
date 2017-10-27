import array
import torch as th
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset

import numpy as np
import scipy.sparse as sp


def libsvm_to_csr(path):
    """ Read data in libsvm (extreme repo) format, and convert it into scipy csr matrices

    """

    with open(path) as f_in:

        num_documents, num_features, num_labels = [int(val) for val in
                                                   next(f_in).strip().split(' ')]

        y_indices = array.array('I')
        y_indptr = array.array('I', [0])

        x_indices = array.array('I')
        x_data = array.array('f')
        x_indptr = array.array('I', [0])

        for i, line in enumerate(f_in):
            labels, *features = line.strip().split()

            features = [item.split(":") for item in features]
            labels = [int(y) for y in labels.split(',')]

            if len(labels) <= 0 or len(features) <= 0:
                continue

            row_indices, row_values = zip(*features)

            x_indices.extend(row_indices)
            x_data.extend(row_values)
            x_indptr.append(len(x_indices))

            y_indices.extend(labels)
            y_indptr.append(len(y_indices))

        x_indices = np.frombuffer(x_indices, dtype=np.uint32)
        x_indptr = np.frombuffer(x_indptr, dtype=np.uint32)
        x_data = np.frombuffer(x_data, dtype=np.float32)
        x_shape = (num_documents, num_features)

        y_indices = np.frombuffer(y_indices, dtype=np.uint32)
        y_indptr = np.frombuffer(y_indptr, dtype=np.uint32)
        y_data = np.ones_like(y_indices, dtype=np.float32)
        y_shape = (num_documents, num_labels)

        X = sp.csr_matrix((x_data, x_indices, x_indptr), shape=x_shape)
        Y = sp.csr_matrix((y_data, y_indices, y_indptr), shape=y_shape)

        return X, Y


class CSRDataset(Dataset):
    def __init__(self, X_csr, Y_csr):
        assert X_csr.shape[0] == Y_csr.shape[0]
        self.X_csr = X_csr
        self.Y_csr = Y_csr

    def __getitem__(self, index):
        row = self.X_csr[index]

        return (row.indices, row.data), self.Y_csr[index].indices

    def __len__(self):
        return self.X_csr.shape[0]


class LocallySequentialSampler(Sampler):
    # noinspection PyMissingConstructor
    def __init__(self, data_source, window_size):
        self.data_source = data_source
        self.window_size = window_size
        self.n_steps = len(self.data_source) // self.window_size

    def __iter__(self):
        window_indices = th.randperm(self.n_steps).long()

        for w_idx in window_indices:
            for i in range(self.window_size):
                yield w_idx * self.window_size + i

    def __len__(self):
        return self.n_steps * self.window_size


class Preprocessor:
    def __init__(self, n_labels,  scale_y=True,
                 log_transform=True, use_bag=False, max_repeat=1):

        self.n_labels      = n_labels
        self.scale_y       = scale_y
        self.log_transform = log_transform
        self.use_bag       = use_bag
        self.max_repeat    = max_repeat

    def x_transform(self, x):
        indices, weights = x
        weights = th.FloatTensor(weights)

        if self.log_transform:
            weights = th.log1p(weights)

        if self.use_bag:
            counts = th.clamp(th.round(weights), 1, self.max_repeat)
            indices = [item
                       for idx, cnt in zip(indices, counts)
                       for item in [idx] * int(cnt)]

        indices = th.LongTensor(indices)
        return indices, weights

    def y_transform(self, labels):
        value = 1 if not self.scale_y else 1. / len(labels)
        y = th.zeros(self.n_labels)
        y[labels] = value

        return y

    def collate_fn(self, batch):
        batch = [(self.x_transform(x), self.y_transform(y))
                 for x, y in batch]

        xs, ys = zip(*batch)
        _indices, _weights = zip(*xs)

        if self.use_bag:
            # prepare offsets for EmbeddingBag

            indices = th.cat(_indices)
            offsets = th.LongTensor([0] + [len(I) for I in _indices])

            return indices, offsets, ys

        else:
            # zero-pad to the longest sequence

            max_len = max(len(I) for I in _indices)
            shape   = (len(_indices), max_len)

            indices = th.zeros(*shape).long()
            weights = th.zeros(*shape)

            for i, (I, W) in enumerate(zip(_indices, _weights)):
                indices[i, :len(I)] = (I+1)
                weights[i, :len(W)] = W

            return indices, weights, ys
