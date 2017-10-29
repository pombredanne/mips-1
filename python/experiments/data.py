import array
import logging
import os

import numpy as np
import scipy.sparse as sp
import torch as th
from torch.autograd import Variable as V
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_data():
    RAW_PATH = '../data/DeliciousLarge/deliciousLarge_train.txt'
    X_PATH = '../data/DeliciousLarge/X_train.csr.npz'
    Y_PATH = '../data/DeliciousLarge/Y_train.csr.npz'

    if os.path.exists(X_PATH):
        logging.info(f"Data already present at {X_PATH}. Loading...")

        X = load_csr(X_PATH)
        Y = load_csr(Y_PATH)

    else:
        logging.info(f"Data not found, I'm going to prepare it and store at {X_PATH}.")

        X, Y = libsvm_to_csr(RAW_PATH)

        save_csr(X, X_PATH)
        save_csr(Y, Y_PATH)

    logging.info("Trimming csr matrices...")

    X_col_mask = trim(X, 0, t=5)
    Y_col_mask = trim(Y, 0, t=3)

    X = X[:, X_col_mask]
    Y = Y[:, Y_col_mask]

    X_row_mask = trim(X, 1, t=1)
    Y_row_mask = trim(Y, 1, t=1)

    mask = X_row_mask & Y_row_mask

    X = X[mask, :]
    Y = Y[mask, :]

    logging.info(f"Prepared data of shapes: {X.shape, Y.shape}")

    return X, Y


class CSRDataset(Dataset):
    def __init__(self, X_csr, Y_csr, sorted=False):
        assert X_csr.shape[0] == Y_csr.shape[0]
        self.X_csr = X_csr
        self.Y_csr = Y_csr
        self.sorted_indices = None

        if sorted:
            self.sort()

    def sort(self):
        row_sizes = np.diff(self.X_csr.indptr)
        self.sorted_indices = np.argsort(row_sizes)

        return self

    def randomize(self):
        self.sorted_indices = None

        return self

    def __getitem__(self, index):
        if self.sorted_indices is not None:
            index = self.sorted_indices[index]

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
                 log_transform=True, use_bag=False, max_repeat=1, subsample=np.inf):

        self.n_labels      = n_labels
        self.scale_y       = scale_y
        self.log_transform = log_transform
        self.use_bag       = use_bag
        self.max_repeat    = max_repeat
        self.subsample     = subsample

    def x_transform(self, x):
        indices, weights = x

        if len(indices) > self.subsample:
            _inds = np.random.choice(np.arange(len(indices)), size=self.subsample, replace=False)
            indices = [int(indices[i])   for i in _inds]
            weights = [float(weights[i]) for i in _inds]

        weights = th.FloatTensor(weights)

        if self.log_transform:
            weights = th.log1p(weights)

        if self.use_bag:
            counts = th.clamp(th.round(weights), 1, self.max_repeat)
            indices = [int(item)
                       for idx, cnt in zip(indices, counts)
                       for item in [idx] * int(cnt)]

        indices = th.LongTensor(indices)
        return indices, weights

    def y_transform(self, labels):
        assert len(labels) > 0

        value = 1 if not self.scale_y else 1. / len(labels)
        y = np.zeros((self.n_labels, ), dtype=np.float32)
        y[labels] = value

        return th.from_numpy(y)

    def collate_fn(self, batch):
        batch = [(self.x_transform(x), self.y_transform(y))
                 for x, y in batch]

        xs, ys = zip(*batch)
        _indices, _weights = zip(*xs)

        if self.use_bag:
            # prepare offsets for EmbeddingBag

            indices = th.cat(_indices)

            offsets, offset = [], 0
            for I in _indices:
                offsets.append(offset)
                offset += len(I)
            offsets = th.LongTensor(offsets)

            return (indices, offsets), th.stack(ys)

        else:
            # zero-pad to the longest sequence

            max_len = max(len(I) for I in _indices)
            shape   = (len(_indices), max_len)

            indices = th.zeros(*shape).long()
            weights = th.zeros(*shape)

            for i, (I, W) in enumerate(zip(_indices, _weights)):
                indices[i, :len(I)] = (I+1)
                weights[i, :len(W)] = W

            return (indices, weights), th.stack(ys)


def to_cuda_var(*x, cuda=False, var=True, volatile=False):

    if len(x) == 1:
        x, = x

    typ = type(x)

    if typ in {list, tuple}:
        return typ(to_cuda_var(item, cuda=cuda) for item in x)

    if var:
        if cuda:
            x = x.cuda(async=True)

        x = V(x, volatile=volatile)

    else:
        if cuda:
            x = x.cuda()

    return x


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
            row_indices, row_values = map(int, row_indices), map(float, row_values)

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


def trim(X, dim, t):
    X       = X.tocsc() if (dim == 0) else X.tocsr()
    mask    = np.array((X > 0).sum(dim) >= t).ravel()

    return mask


def save_csr(obj, filename):
    np.savez(filename, data=obj.data, indices=obj.indices, indptr=obj.indptr,
             shape=obj.shape)


def load_csr(filename: str):
    loader = np.load(filename)

    data    = loader['data']
    indices = loader['indices']
    indptr  = loader['indptr']
    shape   = loader['shape']

    return sp.csr_matrix((data, indices, indptr),
                         shape=shape)