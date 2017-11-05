import numpy as np


def load_sift(fname, dtype=np.float32):
    data = np.fromfile(fname, dtype=dtype)
    d = data[0].view(np.int32)

    data = data.reshape(-1, d + 1)[:, 1:]
    data = np.ascontiguousarray(data.copy())

    return data


def save_sift(obj, fname, dtype=np.float32):
    obj = np.hstack([
        np.ones((obj.shape[0], 1)) * obj.shape[1],
        obj
    ]).astype(dtype)

    obj.tofile(fname)
