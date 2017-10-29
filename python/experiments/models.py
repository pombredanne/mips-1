from typing import List

import torch as th
from torch.nn import functional as F


def softmax_bce_loss(inputs, targets):
    inputs = F.softmax(inputs, dim=1)
    return F.binary_cross_entropy(inputs, targets)


class WeightedEmbeddingBag(th.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.embedding = th.nn.Embedding(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                                         scale_grad_by_freq, sparse)

    def forward(self, indices, weights):

        weights = weights[:, :, None]

        embedded = self.embedding(indices)
        scaled_emb = embedded * weights
        mean_emb = (scaled_emb / weights.sum(1, keepdim=True)).sum(1)

        return mean_emb


class DummyEmbbeder(th.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = th.nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                            max_norm=max_norm, norm_type=norm_type,
                                            scale_grad_by_freq=scale_grad_by_freq)

    def forward(self, indices, offsets):
        assert indices.dim() == offsets.dim() == 1

        return self.embedding(indices, offsets)


class FeedForward(th.nn.Module):
    def __init__(self, embedder: th.nn.Embedding, layer_sizes: List[int],
                 activation=th.nn.ReLU, batchnorm=True, dropout=0.):

        super().__init__()

        self.embedder    = embedder
        self.activation  = activation
        self.dropout     = dropout
        self.layers      = []
        self.layer_sizes = [embedder.embedding_dim] + layer_sizes

        for prev_dim, dim in zip(self.layer_sizes[:-2], self.layer_sizes[1:]):

            self.layers.append(th.nn.Linear(prev_dim, dim))
            self.layers.append(self.activation())

            if batchnorm:
                self.layers.append(th.nn.BatchNorm1d(dim))

            if self.dropout > 0.:
                self.layers.append(th.nn.Dropout(self.dropout))

        self.layers.append(th.nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.core = th.nn.Sequential(*self.layers)

        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.layers:
            if isinstance(layer, th.nn.Linear):
                th.nn.init.kaiming_normal(layer.weight)

        th.nn.init.kaiming_normal(self.embedder.embedding.weight, mode='fan_out')

    def forward(self, x):
        E = self.embedder(*x)
        O = self.core(E)

        return O