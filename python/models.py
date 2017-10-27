import torch as th
from torch.autograd import Variable as V


class WeightedEmbeddingBag(th.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):
        super().__init__()

        self.embedding = th.nn.Embedding(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                                         scale_grad_by_freq, sparse)

    def forward(self, indices, weights):
        weights = weights[:, :, None]

        embedded = self.embedding(indices)
        scaled_emb = embedded * weights
        mean_emb = (scaled_emb / weights.sum(1, keepdim=True)).sum(1)

        return mean_emb