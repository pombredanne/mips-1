import logging
import time

import numpy as np
import torch as th
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from .data import CSRDataset, LocallySequentialSampler, Preprocessor, \
    to_cuda_var, get_data
from .models import WeightedEmbeddingBag, FeedForward, DummyEmbbeder, softmax_bce_loss

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DefaultOpts:

    @classmethod
    def to_string(cls):
        for item in sorted(dir(cls)):

            if item.startswith('__'):
                continue

            yield f"`{item}:\t{str(getattr(cls, item))}`"

    _NAME = "DefaultConfig"

    LOGGING_ROOT  = "/tmp/runs"

    BATCH_SIZE    = 16
    N_EPOCHS      = 1

    SORTED        = True
    CUDA          = True
    SPARSE        = True

    PRINT_ACC     = 10
    PRINT_FREQ    = 1
    NUM_WORKERS   = 0 if not CUDA else 6

    EMBEDDING_DIM = 128
    LAYER_SIZES   = [512, 512, 512]

    LOSS          = softmax_bce_loss
    SCALE_Y       = False

    SUBSAMPLE     = 2048
    LOG_TRANSFORM = True

    USE_BAG       = False
    MAX_REPEAT    = 1


def run_ID(filename='/tmp/__TORCH_RUN_IDX'):
    import pickle

    try:
        with open(filename, 'rb') as f:
            _IDX = pickle.load(f)
    except FileNotFoundError:
        _IDX = 0

    _IDX += 1

    with open(filename, 'wb') as f:
        pickle.dump(_IDX, f)

    logging.info(f"Fetched run index: {_IDX}")

    return f'run_{_IDX}'


def train(loader, model, optimizer_factory, loss_fn, Opts: DefaultOpts):

    logging.info("Initiating training loop...")

    writer = SummaryWriter(Opts.LOGGING_ROOT + run_ID())

    for idx, param in enumerate(Opts.to_string()):
        writer.add_text('params', param, global_step=idx)

    optimizer = optimizer_factory()

    for epoch in range(Opts.N_EPOCHS):
        running_loss = 0.
        t0 = time.time()

        for i, data in enumerate(loader):

            optimizer.zero_grad()

            inputs, labels = to_cuda_var(data, cuda=Opts.CUDA)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            writer.add_scalar(f'loss/all', loss.data[0], global_step=epoch * len(loader) + i)
            writer.add_scalar(f'loss/epoch_{epoch}', loss.data[0], global_step=i)

            if Opts.PRINT_ACC > 0 and (i % Opts.PRINT_ACC) == 0:
                labels  = labels.data.cpu().numpy()
                outputs = outputs.data.cpu().numpy()

                best     = np.argmax(outputs, 1)
                relevant = labels[np.arange(len(best)), best]

                score = np.mean(relevant)

                writer.add_scalar(f'p_at_1/all', float(score),
                                  global_step=epoch * (len(loader) * Opts.BATCH_SIZE) + (i * Opts.BATCH_SIZE))
                writer.add_scalar(f'p_at_1/epoch_{epoch}', float(score),
                                  global_step=i * Opts.BATCH_SIZE)

            if (i % Opts.PRINT_FREQ) == (Opts.PRINT_FREQ - 1):
                msg = f"\repoch {epoch+1}, iter {i+1:_} of {len(loader):_}\t"\
                      f"loss: {running_loss / print_freq: .3f}\t"\
                      f"time per batch: {(time.time() - t0) / print_freq: .3f}"

                print(msg, end='', flush=True)

                running_loss = 0.0
                t0 = time.time()

            del inputs, labels, outputs, loss

    logging.info("bye.")


def main(Opts):
    th.manual_seed(42)
    th.cuda.manual_seed(42)

    X, Y = get_data()

    dataset = CSRDataset(X, Y,
                         sorted=Opts.SORTED)

    sampler = LocallySequentialSampler(dataset,
                                       window_size=Opts.BATCH_SIZE)

    preproc = Preprocessor(n_labels=Y.shape[1],
                           scale_y=Opts.SCALE_Y,
                           log_transform=Opts.LOG_TRANSFORM,
                           use_bag=Opts.USE_BAG,
                           max_repeat=Opts.MAX_REPEAT,
                           subsample=Opts.SUBSAMPLE)

    loader = DataLoader(dataset,
                        batch_size=Opts.BATCH_SIZE,
                        sampler=sampler,
                        collate_fn=preproc.collate_fn,
                        num_workers=Opts.NUM_WORKERS)

    if Opts.USE_BAG:
        embedder = DummyEmbbeder(num_embeddings=X.shape[1] + 1,
                                 embedding_dim=Opts.EMBEDDING_DIM)
    else:
        embedder = WeightedEmbeddingBag(num_embeddings=X.shape[1] + 1,
                                        embedding_dim=Opts.EMBEDDING_DIM,
                                        padding_idx=0,
                                        sparse=Opts.SPARSE)

    model = FeedForward(embedder=embedder,
                        layer_sizes=Opts.LAYER_SIZES + [Y.shape[1]])

    loss_fn = Opts.LOSS

    if Opts.CUDA:
        model = model.cuda()

    optimizer_factory = lambda: th.optim.Adagrad(model.parameters(), lr=0.01)

    train(loader, model, optimizer_factory, loss_fn,
          Opts)


if __name__ == '__main__':
    main(DefaultOpts)