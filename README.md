This library is based on ideas from these articles:
 * Clustering is efficient for approximate maximum inner product search (Auvolat, Larochelle, Chandar, Vincent, Bengio)
 * Quantization based fast inner product search (Guo, Kumar, Choromanski, Simcha)
 * Asymmetric LSH for sublinear maximum inner product search (Shrivastava, Li)


# Compilation
To build the python module you'll need to set proper `PYTHONCFLAGS` in the `makefile.inc`, eg:
`PYTHONCFLAGS=-I/home/user/anaconda3/include/python3.6m -I/home/user/anaconda3/lib/python3.6/site-packages/numpy/core/include`

Then copy the `makefile.inc` to `faiss` directory and run `make.py`

Then you should export `PYTHONPATH` to point to `FAISS` directory, eg.
`export PYTHONPATH=$(pwd)/faiss`
and you should be able to run `python -c "import faiss"`
