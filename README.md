This library is based on ideas from these articles:
 * Clustering is efficient for approximate maximum inner product search (Auvolat, Larochelle, Chandar, Vincent, Bengio)
 * Quantization based fast inner product search (Guo, Kumar, Choromanski, Simcha)
 * Asymmetric LSH for sublinear maximum inner product search (Shrivastava, Li)


# Compilation
You'll need to setup a proper `makefile.inc` in the `faiss` directory, see `faiss/INSTALLATION.md` for details. It should be 
as simple as commenting / uncommenting two lines of code.

To build the python module you'll also need to set proper `PYTHONCFLAGS` in the same `makefile.inc`, eg:
`PYTHONCFLAGS=-I/home/<user>/anaconda3/include/python3.6m -I/home/<user>/anaconda3/lib/python3.6/site-packages/numpy/core/include`

Once this is set up, go to `faiss` directory and run `make py`. 
Confirm that the installtion was sucessful by running

```bash
export PYTHONPATH=/path/to/faiss/root/directory
python -c "import faiss"
```
