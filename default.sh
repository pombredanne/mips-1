export LD_LIBRARY_PATH=${HOME}/anaconda3/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=${HOME}/anaconda3/lib/libmkl_core.so:${HOME}/anaconda3/lib/libmkl_sequential.so
make mkl
./k-means-mips ./vectors/siftsmall_base.fvecs ./vectors/siftsmall_query.fvecs 10 3 8 0
