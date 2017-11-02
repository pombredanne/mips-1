#include "../src/common.h"
#include "../src/bench.h"
#include "../src/alsh.h"

faiss::Index* get_trained_index(const FloatMatrix& xt) {
    faiss::Index* index = new IndexALSH(xt.vector_length, 6, 32, 10.0, 0.9, 2);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main(int argc, char **argv) {
	faiss::Index* index = bench_train(get_trained_index);
	bench_add(index);
	bench_query(index);
}
