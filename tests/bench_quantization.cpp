#include "../src/common.h"
#include "../src/bench.h"
#include "../src/quantization.h"

size_t subspace_count;
size_t centroid_count;

faiss::Index* get_trained_index(const FloatMatrix& xt) {
    faiss::Index* index = new IndexSubspaceQuantization(xt.vector_length, subspace_count, centroid_count);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Arguments missing, terminating.\n");
    } else {
        subspace_count = atoi(argv[1]);
        centroid_count = atoi(argv[2]);
        faiss::Index* index = bench_train(get_trained_index);
        bench_add(index);
        bench_query(index);
    }
}
