#include "../src/common.h"
#include "../src/bench.h"
#include "../src/quantization.h"



faiss::Index* get_trained_index(const FloatMatrix& xt) {
    faiss::Index* index = new IndexSubspaceQuantization(xt.vector_length, 8, 10000);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main() {
    bench(get_trained_index);
}
