#include "../src/common.h"
#include "../src/bench.h"
#include "../src/kmeans.h"



faiss::Index* get_trained_index(const FloatMatrix& xt) {
    faiss::Index* index = new IndexHierarchicKmeans(xt.vector_length, 3, 3, 100);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main() {
	bench(get_trained_index);
}
