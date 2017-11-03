#include "../src/common.h"
#include "../src/bench.h"
#include "../src/kmeans.h"

size_t m;
float U;
size_t layers_count;
size_t opened_trees;

faiss::Index* get_trained_index(const FloatMatrix& xt) {
    faiss::Index* index = new IndexHierarchicKmeans(xt.vector_length, m, layers_count, opened_trees, U);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Arguments missing, terminating.\n");
    } else {
        m = atoi(argv[1]);
        layers_count = atoi(argv[2]);
        sscanf(argv[3], "%f", &U);
        opened_trees = atoi(argv[4]);

        faiss::Index* index = bench_train(get_trained_index);
        bench_add(index);

        for (int i = 4; i < argc; i++) {
            printf("Querying using opened_trees = %d\n", atoi(argv[i]));
            ((IndexHierarchicKmeans*) index)->opened_trees = atoi(argv[i]);
            bench_query(index);
        }
    }
}
