#include "../src/common.h"
#include "../src/bench.h"
#include "../src/alsh.h"

size_t m = 3;
int augtype = 2;
float U = 1.0;

faiss::Index* get_trained_index(const FloatMatrix& xt) {
    MipsAugmentation* aug;
    size_t dim = xt.vector_length;
    switch (augtype) {
    case 0: aug = new MipsAugmentationNeyshabur(dim); break;
    case 1: aug = new MipsAugmentationShrivastava(dim, m, U); break;
    case 2: aug = new MipsAugmentationNone(dim); break;
    default: exit(1);
    }
    faiss::Index* index = new IndexALSH(dim, 6, 32, 2.5, aug);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main(int argc, char **argv) {
    faiss::Index* index = bench_train(get_trained_index);
    bench_add(index);
    bench_query(index);
}
