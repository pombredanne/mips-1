#include "../src/common.h"
#include "../src/bench.h"

#include "../faiss/AutoTune.h"

std::string nprobe_arg;

faiss::Index* get_trained_index(const FloatMatrix& xt) {
    faiss::Index* index = faiss::index_factory(xt.vector_length, "IVF4096,Flat", faiss::METRIC_INNER_PRODUCT);
    faiss::ParameterSpace params;
    std::string nprobe_arg_str = "nprobe=" + nprobe_arg;
    params.set_index_parameters (index, nprobe_arg_str.c_str());
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Arguments missing, terminating.\n");
    } else {
        nprobe_arg = argv[1];
        faiss::Index* index = bench_train(get_trained_index);
        bench_add(index);
        bench_query(index);
    }
}
