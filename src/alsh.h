#include "common.h"

#include "faiss/Index.h"
#include <map>
#include <set>

struct IndexALSH: public faiss::Index {
    IndexALSH(size_t dim, size_t L, size_t K, float r, float U, size_t m);
    void add(idx_t n, const float* data);
    void search(idx_t n, const float* data, idx_t k, float* distances, idx_t* labels) const;
    void reset();
    // void train(idx_t n, const float* data);

    std::vector<std::map<std::vector<int>, std::set<int>>> hash_tables;
    std::vector<FloatMatrix> a_vectors;
    FloatMatrix b_scalars;

    
    // Parameters:
    size_t L;
    size_t K;
    float r;
    float U;
    size_t m;
};
