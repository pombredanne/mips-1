#ifndef MIPS_ALSH_H
#define MIPS_ALSH_H

#include "common.h"
#include "../faiss/Index.h"


struct IndexALSH: public faiss::Index {
    IndexALSH(size_t L, size_t K, size_t d,
              float r=10.f, float U=0.9f, size_t m=3);
    void train(idx_t n, const float* data);
    void add(idx_t n, const float* data);
    void search(idx_t n, const float* data, idx_t k,
                float* distances, idx_t* labels);
    void reset();
    void expand(FloatMatrix& data, std::vector<float>& norms, bool queries);

    typedef std::map<std::vector<size_t>, std::set<size_t>> hashtable_t;

    size_t L_;
    size_t K_;
    size_t d_;
    float r_;
    float U_;
    size_t m_;

    std::vector<hashtable_t> hashtables_;
    std::vector<FloatMatrix> a_vectors_;
    std::vector<std::vector<float>> b_scalars_;

    FloatMatrix train_data_;
};


#endif //MIPS_ALSH_H
