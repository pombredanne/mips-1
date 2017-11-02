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
    float maximum_norm;

    
    // Parameters:
    size_t L;
    size_t K;
    float r;
    float U;
    size_t m;

    void initialize_random_data(std::vector<FloatMatrix> &a_vectors, FloatMatrix &b_scalars, size_t d);
    void hash_vectors(FloatMatrix& data, 
            std::vector<std::map<std::vector<int>, std::set<int>>>& hash_tables, 
            std::vector<FloatMatrix>& a_vectors, 
            FloatMatrix& b_scalars, size_t d);
    int dot_product_hash(const float* a, const float* x, const float b, size_t d) const;
    size_t answer_query(float *query, 
            const std::vector<FloatMatrix>& a_vectors, 
            const FloatMatrix& b_scalars, 
            const std::vector<std::map<std::vector<int>, std::set<int>>> hash_tables,
               size_t d) const;
};
