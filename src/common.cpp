#include "common.h"

#include "../faiss/Clustering.h"
#include "../faiss/utils.h"


kmeans_result perform_kmeans(const FlatMatrix<float>& matrix, size_t k) {
    kmeans_result kr;
    kr.centroids.resize(k, matrix.vector_length);
    kr.assignments.resize(matrix.vector_count());

    faiss::kmeans_clustering(
        matrix.vector_length, 
        matrix.vector_count(), 
        k, 
        matrix.data.data(), 
        kr.centroids.data.data()
    );

    // TODO: I think the assignments could be somehow taken out from faiss
    // - it stores them while computing centroids anyway. I don't see any API
    // to do that though...
    
    // Brute force assignment - for each vector, find closest centroid.
    for (size_t i = 0; i < matrix.vector_count(); i++) {
        float best = std::numeric_limits<float>::max();

        for (size_t c = 0; c < k; c++) {
            float dist = faiss::fvec_L2sqr(
                    matrix.row(i), 
                    kr.centroids.row(c),
                    matrix.vector_length);
            if (dist < best) {
                kr.assignments[i] = c;
                best = dist;
            }
        }
    }
    return kr;
}

void scale(float* vec, float alpha, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] /= alpha;
    }
}

float euclidean_norm(const float* vec, size_t d) {
    float norm = faiss::fvec_inner_product(vec, vec, d);
    return sqrtf(norm);
}

float euclidean_norm(const std::vector<float> &vec) {
    return euclidean_norm(vec.data(), vec.size());
}

int dot_product_hash(float *a, float* x, float b, float r, size_t d) {
    float mult_result = faiss::fvec_inner_product(a, x, d);
    return (int) floor((mult_result + b) / r);
}

int dot_product_hash(std::vector<float> &a, std::vector<float> &x, float b, float r) {
    assert(a.size() == x.size());
    return dot_product_hash(a.data(), x.data(), b, r, a.size());
}

float max_value(std::vector<float> &vec) {
    float maximum = std::numeric_limits<float>::min();
    for (auto v_i: vec) {
        if (v_i > maximum) {
            maximum = v_i;
        }
    }

    return maximum;
}

float randn() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> d(0,1);

    return d(gen);
}

float uniform(float low, float high) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> d(low, high);

    return d(gen);
}
