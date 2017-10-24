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
