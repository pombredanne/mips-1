#include "common.h"

#include "../faiss/Clustering.h"
#include "../faiss/IndexFlat.h"
#include "../faiss/utils.h"

#include <algorithm>


kmeans_result perform_kmeans(const FlatMatrix<float>& matrix, size_t k) {
    kmeans_result kr;
    kr.centroids.resize(k, matrix.vector_length);
    kr.assignments.resize(matrix.vector_count());
    std::vector<float> dist(matrix.vector_count());
    std::vector<faiss::Index::idx_t> assignments(matrix.vector_count());

    faiss::kmeans_clustering(
        matrix.vector_length, 
        matrix.vector_count(), 
        k, 
        matrix.data.data(), 
        kr.centroids.data.data()
    );

    faiss::IndexFlatL2 index(matrix.vector_length);
    index.add(kr.centroids.vector_count(), kr.centroids.data.data());
    index.search(matrix.vector_count(), matrix.data.data(), 1,
            dist.data(), assignments.data());

    for (size_t i = 0; i < matrix.vector_count(); i++) {
        kr.assignments[i] = assignments[i];
    }
    return kr;
}

void scale(float* vec, float alpha, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] /= alpha;
    }
}

FloatMatrix shrivastava_extend(const float* data, size_t nvecs, size_t dim, size_t m, float U) {
    FloatMatrix data_matrix;
    data_matrix.resize(nvecs, dim + m);
    for (size_t i = 0; i < nvecs; i++) {
        memcpy(data_matrix.row(i), data + i * dim, dim * sizeof(float));
    }

    double maxnorm = 0;
    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        maxnorm = std::max(maxnorm, sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), dim)));
    }

    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        scale(data_matrix.row(i), maxnorm / U, dim);

        float vec_norm = sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), dim));
        for (size_t j = dim; j < dim + m; j++) {
            data_matrix.at(i, j) = 0.5 - vec_norm;
            vec_norm *= vec_norm;
        }
    }
    return data_matrix;
}

FloatMatrix shrivastava_extend_queries(const float* data, size_t nvecs, size_t dim, size_t m) {
    FloatMatrix queries;
    queries.resize(nvecs, dim + m);
    for (size_t i = 0; i < nvecs; i++) {
        memcpy(queries.row(i), data + i * dim, dim * sizeof(float));
    }
    for (size_t i = 0; i < queries.vector_count(); i++) {
        float qnorm = sqrt(faiss::fvec_norm_L2sqr(queries.row(i), dim));
        scale(queries.row(i), qnorm, dim);

        for (size_t j = dim; j < dim + m; j++) {
            queries.at(i, j) = 0.0;
        }
    }
    return queries;
}
