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

static FloatMatrix shrivastava_extend(const float* data, size_t nvecs, size_t dim, size_t m, float U) {
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

static FloatMatrix generic_extend_queries(const float* data, size_t nvecs, size_t dim, size_t m) {
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

static FloatMatrix neyshabur_extend(const float* data, size_t nvecs, size_t dim) {
    FloatMatrix data_matrix;
    data_matrix.resize(nvecs, dim + 1);
    for (size_t i = 0; i < nvecs; i++) {
        memcpy(data_matrix.row(i), data + i * dim, dim * sizeof(float));
    }

    double maxnorm = 0;
    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        maxnorm = std::max(maxnorm, sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), dim)));
    }

    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        scale(data_matrix.row(i), maxnorm, dim);

        float norm_sqr = faiss::fvec_norm_L2sqr(data_matrix.row(i), dim);
        data_matrix.at(i, dim) = sqrt(1 - norm_sqr);
    }
    return data_matrix;
}

MipsAugmentation::MipsAugmentation(size_t dim, size_t m):
    dim(dim), m(m) {}

MipsAugmentationShrivastava::MipsAugmentationShrivastava(size_t dim, size_t m, float U):
    MipsAugmentation(dim, m), U(U) {}

FloatMatrix MipsAugmentationShrivastava::extend(const float* data, size_t nvecs) {
    return shrivastava_extend(data, nvecs, dim, m, U);
}

FloatMatrix MipsAugmentationShrivastava::extend_queries(const float* data, size_t nvecs) {
    return generic_extend_queries(data, nvecs, dim, m);
}

MipsAugmentationNeyshabur::MipsAugmentationNeyshabur(size_t dim):
    MipsAugmentation(dim, 1) {}

FloatMatrix MipsAugmentationNeyshabur::extend(const float* data, size_t nvecs) {
    return neyshabur_extend(data, nvecs, dim);
}

FloatMatrix MipsAugmentationNeyshabur::extend_queries(const float* data, size_t nvecs) {
    return generic_extend_queries(data, nvecs, dim, m);
}

MipsAugmentationNone::MipsAugmentationNone(size_t dim):
    MipsAugmentation(dim, 0) {}

FloatMatrix MipsAugmentationNone::extend(const float* data, size_t nvecs) {
    FloatMatrix data_matrix;
    data_matrix.resize(nvecs, dim);
    memcpy(data_matrix.data.data(), data, nvecs * dim * sizeof(float));

    double maxnorm = 0;
    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        maxnorm = std::max(maxnorm, sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), dim)));
    }

    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        scale(data_matrix.row(i), maxnorm, dim);
    }
    return data_matrix;
}

FloatMatrix MipsAugmentationNone::extend_queries(const float* data, size_t nvecs) {
    return generic_extend_queries(data, nvecs, dim, m);
}
