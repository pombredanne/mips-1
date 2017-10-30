#include "common.h"

#include "../faiss/Clustering.h"
#include "../faiss/IndexFlat.h"
#include "../faiss/utils.h"


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
