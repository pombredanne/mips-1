#ifndef COMMON_H_
#define COMMON_H_

#include <vector>
#include <cstdlib>
#include <string>


template <typename T>
struct FlatMatrix {
    std::vector<T> data;
    size_t vector_length;

    T& at(size_t vec, size_t ind);
    const T& at(size_t vec, size_t ind) const;

    T* row(size_t num);
    const T* row(size_t num) const;

    size_t vector_count() const;
    void print() const;
    void resize(size_t cnt, size_t dim);
};

typedef FlatMatrix<float> FloatMatrix;

template <typename T>
FlatMatrix<T> load_text_file(std::string filename);

template<typename T>
FlatMatrix<T> load_vecs (std::string filename);

struct kmeans_result {
    FloatMatrix centroids;
    std::vector<size_t> assignments;
};

kmeans_result perform_kmeans(const FloatMatrix& matrix, size_t k);


void scale(float* vec, float alpha, size_t size);


#include "common.inc.h"

#endif
