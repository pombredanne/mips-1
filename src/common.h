#ifndef COMMON_H_
#define COMMON_H_

#include <vector>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <map>
#include <set>
#include <algorithm>
#include <omp.h>
#include "../faiss/utils.h"


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

template<typename T>
FlatMatrix<T> from_dtype(const T* data, size_t n);

struct kmeans_result {
    FloatMatrix centroids;
    std::vector<size_t> assignments;
};

kmeans_result perform_kmeans(const FloatMatrix& matrix, size_t k);

void scale(float* vec, float alpha, size_t size);

float euclidean_norm(const float* vec, size_t d);
float euclidean_norm(const std::vector<float>& vec);

float max_value(std::vector<float>& vec);

float randn();

float uniform(float low, float high);

int dot_product_hash(float *a, float* x, float b, float r, size_t d);
int dot_product_hash(std::vector<float>& a, std::vector<float>& x, float b, float r);

struct sort_pred {
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) {
        return left.second > right.second;
    }
};


#include "common.inc.h"

#endif
