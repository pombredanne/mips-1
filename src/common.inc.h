#include "common.h"

#include <fstream>
#include <iostream>


template <typename T>
FlatMatrix<T> load_file(std::string filename) {
    FlatMatrix<T> matrix;
    std::ifstream infile(filename);
    size_t cnt, dim;
    infile >> cnt >> dim;
    matrix.resize(cnt, dim);
    for (auto& cell: matrix.data) {
        infile >> cell;
    }
    return matrix;
}

template <typename T>
size_t FlatMatrix<T>::vector_count() const {
    return data.size() / vector_length;
}

template <typename T>
T& FlatMatrix<T>::at(size_t vec, size_t ind) {
    return data.at(vec * vector_length + ind);
}

template <typename T>
T FlatMatrix<T>::at(size_t vec, size_t ind) const {
    return data.at(vec * vector_length + ind);
}

template <typename T>
void FlatMatrix<T>::print() const {
    for (size_t vec = 0; vec < vector_count(); vec++) {
        for (size_t i = 0; i < vector_length; i++) {
            std::cout << at(vec, i) << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void FlatMatrix<T>::resize(size_t cnt, size_t dim) {
    data.resize(cnt * dim);
    vector_length = dim;
}
