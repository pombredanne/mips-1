#include "common.h"

#include <fstream>
#include <iostream>
#include <cassert>


template <typename T>
FlatMatrix<T> load_text_file(std::string filename) {
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
FlatMatrix<T> load_vecs (std::string filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (infile.fail()) {
        std::cout << "Failed to open file." << std::endl;
        exit(1);
    }
    uint32_t dim;
    infile.read((char*) &dim, sizeof(dim));

    infile.seekg(0, std::ios::end);
    size_t fsz = infile.tellg();
    infile.seekg(0, std::ios::beg);

    size_t row_size = sizeof(T) * dim + sizeof(dim);
    size_t n = fsz / row_size;
    if(fsz != n * row_size){
        std::cout << "Wrong file size" << std::endl;
        exit(1);
    }
    FlatMatrix<T> result;
    result.resize(n, dim);
    for(size_t i = 0; i < n; i++){
        uint32_t rowsz;
        infile.read((char*) &rowsz, sizeof(rowsz));
        assert(rowsz == dim);
        infile.read((char*) result.row(i), sizeof(T) * rowsz);
    }
    return result;
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
const T& FlatMatrix<T>::at(size_t vec, size_t ind) const {
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
T* FlatMatrix<T>::row(size_t num) {
    return &at(num, 0);
}

template <typename T>
const T* FlatMatrix<T>::row(size_t num) const {
    return &at(num, 0);
}

template <typename T>
void FlatMatrix<T>::resize(size_t cnt, size_t dim) {
    data.resize(cnt * dim);
    vector_length = dim;
}
