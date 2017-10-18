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
    T at(size_t vec, size_t ind) const;
    size_t vector_count() const;
    void print() const;
    void resize(size_t cnt, size_t dim);
};


template <typename T>
FlatMatrix<T> load_file(std::string filename);


#include "common.inc.h"

#endif
