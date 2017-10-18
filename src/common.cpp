#include "common.h"

#include <fstream>


FlatMatrix load_file(std::string filename) {
	FlatMatrix matrix;
	std::ifstream infile(filename);
    size_t n, m;
    infile >> n >> m;
	matrix.vector_length = m;
	matrix.data.resize(n * m);
    for (auto& cell: matrix.data) {
        infile >> cell;
    }
    return matrix;
}

size_t FlatMatrix::vector_count() const {
	return data.size() / vector_length;
}
