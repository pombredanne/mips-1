#ifndef COMMON_H_
#define COMMON_H_

#include <vector>
#include <cstdlib>
#include <string>


struct FlatMatrix {
	std::vector<float> data;
	size_t vector_length;

	size_t vector_count() const;
};

FlatMatrix load_file(std::string filename);


#endif
