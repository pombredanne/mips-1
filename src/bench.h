#ifndef BENCH_H_
#define BENCH_H_

#include "common.h"

#include "faiss/Index.h"


int bench(faiss::Index* get_trained_index(const FloatMatrix& xt));

#endif
