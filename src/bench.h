#ifndef BENCH_H_
#define BENCH_H_

#include "common.h"

#include "../faiss/Index.h"


faiss::Index* bench_train(faiss::Index* get_trained_index(const FloatMatrix& xt));
void bench_add(faiss::Index* index);
void bench_query(faiss::Index* index);

#endif
