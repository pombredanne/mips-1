#include "alsh.h"
#include "common.h"

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
#include "../faiss/Clustering.h"

using namespace std;

bool sort_pred(const std::pair<int,int> &left, const std::pair<int,int> &right) {
    return left.second > right.second;
}

int IndexALSH::dot_product_hash(const float* a, const float* x, const float b, size_t d) const {
    float ax = faiss::fvec_inner_product(a, x, d);
    return floor((ax + b) / r);
}

static float randn() {
    static random_device rd;
    static mt19937 gen(rd());
    static normal_distribution<float> d(0,1);

    return d(gen);
}

static float uniform(float low, float high) {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<float> d(low, high);

    return d(gen);
}

void IndexALSH::initialize_random_data(vector<FloatMatrix> &a_vectors, FloatMatrix &b_scalars, size_t d) {
    b_scalars.resize(L,K);
    for (size_t l = 0; l < L; l++) {
        a_vectors[l].resize(K,d+2*m);

        // Generate different random data for each time a vector is hashed
        for (size_t k = 0; k < K; k++) {
            for (size_t i = 0; i < d + 2*m; i++) {
                a_vectors[l].at(k, i) = randn();
            }

            auto b = uniform(0, r);
            b_scalars.at(l,k) = b;
        }
    }
}

void IndexALSH::hash_vectors(FloatMatrix &data,vector<map<vector<int>, set<int>>> &hash_tables, vector<FloatMatrix> &a_vectors, FloatMatrix &b_scalars, size_t d){
#pragma omp parallel for
    for (size_t l = 0; l < L; l++) {
        for (size_t i = 0; i < data.vector_count(); i++) {
            vector<int> hash_vector(K);

            for (size_t k = 0; k < K; k++) {
                hash_vector[k] = dot_product_hash(a_vectors[l].row(k), data.row(i), b_scalars.at(l,k),d);
            }

            hash_tables[l][hash_vector].insert(i);
        }
    }
}

static void print_hash_tables(vector<map<vector<int>, set<int>>> &hash_tables) {
    int hash_table_index = 0;
    for (auto& it: hash_tables) {
        printf("\nHash table %d:\n", hash_table_index);

        for (map<vector<int>, set<int> >::iterator it2 = it.begin(); it2 != it.end();it2++) {
            printf("[ ");
            for (vector<int>::const_iterator it3 = it2->first.begin(); it3 != it2->first.end(); it3++) {
                printf("%d ", *it3);
            }

            printf("] -> ");
            for (set<int>::iterator it3 = it2->second.begin(); it3 != it2->second.end(); it3++) {
                printf("%d ", *it3);
            }
            printf("\n");
        }

        hash_table_index++;
    }
}

size_t IndexALSH::answer_query(float *query, const vector<FloatMatrix> &a_vectors, const FloatMatrix &b_scalars, const vector<map<vector<int>, set<int>>> hash_tables, size_t d) const
{
    map<int, int> score;
    for (size_t l = 0; l < L; l++) {
        vector<int> hash_vector(K);
        for (size_t k = 0; k < K; k++) {
            hash_vector[k] = dot_product_hash(a_vectors[l].row(k), query, b_scalars.at(l,k),d);
        }

        map<vector<int>, set<int>> current_hash_table(hash_tables[l]);
        auto& current_bucket = current_hash_table[hash_vector];

        for (auto &it: current_bucket) {
            score[it]++;
        }
    }
    vector<pair<int,int> > score_vector(score.begin(), score.end());
    sort(score_vector.begin(), score_vector.end(), sort_pred);
    size_t T_count = 0;
    size_t T = 1; // TODO...
    for (auto &it: score_vector) {
        if (T_count >= T)
            break;
        printf("%d (%d) ", it.first, it.second);
        T_count++;
    }
    return score_vector[0].first;
}

IndexALSH::IndexALSH(
        size_t dim, size_t L, size_t K, float r, float U, size_t m):
        Index(dim, faiss::METRIC_INNER_PRODUCT),
        L(L),K(K),r(r),U(U),m(m) {}

void IndexALSH::reset() {
    hash_tables.clear();
    a_vectors.clear();
    b_scalars.data.clear();
    maximum_norm = 0.0f;
}

void IndexALSH::add(idx_t n, const float* data) {
    FloatMatrix data_matrix;
    data_matrix.resize(n, d + m);
    for (idx_t i = 0; i < n; i++) {
        memcpy(data_matrix.row(i), data + i * d, d * sizeof(float));
    }
    vector<float> vector_norms(data_matrix.vector_count());
    hash_tables.resize(L);
    a_vectors.resize(L);

    initialize_random_data(a_vectors,b_scalars,d);

#pragma omp parallel for
    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        vector_norms[i] = sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i),d));
    }
    maximum_norm = *max_element(vector_norms.begin(), vector_norms.end());

    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        scale(data_matrix.row(i), maximum_norm / U, d);

        float vec_norm = sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), d));
        vector_norms[i] = vec_norm;

        for (size_t j = d; j < d + m; j++) {
            data_matrix.at(i,j) = vec_norm;
            vec_norm *= vec_norm;
        }
    }

    hash_vectors(data_matrix,hash_tables, a_vectors, b_scalars, d);
}

void IndexALSH::search(idx_t n, const float* data, idx_t k, float* distances, idx_t* labels) const {
    FloatMatrix queries;
    queries.resize(n, d + m);
    for (idx_t i = 0; i < n; i++) {
        memcpy(queries.row(i), data + i * d, d * sizeof(float));
    }
    for (size_t i = 0; i < queries.vector_count(); i++) {
        float qnorm = sqrt(faiss::fvec_norm_L2sqr(queries.row(i), d));
        scale(queries.row(i), qnorm, d);

        for (size_t j = d; j < d + m; j++) {
            queries.at(i,j) = 0.5;
        }
    }

    for (size_t q = 0; q < queries.vector_count(); q++) {
        // TODO return more than 1 result
        size_t ans = answer_query(queries.row(q),a_vectors,b_scalars,hash_tables,d);
        labels[q * k] = ans;
        for (idx_t j = 1; j < k; j++) {
            labels[q * k + j] = -1;
        }
        // TODO write distances
    }
}

