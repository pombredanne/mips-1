#include "common.h"
#include "alsh.h"
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

// Number of additional vector components.
const size_t m = 2;

// Number of hash tables.
const size_t L = 6;

// Hash function parameter, not larger than maximum random number.
const float r = 10.0;

// Scaling coefficient.
const float U = 0.9;

// Number of hash function in one hash table.
const size_t K = 32;

// Number of best vectors returned for each query
const size_t T = 1;

#define time_report_printf(msg) { print_time_difference(start.tv_sec, start.tv_nsec); \
                                  printf(msg); \
                                  clock_gettime(CLOCK_REALTIME, &start); \
}

struct sort_pred {
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) {
        return left.second > right.second;
    }
};

int dot_product_hash(const float* a, const float* x, const float b, size_t d) {
    //assert(a.size() == x.size());

    //TODO: Use fast inner product?
    float mult_result = 0.0f;
    for (size_t i = 0; i < d; i++) {
         mult_result += a[i] * x[i];
    }

    return (int) floor((mult_result + b) / r);
}

void print_time_difference(long time_sec, long time_nsec) {
    timespec temp {};
    clock_gettime(CLOCK_REALTIME, &temp);

    long sec_diff = temp.tv_sec - time_sec;
    long nsec_diff = temp.tv_nsec - time_nsec;

    if (nsec_diff < 0) {
        nsec_diff += 1000 * 1000 * 1000;
        sec_diff--;
    }

    double sec_diff_double = nsec_diff / (1000.0 * 1000.0 * 1000.0);
    sec_diff_double += sec_diff;

    printf(" %.3lf s\n", sec_diff_double);
}

float randn() {
    static random_device rd;
    static mt19937 gen(rd());
    static normal_distribution<float> d(0,1);

    return d(gen);
}

float uniform(float low, float high) {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<float> d(low, high);

    return d(gen);
}

// TODO: Replace with fast inner product?
float euclidean_norm(float* vec, size_t d) {
    float sum = 0;

    for (size_t i = 0;i<d;i++) {
        sum += vec[i]* vec[i];
    }
    return sqrt(sum);
}

float max_value(vector<float>& vec) {
    float maximum = numeric_limits<float>::min();
    for (auto v_i: vec) {
        if (v_i > maximum) {
            maximum = v_i;
        }
    }
    return maximum;
}

void scale_(float* vec, float alpha, size_t d) {
    for (size_t i = 0; i < d; i++) {
        vec[i] *= alpha;
    }
}

vector<vector<float>> load_db(string fname,
                              size_t* number_of_vectors, size_t* vector_length) {

    FILE *database;
    database = fopen(fname.c_str(), "r");

    fscanf(database, "%lu", number_of_vectors);
    fscanf(database, "%lu", vector_length);

    vector<vector<float> > db(*number_of_vectors);

    for (size_t i = 0; i < *number_of_vectors; i++) {
        db[i].resize(*vector_length + 2*m);
    }

    for (size_t i = 0; i < *number_of_vectors; i++) {
        for (size_t j = 0; j < *vector_length; j++) {
            fscanf(database, "%f", &db[i][j]);
        }
    }
    fclose(database);

    return db;
}

void expand(FloatMatrix &vec, vector<float>& norms, float max_norm, size_t length, bool queries) {
    #pragma omp parallel for
    for (size_t i = 0; i < vec.vector_count(); i++) {
        scale_(vec.row(i), U / max_norm, length);

        float vec_norm = euclidean_norm(vec.row(i), length);
        norms[i] = vec_norm;

        for (auto j = length; j < length + m; j++) {
            if (!queries) {
                vec.at(i,j) = vec_norm;
                vec_norm *= vec_norm;
            } else {
                vec.at(i,j) = 0.5f;
            }
        }
    }
}

void initialize_random_data(vector<FloatMatrix> &a_vectors, FloatMatrix &b_scalars, size_t d) {
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

void hash_vectors(FloatMatrix &data,vector<map<vector<int>, set<int>>> &hash_tables, vector<FloatMatrix> &a_vectors, FloatMatrix &b_scalars, size_t d){
#pragma omp parallel for
    for (size_t l = 0; l < L; l++) {
        for (size_t i = 0; i < data.vector_count(); i++) {
            vector<int> hash_vector(K);

            for (size_t k = 0; k < K; k++) {
                hash_vector[k] =  dot_product_hash(a_vectors[l].row(k), data.row(i), b_scalars.at(l,k),d);
            }

            hash_tables[l][hash_vector].insert(i);
        }
    }
}
void print_hash_tables(vector<map<vector<int>, set<int>>> &hash_tables)
{
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
size_t answer_query(float *query, const vector<FloatMatrix> &a_vectors, const FloatMatrix &b_scalars, const vector<map<vector<int>, set<int>>> hash_tables, size_t d)
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
    sort(score_vector.begin(), score_vector.end(), sort_pred());
    size_t T_count = 0;
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
        vector_norms[i] = euclidean_norm(data_matrix.row(i),d);
    }
    // TODO why not STL? why not inside expand?
    maximum_norm = max_value(vector_norms);

    expand(data_matrix, vector_norms, maximum_norm, d, false);

    hash_vectors(data_matrix,hash_tables, a_vectors, b_scalars, d);

}

void IndexALSH::search(idx_t n, const float* data, idx_t k, float* distances, idx_t* labels) const {
    FloatMatrix queries;
    queries.resize(n, d + m);
    for (idx_t i = 0; i < n; i++) {
        memcpy(queries.row(i), data + i * d, d * sizeof(float));
    }
    vector<float> query_norm(d);
    expand(queries, query_norm, maximum_norm, d, true);
    for (size_t q = 0; q < queries.vector_count(); q++) {
        size_t ans = answer_query(queries.row(q),a_vectors,b_scalars,hash_tables,d);
        labels[q * k] = ans;
        for (idx_t j = 1; j < k; j++) {
            labels[q * k + j] = -1;
        }
    }

    // TODO
}

void main_alsh() {
    omp_set_num_threads(4);

    size_t d;

    timespec start{};
    clock_gettime(CLOCK_REALTIME, &start);

    time_report_printf("Setting up data structures... ");

    FloatMatrix data = load_text_file<float>("input");
    FloatMatrix queries = load_text_file<float>("queries");
    vector<float> vector_norms(data.vector_count());
    vector<map<vector<int>, set<int>>> hash_tables(L);
    vector<FloatMatrix> a_vectors(L);
    FloatMatrix b_scalars;

    // Initialize random data for all L hash tables
    initialize_random_data(a_vectors,b_scalars,d);

    time_report_printf("Computing norms... ");

    #pragma omp parallel for
    for (size_t i = 0; i < data.vector_count(); i++) {
        vector_norms[i] = euclidean_norm(data.row(i),d);
    }

    time_report_printf("Scaling vectors, computing norms and extending vectors... ");

    auto maximum_norm = max_value(vector_norms);

    expand(data, vector_norms, maximum_norm, d, false);
    
    time_report_printf("Hashing vectors... ");

    hash_vectors(data,hash_tables, a_vectors, b_scalars, d);

    time_report_printf("Finished hashing, printing hash tables...");

    // Print the contents of hash tables

    print_hash_tables(hash_tables);
    time_report_printf("Loading queries...");
//    vector<vector<float>> queries = load_db("queries", &number_of_queries, &query_length);

    vector<float> query_norm(d);

    time_report_printf("Scaling queries, computing norms and extending queries...");

    expand(queries, query_norm, maximum_norm, d, true);

    time_report_printf("Matching queries...");

    //#pragma omp parallel for
    for (size_t i = 0; i < queries.vector_count(); i++) {
        printf("\nQuery %zu - best matches: ", i);
        answer_query(queries.row(i),a_vectors,b_scalars,hash_tables,d);
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);
}

