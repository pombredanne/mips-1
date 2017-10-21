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
const size_t T = 5;


#define time_report_printf(msg) { print_time_difference(start.tv_sec, start.tv_nsec); \
                                  printf(msg); \
                                  clock_gettime(CLOCK_REALTIME, &start); \
}

struct sort_pred {
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) {
        return left.second > right.second;
    }
};

int dot_product_hash(vector<float>& a, vector<float>& x, float b) {
    assert(a.size() == x.size());

    //TODO: Use fast inner product?
    float mult_result = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
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
float euclidean_norm(vector<float>& vec) {
    float sum = 0;

    for (auto v_i: vec) {
        sum += v_i * v_i;
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

void scale_(vector<float>& vec, float alpha) {
    for (size_t i = 0; i < vec.size(); i++) {
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

void expand(vector<vector<float> >& vec, size_t num, vector<float>& norms, float max_norm, size_t length, bool queries) {
    #pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        scale_(vec[i], U / max_norm);

        float vec_norm = euclidean_norm(vec[i]);
        norms[i] = vec_norm;

        for (auto j = length; j < length + m; j++) {
            if (!queries) {
                vec[i][j] = vec_norm;
                vec_norm *= vec_norm;
            } else {
                vec[i][j] = 0.5f;
            }
        }

        for (auto j = length + m; j < length + 2*m; j++) {
            if (!queries) {
                vec[i][j] = 0.5f;
            } else {
                vec[i][j] = vec_norm;
                vec_norm *= vec_norm;
            }
        }
    }
}

void main_alsh() {
    omp_set_num_threads(4);

    size_t number_of_vectors;
    size_t vector_length;

    timespec start{};
    clock_gettime(CLOCK_REALTIME, &start);

    time_report_printf("Setting up data structures... ");

    vector<vector<float>> db = load_db("input", &number_of_vectors, &vector_length);

    vector<float> vector_norms(number_of_vectors);
    vector<map<vector<int>, set<int>>> hash_tables(L);

    vector<vector<vector<float>>> a_vectors(L);
    vector<vector<float>> b_scalars(L);

    // Initialize random data for all L hash tables
    for (size_t l = 0; l < L; l++) {
        a_vectors[l].resize(K);
        b_scalars[l].resize(K);

        // Generate different random data for each time a vector is hashed
        for (size_t k = 0; k < K; k++) {
            a_vectors[l][k].resize(vector_length + 2*m);

            for (size_t i = 0; i < vector_length + 2*m; i++) {
                a_vectors[l][k][i] = randn();
            }

            auto b = uniform(0, r);
            b_scalars[l][k] = b;
        }
    }

    time_report_printf("Computing norms... ");

    #pragma omp parallel for
    for (size_t i = 0; i < number_of_vectors; i++) {
        vector_norms[i] = euclidean_norm(db[i]);
    }

    time_report_printf("Scaling vectors, computing norms and extending vectors... ");

    auto maximum_norm = max_value(vector_norms);

    expand(db, number_of_vectors, vector_norms, maximum_norm, vector_length, false);
    
    time_report_printf("Hashing vectors... ");

    #pragma omp parallel for
    for (size_t l = 0; l < L; l++) {
        for (size_t i = 0; i < number_of_vectors; i++) {
            vector<int> hash_vector(K);

            for (size_t k = 0; k < K; k++) {
                hash_vector[k] = dot_product_hash(a_vectors[l][k], db[i], b_scalars[l][k]);
            }

            hash_tables[l][hash_vector].insert(i);
        }
    }

    time_report_printf("Finished hashing, printing hash tables...");

    // Print the contents of hash tables
    int hash_table_index = 0;
    for (auto& it: hash_tables) {
        printf("\nHash table %d:\n", hash_table_index);

        for (map<vector<int>, set<int> >::iterator it2 = it.begin(); it2 != it.end(); it2++) {

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

    time_report_printf("Loading queries...");

    size_t number_of_queries;
    size_t query_length;
    vector<vector<float>> queries = load_db("queries", &number_of_queries, &query_length);

    vector<float> query_norm(number_of_queries);

    time_report_printf("Scaling queries, computing norms and extending queries...");

    expand(queries, number_of_queries, query_norm, maximum_norm, query_length, true);

    time_report_printf("Matching queries...");

    //#pragma omp parallel for
    for (size_t i = 0; i < number_of_queries; i++) {
        printf("\nQuery %zu - best matches: ", i);
        map<int, int> score;
        for (size_t l = 0; l < L; l++) {
            vector<int> hash_vector(K);
            for (size_t k = 0; k < K; k++) {
                hash_vector[k] = dot_product_hash(a_vectors[l][k], queries[i], b_scalars[l][k]);
            }

            auto& current_hash_table = hash_tables[l];
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
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);
}

