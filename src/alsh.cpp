#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <map>
#include <set>
#include <omp.h>


using namespace std;

// Number of additional vector components.
const int m = 2;

// Number of hash tables.
const int L = 6;

// Hash function parameter, not larger than maximum random number.
const int r = 10;

// Scaling coefficient.
const float U = 0.9;

// Number of hash function in one hash table.
const int K = 32;


#define time_report_printf(msg) { print_time_difference(start.tv_sec, start.tv_nsec); \
                                  printf(msg); \
                                  clock_gettime(CLOCK_REALTIME, &start); \
}


int dot_product_hash(vector<float> &a, vector<float> &x, float &b) {
    assert(a.size() == x.size());

    float mult_result = 0.0f;
    for (unsigned i = 0; i < a.size(); i++) {
         mult_result += a[i] * x[i];
    }

    auto result = (int) (floor((mult_result + b) / (float) r));

    return result;
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

    double sec_diff_double = (double) nsec_diff / 1000 * 1000 * 1000.0;
    sec_diff_double += sec_diff;

    printf("%.3lf s\n", sec_diff_double);
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


float euclidean_norm(vector<float> &vec) {
    float sum = 0;

    for (auto &v_i : vec) {
        sum += v_i * v_i;
    }

    return sqrt(sum);
}

float max_value(vector<float> &vec) {
    float maximum = 0.0f;
    for (auto &v_i : vec) {
        if (v_i > maximum) {
            maximum = v_i;
        }
    }

    return maximum;
}

void scale_(vector<float> &vec, float alpha) {
    for (auto i=0; i<vec.size(); i++) {
        vec[i] *= alpha;
    }
}

vector<vector<float>> load_db(string fname,
                              uint64_t* number_of_vectors, uint64_t* vector_length) {

    FILE *database;
    database = fopen(fname.c_str(), "r");

    fscanf(database, "%lu", number_of_vectors);
    fscanf(database, "%lu", vector_length);

    printf("Setting up data structures...\t");

    vector<vector<float> > db((uint64_t) number_of_vectors);

    for (int i = 0; i < *number_of_vectors; i++) {
        db[i].resize(*vector_length + 2*m);
    }

    for (int i = 0; i < *number_of_vectors; i++) {
        for (int j = 0; j < *vector_length; j++) {
            fscanf(database, "%f", &db[i][j]);
        }
    }
    fclose(database);

    return db;
}


int main() {
    omp_set_num_threads(4);

    uint64_t number_of_vectors;
    uint64_t vector_length;

    timespec start{};
    clock_gettime(CLOCK_REALTIME, &start);

    vector<vector<float> > db = load_db("input", &number_of_vectors, &vector_length);

    vector<float> vector_norms((uint64_t) number_of_vectors);
    vector<map<vector<int>, set<int> > > hash_tables(L);

    vector<vector<vector<float> > > a_vectors(L);
    vector<vector<float> > b_scalars(L);

    // Initialize random data for all L hash tables
    for (int l = 0; l < L; l++) {
        a_vectors[l].resize(K);
        b_scalars[l].resize(K);

        // Generate different random data for each time a vector is hashed
        for (int k = 0; k < K; k++) {
            a_vectors[l][k].resize(vector_length + 2*m);

            for (int i = 0; i < vector_length + 2*m; i++) {
                a_vectors[l][k][i] = randn();
            }

            auto b = uniform(0, r);
            b_scalars[l][k] = b;
        }
    }

    time_report_printf("Computing norms... ");

    #pragma omp parallel for
    for (int i = 0; i < number_of_vectors; i++) {
        vector_norms[i] = euclidean_norm(db[i]);
    }

    time_report_printf("Scaling vectors, computing norms and extending vectors... ");

    auto maximum_norm = max_value(vector_norms);

    #pragma omp parallel for
    for (int i = 0; i < number_of_vectors; i++) {
        scale_(db[i], U / maximum_norm);

        float vec_norm = euclidean_norm(db[i]);
        vector_norms[i] = vec_norm;

        for (auto j = vector_length; j < vector_length + m; j++) {
            db[i][j] = vec_norm;
            vec_norm *= vec_norm;
        }

        for (auto j = vector_length + m; j < vector_length + 2*m; j++) {
            db[i][j] = 0.5f;
        }
    }
    
    time_report_printf("Hashing vectors... ");

    #pragma omp parallel for
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < number_of_vectors; i++) {
            vector<int> hash_vector(K);

            for (int k = 0; k < K; k++) {
                hash_vector[k] = dot_product_hash(a_vectors[l][k], db[i], b_scalars[l][k]);
            }

            hash_tables[l][hash_vector].insert(i);
        }
    }

    time_report_printf("Finished hashing, printing hash tables...");

    // Print the contents of hash tables
    int hash_table_index = 0;
    for (auto &it : hash_tables) {
        printf("Hash table %d:\n", hash_table_index);

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

    time_report_printf("Loading queries");

    uint64_t number_of_queries;
    uint64_t query_length;
    vector<vector<float>> queries = load_db("queries", &number_of_queries, &query_length);

    vector<set<int> > matched_vectors(number_of_queries);
    vector<float> query_norm(number_of_queries);

    time_report_printf("Scaling queries, computing norms and extending queries...");

    // todo, can we merge this with similar code above?
    #pragma omp parallel for
    for (int i = 0; i < number_of_queries; i++) {
        scale_(queries[i], U / maximum_norm);

        query_norm[i] = euclidean_norm(queries[i]);
        float q_norm = query_norm[i];

        for (auto j = query_length; j < query_length + m; j++) {
            queries[i][j] = 0.5f;
        }

        for (auto j = query_length + m; j < query_length + 2*m; j++) {
            queries[i][j] = q_norm;
            q_norm *= q_norm;
        }

    }

    time_report_printf("Matching queries... ");

    #pragma omp parallel for
    for (int i = 0; i < number_of_queries; i++) {
        for (int l = 0; l < L; l++) {

            vector<int> hash_vector(K);
            for (int k = 0; k < K; k++) {
                hash_vector[k] = dot_product_hash(a_vectors[l][k], queries[i], b_scalars[l][k]);
            }

            auto& current_hash_table = hash_tables[l];
            auto& current_bucket = current_hash_table[hash_vector];
            matched_vectors[i].insert(current_bucket.begin(), current_bucket.end());
        }
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);

    for (auto i = 0; i < number_of_queries; i++) {
        printf("Query %d matches vectors: ", i);

        for (auto &it: matched_vectors) {
            printf(" %d", it);
        }

        printf("\n");
    }
}

