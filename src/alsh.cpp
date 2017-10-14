#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <map>
#include <set>
#include <omp.h>
#include <sys/time.h>
    
// Debug
#define D(x) x
// No debug
//#define D(x)
// OpenMP on
#define OMP(x) x
// OpenMP off
//#define OMP(x)

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

int hash_function(vector<float> a, vector<float> x, float b) {
    assert(a.size() == x.size());
    float mult_result = 0.0f;
    for (unsigned i = 0; i < a.size(); i++) {
         mult_result += a[i] * x[i];
    }
    return floor((float)(mult_result + b)/(float)r);
}

void print_time_difference(long time_sec, long time_nsec) {
    timespec temp;
    clock_gettime(CLOCK_REALTIME, &temp);
    long sec_diff = temp.tv_sec - time_sec;
    long nsec_diff = temp.tv_nsec - time_nsec;
    if (nsec_diff < 0) {
        nsec_diff += 1000000000;
        sec_diff--;
    }
    double sec_diff_double = (double)nsec_diff / (double)1000000000;
    sec_diff_double += sec_diff;
    printf("%.3lf s\n", sec_diff_double);
    
}

int main() {
    srand(time(0));
    OMP(omp_set_num_threads(4);)
    int number_of_vectors;
    int vector_length;
    FILE *database;
    database = fopen("input", "r");
    fscanf(database, "%d", &number_of_vectors);
    fscanf(database, "%d", &vector_length);

    timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    printf("Setting up data structures...\t");
    vector<vector<float> > db(number_of_vectors);
    vector<float> vector_norm(number_of_vectors);
    vector<map<vector<int>, set<int> > > hash_tables(L);
    for (int i = 0; i < number_of_vectors; i++) {
        db[i].resize(vector_length + 2*m);
    }

    // Generate random matrix for hashing.
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0,1);
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
                a_vectors[l][k][i] = d(gen);
            }
            float b = rand()%r;
            b += (float)(rand()%1000) * 0.001f;
            b_scalars[l][k] = b;
        }
    }
  
    print_time_difference(start.tv_sec, start.tv_nsec);
    printf("Loading vectors... ");  
    clock_gettime(CLOCK_REALTIME, &start);

    for (int i = 0; i < number_of_vectors; i++) {
        for (int j = 0; j < vector_length; j++) {
            fscanf(database, "%f", &db[i][j]);
        }
    }
    fclose(database);
    
    print_time_difference(start.tv_sec, start.tv_nsec);
    printf("Computing norms... ");
    clock_gettime(CLOCK_REALTIME, &start);
    
    #pragma omp parallel for
    for (int i = 0; i < number_of_vectors; i++) {
        // Compute vector's Euclidean norm.
        float sum = 0;
        for (int j = 0; j < vector_length; j++) {
             sum += db[i][j] * db[i][j];
        }
        vector_norm[i] = sqrt(sum);
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);
    
    float maximum_norm = 0.0f;
    for (int i = 0; i < number_of_vectors; i++) {
        if (vector_norm[i] > maximum_norm) {
            maximum_norm = vector_norm[i];
        }
    }
    
    printf("Scaling vectors, computing norms and extending vectors... ");
    clock_gettime(CLOCK_REALTIME, &start);
    
    #pragma omp parallel for
    for (int i = 0; i < number_of_vectors; i++) {
        for (int j = 0; j < vector_length; j++) {
            db[i][j] /= maximum_norm;
            db[i][j] *= U;
        }
        float sum = 0;
        for (int j = 0; j < vector_length; j++) {
            sum += db[i][j] * db[i][j];
        }
        vector_norm[i] = sqrt(sum);
        float vec_norm = vector_norm[i];
        for (int j = vector_length; j < vector_length + m; j++) {
            db[i][j] = vec_norm;
            vec_norm *= vec_norm;
        }
        for (int j = vector_length + m; j < vector_length + 2*m; j++) {
            db[i][j] = 0.5f;
        }
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);
    printf("Hashing vectors... ");
    clock_gettime(CLOCK_REALTIME, &start);

    //#pragma omp parallel for
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < number_of_vectors; i++) {
            vector<int> hash_vector(K);
            for (int k = 0; k < K; k++) {
                hash_vector[k] = hash_function(a_vectors[l][k], db[i], b_scalars[l][k]);
            }
            hash_tables[l][hash_vector].insert(i);
        }
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);
    
    // Print the contents of hash tables.
    int i = 0;
	// FIXME iterator below - iterate using integer.
    for (vector<map<vector<int>, set<int> > >::iterator it = hash_tables.begin(); it != hash_tables.end(); it++) {
        D(printf("Hash table %d:\n", i);)
        for (map<vector<int>, set<int> >::iterator it2 = it->begin(); it2 != it->end(); it2++) {
            D(printf("[ ");)
            for (vector<int>::const_iterator it3 = it2->first.begin(); it3 != it2->first.end(); it3++) {
                D(printf("%d ", *it3);)
            }
            printf("] -> ");
            for (set<int>::iterator it3 = it2->second.begin(); it3 != it2->second.end(); it3++) {
                D(printf("%d ", *it3);)
            }
            D(printf("\n");)
        }
        i++;
    }
    
    printf("Loading queries... ");
    clock_gettime(CLOCK_REALTIME, &start);
    
    FILE *q;
    q = fopen("queries", "r");
    int number_of_queries;
    int query_length;
    fscanf(q, "%d", &number_of_queries);
    fscanf(q, "%d", &query_length);
    assert(query_length == vector_length);
    vector<vector<float> > queries(number_of_queries);
    vector<set<int> > matched_vectors(number_of_queries);
    for (int i = 0; i < number_of_queries; i++) {
        queries[i].resize(query_length + 2*m);
    }
    vector<float> query_norm(number_of_queries);
    for (int i = 0; i < number_of_queries; i++) {
        for (int j = 0; j < query_length; j++) {
            fscanf(q, "%f", &queries[i][j]);
        }
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);
    printf("Scaling queries, computing norms and extending queries... ");
    clock_gettime(CLOCK_REALTIME, &start);
    
    #pragma omp parallel for
    for (int i = 0; i < number_of_queries; i++) {
        for (int j = 0; j < query_length; j++) {
            queries[i][j] /= maximum_norm;
            queries[i][j] *= U;
        }
        float sum = 0;
        for (int j = 0; j < query_length; j++) {
            sum += queries[i][j] * queries[i][j]; 
        }
        query_norm[i] = sqrt(sum);
        float q_norm = query_norm[i];
        for (int j = query_length; j < query_length + m; j++) {
            queries[i][j] = 0.5f;
        }
        for (int j = query_length + m; j < query_length + 2*m; j++) {
            queries[i][j] = q_norm;
            q_norm *= q_norm;
        }
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);
    printf("Matching queries... ");
    clock_gettime(CLOCK_REALTIME, &start);
    
    //#pragma omp parallel for
    for (int i = 0; i < number_of_queries; i++) {
        for (int l = 0; l < L; l++) {
            vector<int> hash_vector(K);
            for (int k = 0; k < K; k++) {
                hash_vector[k] = hash_function(a_vectors[l][k], queries[i], b_scalars[l][k]);
            }
            matched_vectors[i].insert(hash_tables[l][hash_vector].begin(), hash_tables[l][hash_vector].end());
        }
    }
    
    print_time_difference(start.tv_sec, start.tv_nsec);
    
    for (int i = 0; i < number_of_queries; i++) {
        printf("Query %d matches vectors: ", i);
        for (set<int>::iterator it = matched_vectors[i].begin(); it != matched_vectors[i].end(); it++) {
            printf(" %d", *it);
        }
        printf("\n");
    }
    
    fclose(q);
}
