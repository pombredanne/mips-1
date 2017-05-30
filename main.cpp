#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "faiss/utils.h"
#include "faiss/Clustering.h"

using namespace std;

// ---------------------------------------------------------------------------------------------------------------------
// MISC

#define LOG(msg, ...) if(VERBOSE) { printf("\n" msg "\n", ##__VA_ARGS__); }
bool VERBOSE = true;

// ---------------------------------------------------------------------------------------------------------------------
// Utilities

struct layer_t {
    vector<size_t> assignments; // Centroid number to which ith vector is assigned.
    vector<float> centroids;
    size_t num_clusters;
};

inline bool comp(const pair<size_t, float> &a, const pair<size_t, float> &b) {
    return (a.second > b.second);
}

// ---------------------------------------------------------------------------------------------------------------------
// Vectors

inline float fvec_norm_L2(float *vec, size_t size) {
    float sqr = faiss::fvec_norm_L2sqr(vec, size);

    return sqrt(sqr);
}

inline void scale_(float* vec, float alpha, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] /= alpha;
    }
}

void normalize_(vector<float>& vectors, size_t n, size_t d) {
    float max_norm = 0;

    for (size_t i = 0; i < n; i++) {
        float norm = fvec_norm_L2(vectors.data() + (i * d), d);
        max_norm = norm > max_norm ? norm : max_norm;
    }

    for (size_t i = 0; i < n; i++) {
        scale_(vectors.data() + (i*d), max_norm, d);
    }
}

void expand_(vector<float>& vectors, size_t n, size_t d, size_t m) {
    for (size_t i = 0; i < n; i++) {
        float norm = fvec_norm_L2(vectors.data() + (i * d), d);
        for (size_t j = d - m; j < d; j++) {
            norm *= norm;
            vectors[i * d + j] = 0.5 - norm;
        }
    }
}

void expand_queries_(vector<float> &queries, size_t n, size_t d, size_t m) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = d - m; j < d; j++) {
            queries[i * d + j] = 0.f;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Clustering

void assign_(float *vectors, size_t n, size_t d, size_t k, size_t *assignments, float *centroids) {
    for (size_t i=0; i<n; i++) {
        float best = numeric_limits<float>::min();
        float sim = 0;

        for (size_t j=0; j<k; j++) {
            sim = faiss::fvec_inner_product(vectors + (i*d), centroids + (j*d), d);
            if (sim > best) {
                assignments[i] = j;
                best = sim;
            }
        }
    }
}

void k_means_clustering_(float *vectors, size_t n, size_t d, size_t k, size_t *assignments, float *centroids) {
    faiss::kmeans_clustering(d, n, k, vectors, centroids);
    assign_(vectors, n, d, k, assignments, centroids);
}

vector<layer_t> train(vector<float>& vectors, size_t n, size_t d, size_t L) {
    vector<layer_t> layers = vector<layer_t>(L);

    for (size_t layer_id = 0; layer_id < L; layer_id++) {
        size_t cluster_size = (size_t) floor(pow(n, (layer_id + 1.f) / (L + 1.f)));
        layers[layer_id].num_clusters = (size_t) floor((float) n / cluster_size);

        size_t n_points = (layer_id == 0) ? n : layers[layer_id - 1].num_clusters;
        float *points   = (layer_id == 0) ? vectors.data() : layers[layer_id - 1].centroids.data();

        layers[layer_id].assignments = vector<size_t>(n_points);
        layers[layer_id].centroids   = vector<float>(layers[layer_id].num_clusters * d);

        // Cluster.
        k_means_clustering_(points,
                            n_points, d, layers[layer_id].num_clusters,
                            layers[layer_id].assignments.data(), layers[layer_id].centroids.data());
    }

    return layers;
}



size_t predict(float *query_vector, vector<layer_t>& layers, vector<float>& vectors, vector<float>& vectors_copy,
        size_t P, size_t L, size_t d, size_t n) {

    // All centroids on the (L-1)th layer should be checked.
    size_t k = layers[layers.size() - 1].num_clusters;
    vector<size_t> candidates(k);

    for (size_t i = 0; i < candidates.size(); i++) {
        candidates[i] = i;
    }

    for (size_t layer_id = L - 1; layer_id != (size_t)(-1); layer_id--) {
        vector<std::pair<size_t, float> > best_centroids;

        // Multiply previously marked centroids with the query.
        for (size_t i = 0; i < candidates.size(); i++) {
            size_t c = candidates[i];
            float result = faiss::fvec_inner_product(query_vector, layers[layer_id].centroids.data() + c * d, d);
            best_centroids.push_back(std::make_pair(c, result));
        }

        // We are interested in exploring P first centroids on this vector.
        sort(best_centroids.begin(), best_centroids.end(), comp);

        size_t num_points = (layer_id == 0) ? n : layers[layer_id - 1].num_clusters;
        candidates.clear();

        // Mark centroids to be checked at next layer.
        for (size_t i = 0; i < num_points; i++) {
            for (size_t j = 0; j < P && j < best_centroids.size(); j++) {
                if (layers[layer_id].assignments[i] == best_centroids[j].first) {
                    candidates.push_back(i);
                    break;
                }
            }
        }

        best_centroids.clear();
    }
    // Last layer - find best match.
    size_t best_result   = -1;
    float maximum_result = std::numeric_limits<float>::min();

    for (size_t i = 0; i < candidates.size(); i++) {
        size_t c = candidates[i];

        float result = faiss::fvec_inner_product(query_vector, vectors.data() + c * d, d);
        if (result > maximum_result) {
            maximum_result = result;
            best_result = c;
        }
    }

    return best_result;
}

template <typename T>
vector<T> load_vecs (const char* fname, size_t& d, size_t& n, size_t m){
    FILE* f = fopen(fname, "rb");
    int dim;
    fread(&dim, 1, sizeof(int), f);
    fseek(f, 0, SEEK_END);
    size_t fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t row_size = sizeof(T) * dim + sizeof(int);
    n = fsz / row_size;
    if(fsz != n * row_size){
        printf("Wrong file size\n");
        exit(1);
    }
    d = dim + m;
    vector<T> v (n * d);
    for(size_t i = 0; i < n; i++){
        int dummy;
        fread(&dummy, 1, sizeof(int), f);
        assert(dummy == dim);
        fread(v.data() + d * i, dim, sizeof(T), f);
    }
    return v;
}

template <typename T>
void dump_vecs(vector<T>& vec, int d, const char* fname) {
    ofstream fout(fname, ios::out | ios::binary);

    fout.write((char*) &d, sizeof(int));
    fout.write((char*) &vec[0], vec.size() * sizeof(T));

    fout.close();
}

// ---------------------------------------------------------------------------------------------------------------------
// Main

int main(int argc, char* argv[]) {
    char* input_file;
    char* query_file;
    vector<float> vectors, vectors_copy, queries, queries_copy;
    size_t d, dq, n, nq, m, L, P;  // vector dim, query vec dim, num vectors, n queries, num components, num layers

    if(argc != 7){
        LOG("Bad usage!");
        return -1;
    }

    input_file = argv[1];
    query_file = argv[2];
    m          = atoi(argv[3]);
    L          = atoi(argv[4]);
    P          = atoi(argv[5]);
    VERBOSE    = (bool) atoi(argv[6]);

    vectors = load_vecs<float>(input_file, d, n, m);
    vectors_copy = vectors;

    normalize_(vectors, n, d);
    expand_(vectors, n, d, m);

    queries = load_vecs<float>(query_file, dq, nq, m);
    queries_copy = queries;
    expand_queries_(queries, nq, dq, m);

    assert(dq == d and "Queries and Vectors dimension mismatch!");

    vector<layer_t> layers = train(vectors, n, d, L);

    vector<int> predictions;
    for (size_t i = 0; i < nq; i++) {
        size_t res = predict(queries.data() + (i*d),
                             layers, vectors, vectors_copy,
                             P, L, d, n);

        printf("p: %zu\n", res);
        predictions.push_back(res);
    }

    dump_vecs(predictions, 1,   "vectors/preds.bin");

    dump_vecs(vectors,      d,  "vectors/vectors.bin");
    dump_vecs(vectors_copy, d,  "vectors/vectors_copy.bin");

    dump_vecs(queries,      dq, "vectors/queries.bin");
    dump_vecs(queries_copy, dq, "vectors/queries_copy.bin");

    return 0;
}
