#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>
#include "faiss/utils.h"
#include "faiss/Clustering.h"

using namespace std;

// ---------------------------------------------------------------------------------------------------------------------
// MISC

#define LOG(msg, ...) if(VERBOSE) { printf("\n" msg "\n", ##__VA_ARGS__); }
bool VERBOSE = true;

struct layer_t {
    vector<size_t> assignments; // Centroid number to which ith vector is assigned.
    vector<float> centroids;
    size_t cluster_num;
};

// ---------------------------------------------------------------------------------------------------------------------
// Utilities

vector<float> load_data(const char *filename, size_t& d, size_t& n, size_t m) {
    FILE *input;
    input = fopen(filename, "r");
    if (input == NULL) {
        printf("Couldn't open input file.\n");
        exit(EXIT_FAILURE);
    }

    if(!fscanf(input, "%zu", &n)){
        printf("Bad input file format.");
        exit(EXIT_FAILURE);
    }
    if(!fscanf(input, "%zu", &d)){
        printf("Bad input file format.");
        exit(EXIT_FAILURE);
    }
    d += m; // Extending here.

    vector<float> vectors(n * d);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d - m; j++) {
            if(!fscanf(input, "%f", &vectors[j + (i * d)])){
                printf("Bad input file format.");
                exit(EXIT_FAILURE);
            }
        }
        for (size_t j = d - m; j < d; j++){
            vectors[j + (i * d)] = 0.0;
        }
    }
    fclose(input);

    return vectors;
}

inline bool comp(const pair<size_t, float> &a, const pair<size_t, float> &b) {
    return (a.second > b.second);
}

void print_centroids(layer_t &layer, size_t d) {
    printf("centroids' coordinates:\n");
    for (size_t i = 0; i < layer.cluster_num; i++) {
        printf("%zu: [ ", i);
        for (size_t j = 0; j < d; j++) {
            printf("%f ", layer.centroids[i * d + j]);
        }
        printf("]\n");
    }
}

void print_query(vector<float>& queries, size_t i, size_t d) {
    printf("\n--------------------------------------");
    printf("\nquery %zu = [ ", i);
    for (size_t j = 0; j < d; j++) {
        printf("%f ", *(queries.data() + i * d + j));
    }
    printf("] ");
}

// ---------------------------------------------------------------------------------------------------------------------
// Linear Algebra

inline float fvec_norm_L2(float *vec, size_t size) {
    float sqr = faiss::fvec_norm_L2sqr(vec, size);

    return sqrt(sqr);
}

inline void scale_(float* vec, float alpha, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] /= alpha;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Vectors

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
        float best = numeric_limits<float>::max();
        float dist = 0;

        for (size_t j=0; j<k; j++) {
            dist = faiss::fvec_inner_product(vectors + (i*d), centroids + (j*d), d);
            if (best > dist) {
                assignments[i] = j;
                best = dist;
            }
        }
    }
}

void k_means_clustering_(float *vectors, size_t n, size_t d, size_t k, size_t *assignments, float *centroids) {
    faiss::kmeans_clustering(d, n, k, vectors, centroids);
    for(size_t i = 0; i < k; i++){
        LOG("Cluster %zu [", i);
        for(size_t j = 0; j < d; j++){
            printf("%f ", centroids[d*i+j]);
        }
    }
    assign_(vectors, n, d, k, assignments, centroids);
    LOG("Clustering %zu points into %zu centroids", n, k);
    for(size_t i = 0; i < n; i++){
        /*
        printf("The point %zu was [", i);
        for(size_t j = 0; j < d; j++){
            printf("%f ", vectors[d*i+j]);
        }
        printf("]");
        */
        LOG("  it got assigned to centroid %zu [", assignments[i]);
        /*
        for(size_t j = 0; j < d; j++){
            printf("%f ", centroids[d*assignments[i]+j]);
        }
        printf("]\n\n");
        */
    }
}

vector<layer_t> train(vector<float>& vectors, size_t n, size_t d, size_t L) {
    vector<layer_t> layers = vector<layer_t>(L);

    for (size_t layer_id = 0; layer_id < L; layer_id++) {
        LOG("layer = %zu", layer_id);

        // Compute number of clusters and cluster size on this layer.
        size_t cluster_size = (size_t) floor(pow(n, (float) (layer_id + 1) / (float) (L+1)));
        layers[layer_id].cluster_num = (size_t) floor((float) n / (float) cluster_size);

        LOG("cluster_num = %zu", layers[layer_id].cluster_num);

        size_t n_points = (layer_id == 0) ? n : layers[layer_id - 1].cluster_num;
        float *points = (layer_id == 0) ? vectors.data() : layers[layer_id - 1].centroids.data();

        layers[layer_id].assignments = vector<size_t>(n_points);
        layers[layer_id].centroids = vector<float>(layers[layer_id].cluster_num * d);

        // Cluster.
        k_means_clustering_(points, n_points, d, layers[layer_id].cluster_num,
                layers[layer_id].assignments.data(), layers[layer_id].centroids.data());

        if (VERBOSE)
            print_centroids(layers[layer_id], d);
    }

    return layers;
}

size_t predict(float *query_vector, vector<layer_t>& layers, vector<float>& vectors, vector<float>& vectors_copy,
        size_t P, size_t L, size_t d, size_t n) {

    // All centroids on the (L-1)th layer should be checked.
    size_t k = layers[layers.size() - 1].cluster_num;
    vector<size_t> candidates(k);

    for (size_t i = 0; i < candidates.size(); i++) {
        candidates[i] = i;
    }

    for (size_t layer_id = L - 1; layer_id != (size_t)(-1); layer_id--) {
        LOG("layer = %zu", layer_id)

            vector<std::pair<size_t, float> > best_centroids;
        // Multiply previously marked centroids with the query.
        for (size_t i = 0; i < candidates.size(); i++) {
            size_t c = candidates[i];
            float result = faiss::fvec_inner_product(query_vector, layers[layer_id].centroids.data() + c * d, d);
            best_centroids.push_back(std::make_pair(c, result));

            LOG("centroid %zu: result %f", c, result);
        }

        // We are interested in exploring P first centroids on this vector.
        sort(best_centroids.begin(), best_centroids.end(), comp);
        if (VERBOSE) {
            printf("best_centroids:\n");
            for (size_t i = 0; i < best_centroids.size() && i < P; i++) {
                cout << best_centroids[i].first << " " << best_centroids[i].second << endl;
            }
        }

        if (VERBOSE) {
            if (layer_id > 0) {
                printf("next layer centroids: ");
            }
            else {
                printf("candidate set (CL): ");
            }
        }
        size_t num_points = (layer_id == 0) ? n : layers[layer_id - 1].cluster_num;
        LOG("np: %zu", num_points);
        candidates.clear();
        // Mark centroids to be checked at next layer.
        for (size_t i = 0; i < num_points; i++) {
            for (size_t j = 0; j < P && j < best_centroids.size(); j++) {
                if (layers[layer_id].assignments[i] == best_centroids[j].first) {
                    candidates.push_back(i);
                    if (VERBOSE) { printf("%zu ", i); }
                    break;
                }
            }
        }

        best_centroids.clear();
    }
    // Last layer - find best match.
    size_t best_result = -1;
    float maximum_result = -1;
    bool maximum_initialized = false;
    for (size_t i = 0; i < candidates.size(); i++) {
        size_t c = candidates[i];
        float result = faiss::fvec_inner_product(query_vector, vectors.data() + c * d, d);
        if (!maximum_initialized) {
            maximum_initialized = true;
            maximum_result = result;
            best_result = c;
        } else if (result > maximum_result) {
            maximum_result = result;
            best_result = c;
        }
    }
    printf("best result = %zu : [", best_result);
    for (size_t i = 0; i < d; i++) {
        // Printing vector before transformations, to see the one after transformations use 'vectors'.
        printf("%f ", *(vectors_copy.data() + best_result * d + i));
    }
    printf("] inner product = %f\n", maximum_result);
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

// ---------------------------------------------------------------------------------------------------------------------
// Main

void main_kmeans() {
    char* input_file;
    char* query_file;
    vector<float> vectors, vectors_copy, queries;
    size_t d, dq, n, nq, m, L, P;  // vector dim, query vec dim, num vectors, n queries, num components, num layers


	// TODO: FIX API!!
	/*
    input_file = argv[1];
    query_file = argv[2];
    m          = atoi(argv[3]);
    L          = atoi(argv[4]);
    P          = atoi(argv[5]);
    VERBOSE    = (bool) atoi(argv[6]);
	*/

    bool FRANKOWY = 0;

    if(!FRANKOWY){
        vectors = load_vecs<float>(input_file, d, n, m);
    }
    else{
        vectors = load_data(input_file, d, n, m);
    }
    vectors_copy = vectors;

    normalize_(vectors, n, d);
    expand_(vectors, n, d, m);

    vector<layer_t> layers = train(vectors, n, d, L);

    if(!FRANKOWY){
        queries = load_vecs<float>(query_file, dq, nq, m);
    }
    else{
        queries = load_data(query_file, dq, nq, m);
    }
    expand_queries_(queries, nq, dq, m);
    assert(dq == d and "Queries and Vectors dimension mismatch!");

    vector<int> predictions;
    for (size_t i = 0; i < nq; i++) {
        if (VERBOSE)
            print_query(queries, i, d);

        float* q = queries.data() + (i*d);
        size_t res = predict(q, layers, vectors, vectors_copy,
                P, L, d, n);
        predictions.push_back(res);
    }
    printf("\n");
    if(FRANKOWY){
        return;
    }
    printf("Checking against ground truth\n");
    for(size_t i = 0; i < nq; i++){
        float inner_with_chosen = faiss::fvec_inner_product(vectors_copy.data() + predictions[i] * d,
                queries.data() + i * d, d - m);
        size_t rank = 0;
        for (size_t j = 0; j < n; j++) {
            float inner_with_other = faiss::fvec_inner_product(vectors_copy.data() + j * d,
                    queries.data() + i * d, d - m);
            if (inner_with_other > inner_with_chosen) {
                rank++;
            }
        }
        printf("Rank for query %zu: %zu\n", i, rank);
    }
}
