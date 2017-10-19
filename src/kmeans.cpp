#include "common.h"

#include "faiss/utils.h"
#include "faiss/Clustering.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

// ---------------------------------------------------------------------------------------------------------------------
// MISC

#define LOG(msg, ...) if(VERBOSE) { printf("\n" msg "\n", ##__VA_ARGS__); }
bool VERBOSE = true;

typedef FlatMatrix<float> FloatMatrix;

struct layer_t {
	kmeans_result kr;
    size_t cluster_num;
};

// ---------------------------------------------------------------------------------------------------------------------
// Utilities

FloatMatrix load_data(const char *filename, size_t m) {
	size_t cnt, dim;
    FILE *input;
    input = fopen(filename, "r");
    if (input == NULL) {
        printf("Couldn't open input file.\n");
        exit(EXIT_FAILURE);
    }

    if(!fscanf(input, "%zu", &cnt)){
        printf("Bad input file format.");
        exit(EXIT_FAILURE);
    }
    if(!fscanf(input, "%zu", &dim)){
        printf("Bad input file format.");
        exit(EXIT_FAILURE);
    }
    dim += m; // Extending here.

	FloatMatrix result;
	result.resize(cnt, dim);

    for (size_t i = 0; i < cnt; i++) {
        for (size_t j = 0; j < dim - m; j++) {
            if(!fscanf(input, "%f", &result.at(i, j))){
                printf("Bad input file format.");
                exit(EXIT_FAILURE);
            }
        }
        for (size_t j = dim - m; j < dim; j++){
            result.at(i, j) = 0.0;
        }
    }
    fclose(input);

    return result;
}

inline bool comp(const pair<size_t, float> &a, const pair<size_t, float> &b) {
    return (a.second > b.second);
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

void normalize_(FloatMatrix& matrix) {
    float max_norm = 0;

    for (size_t i = 0; i < matrix.vector_count(); i++) {
        float norm = fvec_norm_L2(matrix.row(i), matrix.vector_length);
        max_norm = norm > max_norm ? norm : max_norm;
    }

    for (size_t i = 0; i < matrix.vector_count(); i++) {
        scale_(matrix.row(i), max_norm, matrix.vector_length);
    }
}

void expand_(FloatMatrix& matrix, size_t m) {
    for (size_t i = 0; i < matrix.vector_count(); i++) {
        float norm = fvec_norm_L2(matrix.row(i), matrix.vector_length);
        for (size_t j = matrix.vector_length - m; j < matrix.vector_length; j++) {
            norm *= norm;
            matrix.at(i, j) = 0.5 - norm;
        }
    }
}

void expand_queries_(FloatMatrix& matrix, size_t m) {
    for (size_t i = 0; i < matrix.vector_count(); i++) {
        for (size_t j = matrix.vector_length - m; j < matrix.vector_length; j++) {
            matrix.at(i, j) = 0.f;
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

vector<layer_t> train(FloatMatrix& vectors, size_t L) {
    vector<layer_t> layers = vector<layer_t>(L);

    for (size_t layer_id = 0; layer_id < L; layer_id++) {
        LOG("layer = %zu", layer_id);

        // Compute number of clusters and cluster size on this layer.
        size_t cluster_size = floor(
				pow(vectors.vector_count(), (layer_id + 1) / (float) (L + 1)));

        layers[layer_id].cluster_num = floor(
				vectors.vector_count() / (float) cluster_size);

        LOG("cluster_num = %zu", layers[layer_id].cluster_num);

        FloatMatrix& points = (layer_id == 0) ?
		   	vectors : layers[layer_id - 1].kr.centroids;

		layers[layer_id].kr = perform_kmeans(points, layers[layer_id].cluster_num);

        if (VERBOSE) {
			std::cout << "Centroids in layer " << layer_id << std::endl;
			layers[layer_id].kr.centroids.print();
		}
    }

    return layers;
}

size_t predict(vector<layer_t>& layers, FloatMatrix& queries, size_t qnum,
		size_t opened_trees, FloatMatrix& vectors) {

    // All centroids on the (L-1)th layer should be checked.
    vector<size_t> candidates;

    for (size_t i = 0; i < layers.back().cluster_num; i++) {
        candidates.push_back(i);
    }

    for (size_t layer_id = layers.size() - 1; layer_id != (size_t)(-1); layer_id--) {
        LOG("layer = %zu", layer_id)

		vector<std::pair<size_t, float>> best_centroids;
        for (size_t i = 0; i < candidates.size(); i++) {
            size_t c = candidates[i];
            float result = faiss::fvec_inner_product(
					queries.row(qnum), 
					layers[layer_id].kr.centroids.row(c),
					queries.vector_length);

            best_centroids.push_back({c, result});
            LOG("centroid %zu: result %f", c, result);
        }

        // We are interested in exploring P first centroids on this vector.
        sort(best_centroids.begin(), best_centroids.end(), comp);
        if (VERBOSE) {
            printf("best_centroids:\n");
            for (size_t i = 0; i < best_centroids.size() && i < opened_trees; i++) {
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

        size_t num_points = (layer_id == 0) ?
		   	vectors.vector_count() : layers[layer_id - 1].cluster_num;
        LOG("np: %zu", num_points);
        candidates.clear();
        // Mark centroids to be checked at next layer.
		// TODO: Check this carefully - dumb or wrong.
        for (size_t i = 0; i < num_points; i++) {
            for (size_t j = 0; j < opened_trees && j < best_centroids.size(); j++) {
                if (layers[layer_id].kr.assignments[i] == best_centroids[j].first) {
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
    float maximum_result = numeric_limits<float>::min();
    bool maximum_initialized = false;
    for (size_t i = 0; i < candidates.size(); i++) {
        size_t c = candidates[i];
        float result = faiss::fvec_inner_product(
				queries.row(qnum),
			   	vectors.row(c),
			   	vectors.vector_length);
        if (!maximum_initialized) {
            maximum_initialized = true;
            maximum_result = result;
            best_result = c;
        } else if (result > maximum_result) {
            maximum_result = result;
            best_result = c;
        }
    }
    printf("Vector %zu, inner product = %f\n", best_result, maximum_result);
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

void main_kmeans() {
    const char* input_file;
    const char* query_file;
    size_t m, L;
	size_t opened_trees = 2;

	input_file = "input";
	query_file = "queries";
	m = 3;
	L = 2;
	VERBOSE = 0;

	auto vectors = load_data(input_file, m);
    auto vectors_copy = vectors;

    normalize_(vectors);
    expand_(vectors, m);

    vector<layer_t> layers = train(vectors, L);

	auto queries = load_data(query_file, m);
    expand_queries_(queries, m);

    assert(vectors.vector_length == queries.vector_length and
		   	"Queries and Vectors dimension mismatch!");

    vector<int> predictions;
    for (size_t i = 0; i < queries.vector_count(); i++) {
		std::cout << "Query " << i << std::endl;

        size_t res = predict(layers, queries, i, opened_trees, vectors);
        predictions.push_back(res);
		printf("%zu\n", res);
    }
    printf("\n");
}
