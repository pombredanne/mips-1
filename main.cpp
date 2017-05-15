#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>
#include "faiss/utils.h"

using namespace std;

float compute_vector_length(float *vec, int length) {
	float sqr = faiss::fvec_norm_L2sqr(vec, length);
    return sqrt(sqr);
}

void scale_vector(float *vec, int length, float factor) {
    for (int i = 0; i < length; i++){
        vec[i] /= factor;
    }
}

float dot_product(float *vec1, float *vec2, int size) {
	return faiss::fvec_inner_product(vec1, vec2, size);
}

bool sorting_criterion(const pair<int, float> &a, const pair<int, float> &b) {
    return (a.second > b.second);
}

// Procedure clustering number_of_vectors vectors (each of them having dim components) into k clusters.
// Assignments of vectors to clusters are written to assignments array.
// Centroids' coordinates are written to centroids array.
// Clusters are numbered from 0 to k-1.
// TODO: Change to kmeans_clustering from faiss.
void k_means_clustering(float* vectors, int number_of_vectors,
	   	int dim, int k, int *assignments, float *centroids){
    float **sum_of_vectors = new float*[k];
    for (int i = 0; i < k; i++) {
        sum_of_vectors[i] = new float[dim];
        for (int j = 0; j < dim; j++) {
            sum_of_vectors[i][j] = 0;
        }
    }
    bool assignments_changed = true;
    int iter = 0;
    while(assignments_changed) {
        assignments_changed = false;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                sum_of_vectors[i][j] = 0;
            }
        }

        printf("iter %d: ", iter);
        for (int i = 0; i < number_of_vectors; i++) {
            printf("%d ", assignments[i]);
        }
        printf("\n");
        
        /* compute sum of all vectors */
        for (int i = 0; i < number_of_vectors; i++) {
            for (int j = 0; j < dim; j++) {
                sum_of_vectors[assignments[i]][j] += vectors[i*dim+j];
            }
        }
        
        /* divide resulting vector by its length */
        for (int i = 0; i < k; i++) {
            float vec_len = compute_vector_length(sum_of_vectors[i], dim);
            
            /* probably think of something better to do with vectors of length zero */
            if (vec_len != 0)
                scale_vector(sum_of_vectors[i], dim, vec_len);
            for (int j = 0; j < dim; j++) {
                centroids[i*dim+j] = sum_of_vectors[i][j];
            }
        }
        
        /* assign vectors to new centroids */
        /* this is parallelizable */
        for (int i = 0; i < number_of_vectors; i++) {
            bool maximum_initialized = false;
            int assignment;
            float maximum = 0;
            for (int j = 0; j < k; j++) {
                float result = dot_product(vectors+i*dim, centroids+j*dim, dim);
                if (!maximum_initialized) {
                    maximum_initialized = true;
                    maximum = result;
                    assignment = j;
                } else if (result > maximum) {
                    maximum = result;
                    assignment = j;
                }
            }
            if (assignment != assignments[i])
                assignments_changed = true;
            assignments[i] = assignment;
        }
        iter++;
    }
    for (int i = 0; i < k; i++) {
        delete[] sum_of_vectors[i];
    }
    delete[] sum_of_vectors;
}

const int m = 0;
const int L = 4; // Number of layers.
const int P = 3; // Number of checked best centroids.

vector<float> load_data(const char* filename, int* dim,
	   	int* number_of_vectors){
    FILE *input;
    input = fopen(filename, "r");
    if (input == NULL) {
        printf("Couldn't open input file.\n");
        exit(EXIT_FAILURE);
    }
    fscanf(input, "%d", number_of_vectors);
    fscanf(input, "%d", dim);
	*dim+=m;
    vector<float> vectors(*number_of_vectors * *dim);
    
    for (int i = 0; i < *number_of_vectors; i++) {
        for (int j = 0; j < *dim; j++) {
            fscanf(input, "%f", &vectors[i* (*dim) + j]);
        }
	}
    fclose(input);
	return vectors;
}

struct layer_t {
    vector<int> assignments; // Centroid number to which ith vector is assigned.
    vector<float> centroids;
    int cluster_num;
    int cluster_size;
};

vector<layer_t> layers;

int dim;
int number_of_vectors;
vector<float> vectors;

int number_of_queries;
vector<float> query;

void transform_data(){
	// Scale every vector so that maximum length is smaller than 1.
    float maximum_vector_length = 0;
    for (int i = 0; i < number_of_vectors; i++) {
        float vec_len = compute_vector_length(vectors.data()+i*dim, dim);
        if (vec_len > maximum_vector_length) {
            maximum_vector_length = vec_len;
        }
    }
    
    // Append m components to loaded vectors.
    for (int i = 0; i < number_of_vectors; i++){
        scale_vector(vectors.data()+i*dim, dim-m, maximum_vector_length);
        float new_vector_length = compute_vector_length(vectors.data()+i*dim, dim-m);
        int power = 2;
        for (int j = dim-m; j < dim; j++) {
            vectors[i*dim+j] = 0.5 - pow(new_vector_length, power);
            power *= 2;
        }
    }
}

void transform_queries(){
    for (int i = 0; i < number_of_queries; i++) {
        for(int j = dim-m; j < dim; j++) {
            query[i*dim+j] = 0;
        }
    }
}

void preprocess(){
	layers.resize(L);
    for (int lay = 0; lay < L; lay++) {
        printf("\nlayer = %d\n", lay);
        
        // Compute number of clusters and cluster size on this layer.
        layers[lay].cluster_size = floor(pow(number_of_vectors, (float)(lay+1)/(float)L));
        layers[lay].cluster_num = floor((float)number_of_vectors/(float)layers[lay].cluster_size);
        printf("cluster_num = %d\n", layers[lay].cluster_num);
        
		int number_of_points = (lay==0) ? number_of_vectors : layers[lay-1].cluster_num;
		float* points = (lay==0) ? vectors.data() : layers[lay-1].centroids.data();

		// Initial assignments are random.
		layers[lay].assignments = vector<int>(number_of_points);
		for (int i = 0; i < number_of_points; i++) {
			layers[lay].assignments[i] = rand() % layers[lay].cluster_num;
		}
        
        layers[lay].centroids = vector<float>(layers[lay].cluster_num * dim);
        
		// Cluster.
        printf("assignments of vectors to centroids:\n");
		k_means_clustering(points, number_of_points, dim, 
				layers[lay].cluster_num, layers[lay].assignments.data(), layers[lay].centroids.data());
                               
        printf("centroids' coordinates:\n");
        for (int i = 0; i < layers[lay].cluster_num; i++) {
            printf("%d: [ ", i);
            for (int j = 0; j < dim; j++) {
                printf("%f ", layers[lay].centroids[i*dim+j]);
            }
            printf("]\n");
        }
    }
}

void answer_query(float* query_vector){ 
	vector<int> candidates(layers[L-1].cluster_num);
	for(size_t i=0; i<candidates.size(); i++){
		candidates[i]=i;
	}
	for (int lay = L - 1; lay >= 0; lay--) {
		printf("\nlayer = %d\n", lay);
		vector< std::pair<int, float> > best_centroids;
		for(size_t i=0; i<candidates.size(); i++){
			int c=candidates[i];
			// This centroid is worth checking - find P highest inner products with query.
			float result = dot_product(query_vector, layers[lay].centroids.data()+c*dim, dim);
			printf("centroid %d: result %f\n", c, result);
			best_centroids.push_back(std::make_pair(c, result));
		}
		
		sort(best_centroids.begin(), best_centroids.end(), sorting_criterion);
		printf("best_centroids:\n");
		for (unsigned i = 0; i < best_centroids.size() && i < P; i++) {
			cout << best_centroids[i].first << " " << best_centroids[i].second << endl;
		}
		
		if (lay > 0) {
			printf("next layer centroids: ");
		}
		else{
			printf("candidate set (CL): ");
		}

		int num_points = (lay==0) ? number_of_vectors : layers[lay-1].cluster_num;
		candidates.clear();
		// Mark centroids to be checked at next layer.
		for (int i = 0; i < num_points; i++) {
			for (unsigned j = 0; j < P && j < best_centroids.size(); j++) {
				if (layers[lay].assignments[i] == best_centroids[j].first) {
					candidates.push_back(i);
					printf("%d ", i);
					break;
				}
			}
		}
		printf("\n");
		if (lay==0) {
			// Last layer - find best match.
			int best_result = -1;
			float maximum_result = -1;
			bool maximum_initialized = false;
			for(size_t i=0; i<candidates.size(); i++){
				int c=candidates[i];
				float result = dot_product(query_vector, vectors.data()+c*dim, dim);
				if (!maximum_initialized) {
					maximum_initialized = true;
					maximum_result = result;
					best_result = c;
				} else if (result > maximum_result) {
					maximum_result = result;
					best_result = c;
				}
			}
			printf("best result = %d with inner product = %f\n", best_result, maximum_result);
		}
		best_centroids.clear();
	}
}

int main(){
	vectors = load_data("data/input", &dim, &number_of_vectors);
	transform_data();
	preprocess();

	int dim_queries;
	query = load_data("data/queries", &dim_queries,
			&number_of_queries);
	transform_queries();
	assert(dim_queries == dim);
    
    for (int q = 0; q < number_of_queries; q++) {
		printf("\n--------------------------------------\n");
		printf("searching for query %d...\n", q);
		answer_query(query.data()+dim*q);
    }
}
