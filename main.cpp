#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

float compute_vector_length(float *vec, int length) {
    float sum_of_squares = 0;
    for (int i = 0; i < length; i++) {
        sum_of_squares += vec[i] * vec[i];
    }
    return sqrt(sum_of_squares);
}

void scale_vector(float *vec, int length, float factor) {
    for (int i = 0; i < length; i++){
        vec[i] /= factor;
    }
}

float dot_product(float *vec1, float *vec2, int size) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

bool sorting_criterion(const pair<int, float> &a, const pair<int, float> &b) {
    return (a.second > b.second);
}

// procedure clustering number_of_vectors vectors (each of them having vector_comp components) into k clusters
// assignments of vectors to clusters are written to assignments array
// centroids' coordinates are written to centroids array
// clusters are numbered from 0 to k-1
void k_means_clustering(float **vectors, int number_of_vectors,
	   	int vector_comp, int k, int *assignments, float **centroids){
    float **sum_of_vectors = new float*[k];
    for (int i = 0; i < k; i++) {
        sum_of_vectors[i] = new float[vector_comp];
        for (int j = 0; j < vector_comp; j++) {
            sum_of_vectors[i][j] = 0;
        }
    }
    bool assignments_changed = true;
    int iter = 0;
    while(assignments_changed) {
        assignments_changed = false;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < vector_comp; j++) {
                sum_of_vectors[i][j] = 0;
            }
        }

        /* print assignments */
        printf("iter %d: ", iter);
        for (int i = 0; i < number_of_vectors; i++) {
            printf("%d ", assignments[i]);
        }
        printf("\n");
        
        /* compute sum of all vectors */
        for (int i = 0; i < number_of_vectors; i++) {
            for (int j = 0; j < vector_comp; j++) {
                sum_of_vectors[assignments[i]][j] += vectors[i][j];
            }
        }
        
        /* divide resulting vector by its length */
        for (int i = 0; i < k; i++) {
            float vec_len = compute_vector_length(sum_of_vectors[i], vector_comp);
            
            /* probably think of something better to do with vectors of length zero */
            if (vec_len != 0)
                scale_vector(sum_of_vectors[i], vector_comp, vec_len);
            for (int j = 0; j < vector_comp; j++) {
                centroids[i][j] = sum_of_vectors[i][j];
            }
        }
        
        /* assign vectors to new centroids */
        /* this is parallelizable */
        for (int i = 0; i < number_of_vectors; i++) {
            bool maximum_initialized = false;
            int assignment;
            float maximum = 0;
            for (int j = 0; j < k; j++) {
                float result = dot_product(vectors[i], centroids[j], vector_comp);
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
const int layer_num = 4;
const int p = 3;

struct layer_type {
    int *assignments;
    /* assignments[i] == j means that vector i is assigned to centroid j */
    float **centroids;
    int cluster_num;
    int cluster_size;
    bool *search;
    /* search array indicates if centroids assigned to this cluster should be chceked in the next layer */
};

float** load_data(const char* filename, int* vector_components,
	   	int* number_of_vectors){
    FILE *input;
    input = fopen(filename, "r");
    if (input == NULL) {
        printf("Couldn't open input file.\n");
        exit(EXIT_FAILURE);
    }
    fscanf(input, "%d", number_of_vectors);
    fscanf(input, "%d", vector_components);
    float **vectors = new float*[*number_of_vectors];
    
    /* load vectors and find longest one */
    for (int i = 0; i < *number_of_vectors; i++) {
        vectors[i] = new float[*vector_components + m];
        for (int j = 0; j < *vector_components; j++) {
            fscanf(input, "%f", &vectors[i][j]);
        }
	}
    fclose(input);
	return vectors;
}

int main(){
    //srand(time(NULL));
	int vector_components;
	int number_of_vectors;

	float** vectors = load_data("data/input", &vector_components, &number_of_vectors);
    float maximum_vector_length = 0;
    for (int i = 0; i < number_of_vectors; i++) {
        float vec_len = compute_vector_length(vectors[i], vector_components);
        if (vec_len > maximum_vector_length) {
            maximum_vector_length = vec_len;
        }
    }
    
    /* append m components to loaded vectors */
    for (int i = 0; i < number_of_vectors; i++){
        scale_vector(vectors[i], vector_components, maximum_vector_length);
        float new_vector_length = compute_vector_length(vectors[i], vector_components);
        int power = 2;
        for (int j = vector_components; j < vector_components + m; j++) {
            vectors[i][j] = 0.5 - pow(new_vector_length, power);
            power *= 2;
        }
    }
    
    /* prepare search tree */
	layer_type* layer = new layer_type[layer_num];
    for (int lay = 0; lay < layer_num; lay++) {
        printf("\nlayer = %d\n", lay);
        
        /* compute number of clusters and cluster size on this layer */
        layer[lay].cluster_size = floor(pow(number_of_vectors, (float)(lay+1)/(float)layer_num));
        layer[lay].cluster_num = floor((float)number_of_vectors/(float)layer[lay].cluster_size);
        printf("cluster_num = %d\n", layer[lay].cluster_num);
        
        /* allocate assignments array */
        if (lay == 0) {
            layer[lay].assignments = new int[number_of_vectors];
            for (int i = 0; i < number_of_vectors; i++) {
                layer[lay].assignments[i] = rand() % layer[lay].cluster_num;
            }
        } else {
            layer[lay].assignments = new int[layer[lay-1].cluster_num];
            for (int i = 0; i < layer[lay-1].cluster_num; i++) {
                layer[lay].assignments[i] = rand() % layer[lay].cluster_num;
            }
        }
        
        /* allocate centroids array */
        layer[lay].centroids = new float*[layer[lay].cluster_num];
        for (int i = 0; i < layer[lay].cluster_num; i++) {
            layer[lay].centroids[i] = new float[vector_components + m];
        }
        layer[lay].search = new bool[layer[lay].cluster_num];
        
        /* do the clustering */
        printf("assignments of vectors to centroids:\n");
        if (lay == 0)
            k_means_clustering(vectors, number_of_vectors, vector_components + m, layer[lay].cluster_num,
                               layer[lay].assignments, layer[lay].centroids);
        else 
            k_means_clustering(layer[lay-1].centroids, layer[lay-1].cluster_num, vector_components + m,
          
                               layer[lay].cluster_num, layer[lay].assignments, layer[lay].centroids);
                               
        /* print the centroids */
        printf("centroids' coordinates:\n");
        for (int i = 0; i < layer[lay].cluster_num; i++) {
            printf("%d: [ ", i);
            for (int j = 0; j < vector_components + m; j++) {
                printf("%f ", layer[lay].centroids[i][j]);
            }
            printf("]\n");
        }
    }
    
    /* load queries */
    int number_of_queries;
    int vector_components_queries;
	float** query = load_data("data/queries", &vector_components_queries,
			&number_of_queries);
    for (int i = 0; i < number_of_queries; i++) {
        /* append m additional zeros at the end of query */
        for(int j = vector_components; j < vector_components + m; j++) {
            query[i][j] = 0;
        }
    }
    
    for (int q = 0; q < number_of_queries; q++) {
    
        /* getting -1 as the best_result means that program failed */
        int best_result = -1;
        float maximum_result = -1;
        printf("\n--------------------------------------\n");
        printf("searching for query %d...\n", q);
        
        for (int lay = layer_num - 1; lay >= 0; lay--) {
			vector< std::pair<int, float> > best_centroids;
            printf("\nlayer = %d\n", lay);
            for (int i = 0; i < layer[lay-1].cluster_num; i++) {
                layer[lay-1].search[i] = false;
            }
            bool maximum_initialized = false;
            for (int i = 0; i < layer[lay].cluster_num; i++) {
                bool search = false;
                /* in the first layer we check all centroids */
                if (lay < layer_num - 1) {
                    /* we search only in the interesting centroids marked in layer above */
                    if (layer[lay].search[i]) {
                        search = true;
                    }
                }
                
                /* this centroid is worth checking - find p highest inner products with query */
                /* A_l = argmax_{i in C_l}^{(p)} q^T c_i^{(l)} */
                if (lay == layer_num - 1 || search) {
                    float result = dot_product(query[q], layer[lay].centroids[i], vector_components + m);
                    printf("centroid %d: result %f\n", i, result);
                    best_centroids.push_back(std::make_pair(i, result));
                }
            }
            
            sort(best_centroids.begin(), best_centroids.end(), sorting_criterion);
            printf("best_centroids:\n");
            for (unsigned i = 0; i < best_centroids.size() && i < p; i++) {
                cout << best_centroids[i].first << " " << best_centroids[i].second << endl;
            }
            
            if (lay > 0) {
                /* mark centroids to be checked at next layer */
                /* C_{l+1} = {i|a_i^{(l+1)} in A_l} */
                printf("next layer centroids: ");
                for (int i = 0; i < layer[lay-1].cluster_num; i++) {
                    for (unsigned j = 0; j < p && j < best_centroids.size(); j++) {
                        if (layer[lay].assignments[i] == best_centroids[j].first) {
                            layer[lay-1].search[i] = true;
                            printf("%d ", i);
                            break;
                        }
                    }
                }
                printf("\n");
            } else {
                /* we are at the last layer - print candidate set and find best */
                maximum_initialized = false;
                printf("candidate set (CL): ");
                for (int i = 0; i < number_of_vectors; i++) {
                    for (unsigned j = 0; j < p && j < best_centroids.size(); j++) {
                        if (layer[lay].assignments[i] == best_centroids[j].first) {
                            /* this means that ith vector is in candidate set */
                            float result = dot_product(query[q], vectors[i], vector_components + m);
                            if (!maximum_initialized) {
                                maximum_initialized = true;
                                maximum_result = result;
                                best_result = i;
                            } else if (result > maximum_result) {
                                maximum_result = result;
                                best_result = i;
                            }
                            printf("%d ", i);
                        }
                    }
                }
            }
            best_centroids.clear();
        }
        
        printf("\nbest result = %d with inner product = %f\n", best_result, maximum_result);
    }
    
    for (int lay = 0; lay < layer_num; lay++) {
        delete[] layer[lay].assignments;
        for (int i = 0; i < layer[lay].cluster_num; i++) {
            delete[] layer[lay].centroids[i];
        }
        delete[] layer[lay].centroids;
        delete[] layer[lay].search;
    }
    delete[] layer;
    for (int i = 0; i < number_of_vectors; i++) {
        delete[] vectors[i];
    }
    delete[] vectors;
    for (int i = 0; i < number_of_queries; i++) {
        delete[] query[i];
    }
    delete[] query;
    return 0;
}
