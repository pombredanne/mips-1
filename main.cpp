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
typedef unsigned long ulong;

typedef struct layer_t_ {
   vector<int>   assignments; // Centroid number to which ith vector is assigned.
   vector<float> centroids;
   int cluster_num;
   int cluster_size;
} layer_t;

// ---------------------------------------------------------------------------------------------------------------------
// Utilities

vector<float> load_data(const char *filename, int& d, int& n, int m) {
   FILE *input;
   input = fopen(filename, "r");
   if (input == NULL) {
      printf("Couldn't open input file.\n");
      exit(EXIT_FAILURE);
   }

   fscanf(input, "%d", &n);
   fscanf(input, "%d", &d);

   vector<float> vectors((unsigned long) n * (d+m));
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < d; j++) {
         fscanf(input, "%f", &vectors[j + (i*d)]);
      }
   }
   fclose(input);

   return vectors;
}

inline bool comp(const pair<int, float> &a, const pair<int, float> &b) {
   return (a.second > b.second);
}

void print_centroids(layer_t_& layer, int d) {
   printf("centroids' coordinates:\n");
   for (int i = 0; i < layer.cluster_num; i++) {
      printf("%d: [ ", i);
      for (int j = 0; j < d; j++) {
         printf("%f ", layer.centroids[i * d + j]);
      }
      printf("]\n");
   }
}

void print_query(vector<float>& queries, int i, int d) {
   printf("\n--------------------------------------");
   printf("\nquery %d = [ ", i);
   for (int j = 0; j < (unsigned) d; j++) {
      printf("%f ", *(queries.data() + i * d + j));
   }
   printf("] ");
}

// ---------------------------------------------------------------------------------------------------------------------
// Linear Algebra

inline float fvec_norm_L2(float *vec, int size) {
   float sqr = faiss::fvec_norm_L2sqr(vec, (ulong) size);

   return sqrt(sqr);
}

inline void scale_(float* vec, float alpha, int size) {
   for (int i = 0; i < size; i++) {
      vec[i] /= alpha;
   }
}

// ---------------------------------------------------------------------------------------------------------------------
// Vectors

void normalize_(vector<float>& vectors, int n, int d) {
   float max_norm = 0;
   float norm;

   for (int i = 0; i < n; i++) {
      norm = fvec_norm_L2(vectors.data() + (i * d), d);
      max_norm = norm > max_norm ? norm : max_norm;
   }

   for (int i = 0; i < n; i++) {
      scale_(vectors.data() + (i*d), max_norm, d);
   }
}

void expand_(vector<float>& vectors, int n, int d, int m) {
   float power = 2.f;

   for (int i = 0; i < n; i++) {
      float norm = fvec_norm_L2(vectors.data() + (i*d), d);

      for (int j = d; j < d + m; j++) {
         vectors[i * d + j] = (float) (0.5 - pow(norm, power));
         power *= 2;
      }
   }
}

void expand_queries_(vector<float> &queries, int n, int d, int m) {
   for (int i = 0; i < n; i++) {
      for (int j = d-m; j < d; j++) {
         queries[j + (i*d)] = 0.f;
      }
   }
}

// ---------------------------------------------------------------------------------------------------------------------
// Clustering

void assign_(float *vectors, int n, int d, int k, int *assignments, float *centroids) {
   for (int i=0; i<n; i++) {
      float best = numeric_limits<float>::max();
      float dist = 0;

      for (int j=0; j<k; j++) {
         dist = faiss::fvec_inner_product(vectors + (i*d), centroids + (j*d), (ulong) d);
         if (best > dist) {
            assignments[i] = j;
            best = dist;
         }
      }
   }
}

void k_means_clustering_(float *vectors, int n, int d, int k, int *assignments, float *centroids) {
   faiss::kmeans_clustering((ulong) d, (ulong) n, (ulong) k, vectors, centroids);
   assign_(vectors, n, d, k, assignments, centroids);
}

vector<layer_t> train(vector<float>& vectors, int n, int d, int L) {
   vector<layer_t> layers = vector<layer_t>((ulong) L);

   for (int layer_id = 0; layer_id < L; layer_id++) {
      LOG("layer = %d", layer_id);

      // Compute number of clusters and cluster size on this layer.
      layers[layer_id].cluster_size = (int) floor(pow(n, (float) (layer_id + 1) / (float) L));
      layers[layer_id].cluster_num  = (int) floor((float) n / (float) layers[layer_id].cluster_size);

      LOG("cluster_num = %d\n", layers[layer_id].cluster_num);

      int n_points = (layer_id == 0) ? n : layers[layer_id - 1].cluster_num;
      float *points = (layer_id == 0) ? vectors.data() : layers[layer_id - 1].centroids.data();

      // Initial assignments are random.
      layers[layer_id].assignments = vector<int>((ulong) n_points);
      for (int i = 0; i < n_points; i++) {
         layers[layer_id].assignments[i] = rand() % layers[layer_id].cluster_num;
      }
      layers[layer_id].centroids = vector<float>((ulong) (layers[layer_id].cluster_num * d));

      // Cluster.
      LOG("assignments of vectors to centroids:\n");
      k_means_clustering_(points, n_points, d, layers[layer_id].cluster_num,
                          layers[layer_id].assignments.data(), layers[layer_id].centroids.data());

      if (VERBOSE)
         print_centroids(layers[layer_id], d);
   }

   return layers;
}

void predict(float *query_vector, vector<layer_t>& layers, vector<float>& vectors, vector<float>& vectors_copy,
             int P, int L, int d, int n) {

   // All centroids on the (L-1)th layer should be checked.
   int k = layers[layers.size() - 1].cluster_num;
   vector<int> candidates((ulong) k);

   for (int i = 0; i < candidates.size(); i++) {
      candidates[i] = i;
   }

   for (int layer_id = L - 1; layer_id >= 0; layer_id--) {
      LOG("layer = %d", layer_id)

      vector<std::pair<int, float> > best_centroids;
      // Multiply previously marked centroids with the query.
      for (int i = 0; i < candidates.size(); i++) {
         int c = candidates[i];
         float result = faiss::fvec_inner_product(query_vector, layers[layer_id].centroids.data() + c * d, (ulong) d);
         best_centroids.push_back(std::make_pair(c, result));

         LOG("centroid %d: result %f", c, result);
      }

      // We are interested in exploring P first centroids on this vector.
      sort(best_centroids.begin(), best_centroids.end(), comp);
      if (VERBOSE) {
         printf("best_centroids:\n");
         for (unsigned i = 0; i < best_centroids.size() && i < P; i++) {
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
      int num_points = (layer_id == 0) ? n : layers[layer_id - 1].cluster_num;
      candidates.clear();
      // Mark centroids to be checked at next layer.
      for (int i = 0; i < num_points; i++) {
         for (unsigned j = 0; j < P && j < best_centroids.size(); j++) {
            if (layers[layer_id].assignments[i] == best_centroids[j].first) {
               candidates.push_back(i);
               if (VERBOSE) { printf("%d ", i); }
               break;
            }
         }
      }
      LOG("\n");

      if (layer_id == 0) {
         // Last layer - find best match.
         int best_result = -1;
         float maximum_result = -1;
         bool maximum_initialized = false;
         for (int i = 0; i < candidates.size(); i++) {
            int c = candidates[i];
            float result = faiss::fvec_inner_product(query_vector, vectors.data() + c * d, (ulong) d);
            if (!maximum_initialized) {
               maximum_initialized = true;
               maximum_result = result;
               best_result = c;
            } else if (result > maximum_result) {
               maximum_result = result;
               best_result = c;
            }
         }
         printf("best result = %d : [", best_result);
         for (int i = 0; i < (unsigned) d; i++) {
            // Printing vector before transformations, to see the one after transformations use 'vectors'.
            printf("%f ", *(vectors_copy.data() + best_result * d + i));
         }
         printf("] inner product = %f", maximum_result);
      }
      best_centroids.clear();
   }
}

// ---------------------------------------------------------------------------------------------------------------------
// Main

int main(int argc, char* argv[]) {
   char* input_file;
   char* query_file;
   vector<float> vectors, vectors_copy, queries;
   int d, dq, n, nq, m, L, P;  // vector dim, query vec dim, num vectors, n queries, num components, num layers

   input_file = argv[1];
   query_file = argv[2];
   m          = atoi(argv[3]);
   L          = atoi(argv[4]);
   P          = atoi(argv[5]);
   VERBOSE    = (bool) atoi(argv[6]);

   vectors = load_data(input_file, d, n, m);
   vectors_copy = vectors;

   normalize_(vectors, n, d);
   expand_(vectors, n, d, m);

   vector<layer_t> layers = train(vectors, n, d+m, L);

   queries = load_data(query_file, dq, nq, m);
   expand_queries_(queries, nq, dq, m);
   assert(dq == d and "Queries and Vectors dimension mismatch!");

   for (int i = 0; i < nq; i++) {
      if (VERBOSE)
         print_query(queries, i, d);

      float* q = queries.data() + (i*d);
      predict(q, layers, vectors, vectors_copy,
              P, L, d, n);
   }
   printf("\n");

   return 0;
}
