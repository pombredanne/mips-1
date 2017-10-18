#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> 
#include <numeric>  
#include "faiss/utils.h"
#include "faiss/Clustering.h"

using namespace std;

struct kmeans_result {
    vector<float> centroids;
    vector<size_t> assigments;
};

vector<float> load_data(string filename, size_t* vec_cnt, size_t* dim_cnt) {
    vector<float> data;
    ifstream infile(filename);
    size_t n, m;
    infile >> n >> m;
    cout << n << " " << m << endl;
    for (size_t i = 0; i < n * m; i++) {
        float tmp;
        infile >> tmp;
        data.push_back(tmp);
    }
    *vec_cnt = n;
    *dim_cnt = m;
    return data;
}

void print_data(vector<float> data, size_t n, size_t m) {
    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; j++) {
            cout << data[i*m+j]<<" ";
        }
        cout<<endl;
    }
}

vector<size_t> prepare_indices_vector(size_t m) {
    // TODO: Don't shuffle, but rotate randomly.
    vector<size_t> indices;
    for (size_t i = 0; i < m; i++) {
        indices.push_back(i);
    }
    random_shuffle(indices.begin(), indices.end());
    return indices;
}

template<typename T>
void print_vector(vector<T> vec) {
    for (auto& val: vec) {
        cout << val << " ";
    }
    cout << endl;
}

void print_parts(vector<vector<float>> data, size_t parts_cnt, size_t n, size_t m) {
    for (size_t i = 0; i < parts_cnt;i++) {
        print_data(data[i], n, m / parts_cnt);
        cout << endl;
    }
}

void apply_permutation(float* vec, vector<size_t> indices) {
    for (size_t i = 0; i < indices.size(); i++) {
        size_t current = i;
        while (i != indices[current]) {
            int next = indices[current];
            swap(vec[current], vec[next]);
            indices[current] = current;
            current = next;
        }
        indices[current] = current;
    }
}

// TODO: Call it "subspace" and make class for storing matrix - different sizes!
vector<vector<float>> make_parts(vector<float> data, int parts_number, size_t n, size_t m) {
    vector<vector<float>> result(parts_number);
    int length = m / parts_number;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            result[j/length].push_back(data[i*m+j]);
        }
    }
    return result;
}

// TODO: same
vector<vector<float>> make_query_parts(vector<float> query, int parts_number) {
    int length = query.size() / parts_number;
    vector<vector<float>> result(parts_number);
    for(size_t i = 0; i < query.size(); i++) {
        result[i/length].push_back(query[i]);
    }
    return result;
}

// For each vector, find closest centroid and store its index in assignments.
void assign(float* vectors, size_t n, size_t d, size_t k, size_t* assignments, float* centroids) {
    for (size_t i = 0; i < n; i++) {
        float best = numeric_limits<float>::max();
        float dist = 0;

        for (size_t j = 0; j < k; j++) {
            dist = faiss::fvec_inner_product(vectors + (i*d), centroids + (j*d), d);
            if (best > dist) {
                assignments[i] = j;
                best = dist;
            }
        }
    }
}

void perform_kmeans(float* vectors, size_t n, size_t d, size_t k, size_t* assignments, float* centroids) {
    faiss::kmeans_clustering(d, n, k, vectors, centroids);
    assign(vectors, n, d, k, assignments, centroids);
}

vector<float> build_table(size_t k, vector<kmeans_result> data, vector<vector<float>> query, size_t m) {
    size_t length = m / data.size();
    vector<float> table;
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < data.size(); j++) {
            // TODO: is this correct? I think it should be i*length or sth here.
            float inner_product = faiss::fvec_inner_product(&data[j].centroids[i], query[j].data(), length);
            table.push_back(inner_product);
        }
    }
    return table;
}

int choose_vector_index(vector<float> inner_table, vector<kmeans_result> data, size_t k, size_t n) {
    vector<float> results;
    for (size_t i = 0; i < n; i++) {
        float partialResult = 0;
        for (size_t j = 0; j < data.size(); j++) {
            partialResult += inner_table[data[j].assigments[i] * k + j];
        }
        results.push_back(partialResult);
    }
    vector<float>::iterator max = max_element(results.begin(), results.end());
    return distance(results.begin(), max);
}

int main_quantization() {
    int parts_number=2;
    size_t n,m;
    size_t k = 1; // Centroid count.
    vector<float> query = {2, 3, 4, 5, 6, 7, 8, 9};
    vector<float> data = load_data("test.txt", &n, &m);
    print_data(data, n, m);
    vector<size_t> indices = prepare_indices_vector(8);
    print_vector(indices);
    for(size_t i = 0; i < n; i++) {
        apply_permutation(&data[m*i], indices);
    }
    apply_permutation(&query[0], indices);
    cout << endl;
    //print_data(data,n,m);
    print_vector(query);
    cout << endl;
    vector<vector<float>> parts = make_parts(data, parts_number, n, m);
    vector<vector<float>> query_parts = make_query_parts(query, parts_number);
    //print_parts(parts,parts_number,n,m);
    cout << endl;
    print_parts(query_parts, parts_number, 1, m);
    vector<kmeans_result> kmeans(parts_number);
    for(int i = 0; i < parts_number; i++) {
        kmeans[i].centroids = vector<float>(k*m);
        kmeans[i].assigments = vector<size_t>(n);
        perform_kmeans(parts[i].data(), n, m, k, kmeans[i].assigments.data(), kmeans[i].centroids.data());
    }
    vector<float>query_table = build_table(k, kmeans, query_parts, m);
    print_data(query_table, k, parts_number);
    cout << endl;
    cout << choose_vector_index(query_table, kmeans, k, n);
    return 0;
}
