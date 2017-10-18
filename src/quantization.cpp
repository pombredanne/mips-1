#include "common.h"

#include "faiss/utils.h"
#include "faiss/Clustering.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> 
#include <numeric>  

using namespace std;

typedef FlatMatrix<float> FloatMatrix;

struct kmeans_result {
    FloatMatrix centroids;
    vector<size_t> assignments;
};

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

void print_parts(const vector<FloatMatrix>& parts) {
    for (const auto& mat: parts) {
        mat.print();
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

vector<FloatMatrix> make_parts(const FloatMatrix& data, size_t parts_count) {
    vector<FloatMatrix> result(parts_count);
    // Ceil division.
    size_t len = (data.vector_length + parts_count - 1) / parts_count;
    for (size_t i = 0; i < parts_count; i++) {
        if (i != parts_count - 1) {
            result[i].resize(data.vector_count(), len);
        }
        else {
            result[i].resize(data.vector_count(), data.vector_length % len);
        }
    }
    for (size_t vec = 0; vec < data.vector_count(); vec++) {
        for (size_t ind = 0; ind < data.vector_length; ind++) {
            result[ind / len].at(vec, ind % len) = data.at(vec, ind);
        }
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

kmeans_result perform_kmeans(FloatMatrix& matrix, size_t k) {
    kmeans_result kr;
    kr.centroids.resize(k, matrix.vector_length);
    kr.assignments.resize(matrix.vector_count());
    faiss::kmeans_clustering(
        matrix.vector_length, 
        matrix.vector_count(), 
        k, 
        &matrix.at(0, 0), 
        &kr.centroids.at(0, 0));
    // TODO XXX
    //assign(vectors, n, d, k, assignments, centroids);
    return kr;
}

// Returns best guess of index of vector closest to query.
size_t answer_query(
        vector<kmeans_result>& kmeans, vector<FloatMatrix>& queries, size_t query_number) {

    assert(kmeans.size() == queries.size());
    assert(kmeans.size() > 0);
    size_t parts_count = kmeans.size();
    size_t centroids_count = kmeans[0].centroids.vector_count();

    FloatMatrix table;
    table.resize(parts_count, centroids_count);
    for (size_t part = 0; part < parts_count; part++) {
        size_t part_length = kmeans[part].centroids.vector_length;
        assert(part_length == queries[part].vector_length);

        for (size_t j = 0; j < centroids_count; j++) {
            float product = faiss::fvec_inner_product(
                    &kmeans[part].centroids.at(j, 0),
                       &queries[part].at(query_number, 0),
                       part_length);
            table.at(part, j) = product;
        }
    }

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
    int parts_count=2;
    size_t k = 3; // Centroid count.
    FloatMatrix data = load_file<float>("input");
    FloatMatrix queries = load_file<float>("queries");
    data.print();
    vector<size_t> indices = prepare_indices_vector(6);
    print_vector(indices);

    for (size_t i = 0; i < data.vector_count(); i++) {
        apply_permutation(&data.at(i, 0), indices);
    }
    for (size_t i = 0; i < queries.vector_count(); i++) {
        apply_permutation(&queries.at(i, 0), indices);
    }

    auto parts = make_parts(data, parts_count);
    auto query_parts = make_parts(queries, parts_count);

    print_parts(query_parts);
    vector<kmeans_result> kmeans(parts_count);
    for(int i = 0; i < parts_count; i++) {
        kmeans[i] = perform_kmeans(parts[i], k);
    }
    std::cout << "Preprocessing phase finished." << std::endl;

    for (size_t q = 0; q < queries.vector_count(); q++) {
        std::cout << "Query " << q << std::endl;
        cout << answer_query(kmeans, query_parts, q) << endl;
    }
    return 0;
}
