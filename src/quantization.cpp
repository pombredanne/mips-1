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

vector<size_t> prepare_permutation(size_t m) {
    // TODO: Don't shuffle, but rotate randomly.
    // TODO: Also, probably use some better source of randomness.
    vector<size_t> permutation;
    for (size_t i = 0; i < m; i++) {
        permutation.push_back(i);
    }
    random_shuffle(permutation.begin(), permutation.end());
    return permutation;
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

void apply_permutation(float* vec, vector<size_t> permutation) {
    for (size_t i = 0; i < permutation.size(); i++) {
        size_t current = i;
        while (i != permutation[current]) {
            int next = permutation[current];
            swap(vec[current], vec[next]);
            permutation[current] = current;
            current = next;
        }
        permutation[current] = current;
    }
}

vector<FloatMatrix> make_parts(const FloatMatrix& data, size_t parts_count) {
    vector<FloatMatrix> result(parts_count);
    // Ceil division.
    size_t len = (data.vector_length + parts_count - 1) / parts_count;
    for (size_t i = 0; i < parts_count; i++) {
        if (i != parts_count - 1 || data.vector_length % parts_count == 0) {
            result[i].resize(data.vector_count(), len);
        }
        else {
            // If vector length is not a multiple of parts_count, the last part
            // will have less elements.
            result[i].resize(data.vector_count(), data.vector_length % len);
        }
        cout << "Length of part " << i << ": " << result[i].vector_length << endl;
    }
    for (size_t vec = 0; vec < data.vector_count(); vec++) {
        for (size_t ind = 0; ind < data.vector_length; ind++) {
            result[ind / len].at(vec, ind % len) = data.at(vec, ind);
        }
    }
    return result;
}

// Returns best guess of index of vector closest to query.
size_t answer_query(
        vector<kmeans_result>& kmeans, vector<FloatMatrix>& queries, size_t query_number) {

    assert(kmeans.size() == queries.size());
    assert(kmeans.size() > 0);
    size_t part_count = kmeans.size();
    size_t centroid_count = kmeans[0].centroids.vector_count();
    size_t vector_count = kmeans[0].assignments.size();

    FloatMatrix table;
    table.resize(part_count, centroid_count);
    for (size_t part = 0; part < part_count; part++) {
        size_t part_length = kmeans[part].centroids.vector_length;
        assert(part_length == queries[part].vector_length);

        for (size_t j = 0; j < centroid_count; j++) {
            float product = faiss::fvec_inner_product(
                    kmeans[part].centroids.row(j),
                    queries[part].row(query_number),
                    part_length);
            table.at(part, j) = product;
        }
    }

    vector<float> results;
    for (size_t vec = 0; vec < vector_count; vec++) {
        float sum = 0;
        for (size_t part = 0; part < part_count; part++) {
            sum += table.at(part, kmeans[part].assignments[vec]);
        }
        results.push_back(sum);
    }
    vector<float>::iterator max = max_element(results.begin(), results.end());
    // debug:
        float sum = 0;
        size_t i = max - results.begin();
        for (size_t part = 0; part < part_count; part++) {
            sum += table.at(part, kmeans[part].assignments[i]);
            int cc = kmeans[part].assignments[i];
            cout << "mips part " << cc << " " << table.at(part, cc) << endl;
        }
        cout << "MIPS: " << sum << endl;

    return distance(results.begin(), max);
}

int main_quantization() {
    int parts_count=2;
    size_t k = 3; // Centroid count.
    FloatMatrix data = load_text_file<float>("input");
    FloatMatrix queries = load_text_file<float>("queries");
    std::cout << "Data:\n";
    data.print();
    std::cout << "Queries:\n";
    queries.print();
    vector<size_t> permutation = prepare_permutation(data.vector_length);
    std::cout << "Permutation:\n";
    print_vector(permutation);

    for (size_t i = 0; i < data.vector_count(); i++) {
        apply_permutation(data.row(i), permutation);
    }
    for (size_t i = 0; i < queries.vector_count(); i++) {
        apply_permutation(queries.row(i), permutation);
    }
    std::cout << "Permuted data:\n";
    data.print();
    std::cout << "Permuted queries:\n";
    queries.print();

    auto parts = make_parts(data, parts_count);
    print_parts(parts);
    auto query_parts = make_parts(queries, parts_count);
    print_parts(query_parts);

    vector<kmeans_result> kmeans(parts_count);
    for(int i = 0; i < parts_count; i++) {
        cout << "Clustering for part " << i << endl;
        kmeans[i] = perform_kmeans(parts[i], k);
    }
    std::cout << "Preprocessing phase finished." << std::endl;

    for (size_t q = 0; q < queries.vector_count(); q++) {
        std::cout << "Query " << q << std::endl;
        cout << answer_query(kmeans, query_parts, q) << endl;
    }
    return 0;
}
