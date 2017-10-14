// MIPS_Quantization.cpp : Defines the entry point for the console application.
//

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
	vector<vector<float> > centroids;
	vector<float> assignedCentroids;
};

vector<float> load_data(string filename, int *n, int *m) {//numberOfVectors, lengthOfVector
	vector<float> data;
	ifstream infile(filename);
	infile >> *n >> *m;
	cout << *n << " " << *m << endl;
	for (size_t i = 0; i < *n * *m; i++) {
		float tmp;
		infile >> tmp;
		data.push_back(tmp);
	}
	infile.close();
	return data;
}

void print_data(vector<float> data, int n, int m) {
	for (int i = 0; i<n; i++) {
		for(int j = 0; j<m; j++){

			cout << data[i*m+j]<<" ";
		}
		cout<<endl;
	}
}

vector<size_t> prepare_indices_vector(int m) {
	vector<size_t> indices;
	for (int i = 0; i < m; i++) {
		indices.push_back(i);
	}
	random_shuffle(indices.begin(), indices.end());
	return indices;
}

void print_vector(vector<size_t> vec) {
	for (auto& val: vec) {
		cout << val << " ";
	}
	cout << endl;
}

void print_parts(vector<vector<float> >  data, int parts_number, int n, int m) {
	for (int i = 0;i<parts_number;i++) {
		print_data(data[i],n,m/parts_number);
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

vector<vector<float> >  make_parts(vector<float> data, int numberOfParts, int n, int m) {
	vector<vector<float> > result(numberOfParts);
	int length = m / numberOfParts;
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < m; j++) {
			result[j/length].push_back(data[i*m+j]);
		}
	}
	return result;
}
//
//void assign(float *vectors, size_t n, size_t d, size_t k, size_t *assignments, float *centroids) {
//	for (size_t i=0; i<n; i++) {
//		float best = numeric_limits<float>::max();
//		float dist = 0;
//
//		for (size_t j=0; j<k; j++) {
//			dist = faiss::fvec_inner_product(vectors + (i*d), centroids + (j*d), d);
//			if (best > dist) {
//				assignments[i] = j;
//				best = dist;
//			}
//		}
//	}
//}
//void perform_kmeans(float *vectors, size_t n, size_t d, size_t k, size_t *assignments, float *centroids) {
//	faiss::kmeans_clustering(d, n, k, vectors, centroids);
//	assign(vectors, n, d, k, assignments, centroids);
//}

vector<vector<float> > build_table(
		int numberOfCentroids, vector<kmeans_result> data, vector<vector<int> > query) {
	vector<vector<float> > table;
	for (int i = 0; i < numberOfCentroids; i++) {
		vector<float>tmpVector;
		for (size_t j = 0; j < data.size(); j++) {
			int innerProduct = inner_product(data[j].centroids[i].begin(), data[j].centroids[i].end(), query[j].begin(), 0.0);
			tmpVector.push_back(innerProduct);
		}
		table.push_back(tmpVector);
		tmpVector.clear();
	}
	return table;
}

int choose_vector_index(vector<vector<float> > innerTable, vector<kmeans_result> data) {
	vector<float> results;
	for (size_t i = 0; i < data[0].assignedCentroids.size(); i++) {
		int partialResult = 0;
		for (size_t j = data.size(); j < data.size(); j++) {
			partialResult += innerTable[data[j].assignedCentroids[i]][j];
		}
		results.push_back(partialResult);
	}
	vector<float>::iterator max = max_element(results.begin(), results.end());
	return distance(results.begin(), max);
}

int main() {
	int n,m;
	vector<float> data = load_data("test.txt",&n,&m);
	print_data(data,n,m);
	vector<size_t> indices = prepare_indices_vector(8);
	print_vector(indices);
	for(int i = 0; i<n;i++)
	{
		apply_permutation(&data[m*i], indices);
	}
	cout << endl;
	print_data(data,n,m);
	cout << endl;
	int numberOfParts=2;
	vector<vector<float> >  parts = make_parts(data,numberOfParts,n,m);
	//perform_kmeans(parts[0],)
	print_parts(parts,numberOfParts,n,m);
	return 0;
}

