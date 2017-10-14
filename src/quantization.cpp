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

struct kmeansResult {
	vector<vector<int> > centroids;
	vector<int> assignedCentroids;
};

vector<vector<int> > load_data(string filename) {
	vector<vector<int> > vectors;
	ifstream infile(filename);
	int m, n;//liczba wektorow, dlugsc wektora
	infile >> m >> n;
	cout << m << " " << n << endl;
	for (int i = 0; i < m; i++) {
		vector<int> vector;
		for (int j = 0; j < n; j++)	{
			int tmp;
			infile >> tmp;
			vector.push_back(tmp);
		}
		vectors.push_back(vector);
	}
	infile.close();
	return vectors;
}

void printData(vector<vector<int> > vectors) {
	for (auto& vec: vectors) {
		for (auto& val: vec) {
			cout << val;
		}
	}
}

vector<size_t> prepareIndicesVector(int m) {
	vector<size_t> indices;
	for (int i = 0; i < m; i++) {
		indices.push_back(i);
	}
	random_shuffle(indices.begin(), indices.end());
	return indices;
}

void printVector(vector<size_t> vec) {
	for (auto& val: vec) {
		cout << val << " ";
	}
	cout << endl;
}

void printParts(vector<vector<vector<int> > > data) {
	for (auto& mat: data) {
		for (auto& vec: mat) {
			for (auto& val: vec) {
				cout << val << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

void applyPermutation(vector<int>& vec, vector<size_t> indices) {
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

vector<vector<vector<int> > > makeParts(vector<vector<int> > data, int numberOfParts) {
	vector < vector<vector<int> > > partialData;
	int length = data[0].size() / numberOfParts;
	vector<vector<int> > tmpVector;

	for (size_t it = 0; it < data[0].size(); it+=length) {
		for (size_t i = 0; i < data.size(); i++) {
			vector<int> tmpIntVector;
			for (size_t j = it; j < it+length; j++) {
				tmpIntVector.push_back(data[i][j]);				
			}
			tmpVector.push_back(tmpIntVector);
			tmpIntVector.clear();
		}
		partialData.push_back(tmpVector);
		tmpVector.clear();
	}
	
	return partialData;
}

kmeansResult DoKmeans() {
	return kmeansResult();
}

vector<vector<int> > BuildTable(
		int numberOfCentroids, vector<kmeansResult> data, vector<vector<int> > query) {
	vector<vector<int> > table;
	for (int i = 0; i < numberOfCentroids; i++) {
		vector<int>tmpVector;
		for (size_t j = 0; j < data.size(); j++) {
			int innerProduct = inner_product(data[j].centroids[i].begin(), data[j].centroids[i].end(), query[j].begin(), 0.0);
			tmpVector.push_back(innerProduct);
		}
		table.push_back(tmpVector);
		tmpVector.clear();
	}
	return table;
}

int chooseVectorIndex(vector<vector<int> > innerTable, vector<kmeansResult> data) {
	vector<int> results;
	for (size_t i = 0; i < data[0].assignedCentroids.size(); i++) {
		int partialResult = 0;
		for (size_t j = data.size(); j < data.size(); j++) {
			partialResult += innerTable[data[j].assignedCentroids[i]][j];
		}
		results.push_back(partialResult);
	}
	vector<int>::iterator max = max_element(results.begin(), results.end());
	return distance(results.begin(), max);
}

int main() {
	vector<vector<int> > data = load_data("test.txt");
	printData(data);
	vector<size_t> indices = prepareIndicesVector(8);
	printVector(indices);
	applyPermutation(data[0], indices);
	cout << endl;
	printData(data);
	cout << endl;
	vector<vector<vector<int> > > parts = makeParts(data, 4);
	printParts(parts);
	return 0;
}

