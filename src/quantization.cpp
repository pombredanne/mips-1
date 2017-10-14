// MIPS_Quantization.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> 
#include <numeric>  
#include "faiss/faiss/utils.h"
#include "faiss/faiss/Clustering.h"

using namespace std;

struct kmeansResult
{
	vector<vector<int> > centroids;
	vector<int> assignedCentroids;
};
vector<vector<int> > loadData(string filename)
{
	vector<vector<int> > vectors;
	ifstream infile(filename);
	//infile.open(filename);
	int m, n;//liczba wektorow, dlugsc wektora
	infile >> m >> n;
	cout << m << " " << n << endl;
	for (int i = 0; i < m; i++)
	{
		vector<int> vector;
		for (int j = 0; j < n; j++)
		{
			int tmp;
			infile >> tmp;
			vector.push_back(tmp);
		}
		vectors.push_back(vector);
	}
	infile.close();
	return vectors;
}

void printData(vector<vector<int> > vectors)
{
	for (int i = 0; i < vectors.size(); i++)
	{
		for (int j = 0; j < vectors[i].size(); j++)
		{
			cout << vectors[i][j] << " ";
		}
		cout << endl;
	}
}

vector<int> prepareIndicesVector(int m)
{
	vector<int> indices;
	for (int i = 0; i < m; i++)
	{
		indices.push_back(i);
	}
	random_shuffle(indices.begin(), indices.end());
	return indices;
}
void printVector(vector<int>vector)
{
	for (int i = 0; i < vector.size(); i++)
	{
		cout << vector[i] << " ";
	}
	cout << endl;
}

void printParts(vector<vector<vector<int> > > data)
{
	for (int i = 0; i < data.size(); i++)
	{
		for (int j = 0; j < data[i].size(); j++)
		{
			for (int k = 0; k < data[i][j].size(); k++)
			{
				cout << data[i][j][k] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}
void applyPermutation(vector<int>& vec, vector<int> indices)
{
	for (size_t i = 0; i < indices.size(); i++)
	{
		int current = i;
		while (i != indices[current])
		{
			int next = indices[current];
			swap(vec[current], vec[next]);
			indices[current] = current;
			current = next;
		}
		indices[current] = current;
	}
}

vector<vector<vector<int> > > makeParts(vector<vector<int> > data, int numberOfParts)
{
	vector < vector<vector<int> > > partialData;
	int length = data[0].size() / numberOfParts;
	vector<vector<int> > tmpVector;
	for (int iterator = 0; iterator < data[0].size(); iterator+=length)
	{
		for (int i = 0; i < data.size(); i++)
		{
			vector<int> tmpIntVector;

			for (int j = iterator; j < iterator+length; j++)
			{
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
kmeansResult DoKmeans()
{
	faiss::kmeans_clustering()
	return kmeansResult();
}

vector<vector<int> > BuildTable(int numberOfCentroids, vector<kmeansResult> data, vector<vector<int> > query)
{
	vector<vector<int> > table;
	for (int i = 0; i < numberOfCentroids; i++)
	{
		vector<int>tmpVector;
		for (int j = 0; j < data.size(); j++)
		{
			int innerProduct = inner_product(data[j].centroids[i].begin(), data[j].centroids[i].end(), query[j].begin(), 0.0);
			tmpVector.push_back(innerProduct);
		}
		table.push_back(tmpVector);
		tmpVector.clear();
	}
	return table;
}
int chooseVectorIndex(vector<vector<int> > innerTable, vector<kmeansResult> data)
{
	vector<int> results;
	for (int i = 0; i < data[0].assignedCentroids.size(); i++)
	{
		int partialResult = 0;
		for (int j = data.size(); j < data.size(); j++)
		{
			partialResult += innerTable[data[j].assignedCentroids[i]][j];
		}
		results.push_back(partialResult);
	}
	vector<int>::iterator max = max_element(results.begin(), results.end());
	return distance(results.begin(), max);
}
int main()
{
	vector<vector<int> > data = loadData("test.txt");
	printData(data);
	vector<int> indices = prepareIndicesVector(8);
	printVector(indices);
	applyPermutation(data[0], indices);
	cout << endl;
	printData(data);
	cout << endl;
	vector<vector<vector<int> > > parts = makeParts(data, 4);
	printParts(parts);
	return 0;
}

