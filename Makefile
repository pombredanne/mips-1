all: k-means-mips
k-means-mips: main.cpp
	g++ -Wall -Wextra -o k-means-mips main.cpp faiss/*.o \
	   	-std=c++03 -g -Dnullptr=NULL -Doverride= -DFINTEGER=int -fopenmp -lopenblas -L/usr/lib
clean:
	rm -rf k-means-mips
