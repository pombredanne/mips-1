all: k-means-mips

k-means-mips: main.cpp
	g++ -Wall -Wextra -o k-means-mips main.cpp faiss/*.o \
	   	-std=c++03 -g -Dnullptr=NULL -Doverride= -DFINTEGER=int -fopenmp -lopenblas -L/usr/lib -O3 -funroll-loops

debug: main.cpp
	g++ -Wall -Wextra -o k-means-mips main.cpp faiss/*.o \
	   	-std=c++03 -g -Dnullptr=NULL -Doverride= -DFINTEGER=int -fopenmp -lopenblas -L/usr/lib

mkl: main.cpp
	g++ -Wall -Wextra -o k-means-mips main.cpp faiss/*.o \
	   	-std=c++03 -g -Dnullptr=NULL -Doverride= -DFINTEGER=int -O3 -funroll-loops \
			-L/usr/lib -L${HOME}/anaconda3/lib/ -fopenmp -lmkl_core -lmkl_intel_ilp64 -lmkl_intel_thread -ldl -lpthread -liomp5 

clean:
	rm -rf k-means-mips
