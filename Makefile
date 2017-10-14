

COMPILE=g++ -O2 -I. -std=c++11 -fopenmp -Lfaiss -lfaiss -lopenblas

all: dirs bin/quantization bin/alsh

bin/quantization: src/quantization.cpp faiss/libfaiss.a
	${COMPILE} $^ -o $@

bin/alsh: src/alsh.cpp faiss/libfaiss.a
	${COMPILE} $^ -o $@

faiss/libfaiss.a:
	(cd faiss; make -j 4)

dirs:
	mkdir -p bin

clean:
	rm -rf bin
