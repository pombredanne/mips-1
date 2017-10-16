
LIBS=-Lfaiss -lfaiss -lopenblas

COMPILE=g++ -O2 -I. -Wall -Wextra -std=c++11 -fopenmp

all: dirs bin/quantization bin/alsh

bin/quantization: src/quantization.cpp faiss/libfaiss.a
	${COMPILE} $^ -o $@ ${LIBS}

bin/alsh: src/alsh.cpp faiss/libfaiss.a
	${COMPILE} $^ -o $@ ${LIBS}

faiss/libfaiss.a:
	(cp makefile.inc faiss/makefile.inc)
	(cd faiss; make -j 4)

dirs:
	mkdir -p bin

clean:
	rm -rf bin
	(cd faiss; make clean)
