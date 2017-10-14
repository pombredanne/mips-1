

COMPILE=g++ -I. -std=c++11 -fopenmp

all: dirs bin/quantization bin/alsh

bin/quantization: src/quantization.cpp
	${COMPILE} $^ -o $@

bin/alsh: src/alsh.cpp
	${COMPILE} $^ -o $@

dirs:
	mkdir -p bin

clean:
	rm -rf bin
