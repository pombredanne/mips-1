

COMPILE=g++ -I. -std=c++11

all: dirs bin/quantization

bin/quantization: src/quantization.cpp
	${COMPILE} $^ -o $@

dirs:
	mkdir -p bin

clean:
	rm -rf bin
