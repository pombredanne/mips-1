

COMPILE=g++ -I.

all: dirs bin/quantization

bin/quantization: src/quantization.cpp
	${COMPILE} $^ -o $@

dirs:
	mkdir -p bin
