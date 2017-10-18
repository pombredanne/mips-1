
LD_FLAGS=-Lfaiss -lfaiss -lopenblas

CPP_FLAGS= -O2 -I. 
CPP_FLAGS+= -Wall -Wextra -Wno-unused-result
CPP_FLAGS+= -std=c++11 -fopenmp

COMPILE=g++ $(CPP_FLAGS)


SOURCES=$(shell find src -name \*.cpp)
OBJECTS=$(subst .cpp,.o,$(subst src,build,$(SOURCES)))
DEPS=$(subst .o,.d,$(OBJECTS))

TESTSRC=$(shell find tests -name \*.cpp)
TESTBINS=$(subst .cpp,,$(subst tests,bin,$(TESTSRC)))

all: dirs $(TESTBINS)

$(OBJECTS): build/%.o: src/%.cpp
	$(COMPILE) -MMD -MP -c $< -o $@

bin/%: tests/%.cpp faiss/libfaiss.a $(OBJECTS)
	$(COMPILE) $^ -o $@ $(LD_FLAGS)

faiss/libfaiss.a:
	(cp makefile.inc faiss/makefile.inc)
	(cd faiss; make -j 4)

dirs:
	mkdir -p bin build

clean:
	rm -rf bin build
	(cd faiss; make clean)

-include $(DEPS)
