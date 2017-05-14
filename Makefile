all: k-means-mips
k-means-mips: main.cpp
	g++ -Wall -Wextra -o k-means-mips main.cpp
clean:
	rm k-means-mips
