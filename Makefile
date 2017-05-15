all: k-means-mips
k-means-mips: main.cpp
	g++ -Wall -Wextra -o k-means-mips main.cpp -std=c++03
clean:
	rm k-means-mips
