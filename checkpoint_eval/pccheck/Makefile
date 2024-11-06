CC=g++
CFLAGS=-pthread -I/usr/local/cuda-12.1/include/ -L/usr/local/cuda-12.1/lib64/ -latomic -lpmem -lcudart -O3 -march=native -mtune=native -ffast-math

main.o: main.cpp
	$(CC) -c -fPIC main.cpp -o main.o

main: main.o FAAQueue.h
	$(CC) -o main main.o $(CFLAGS)

main_ssd.o: main_ssd.cpp
	$(CC) -c -fPIC main_ssd.cpp -o main_ssd.o $(CFLAGS)

main_ssd_memory.o: main_ssd_memory.cpp
	$(CC) -c -fPIC main_ssd_memory.cpp -o main_ssd_memory.o $(CFLAGS)

libtest.so: main.o FAAQueue.h
	$(CC) -shared -o libtest.so main.o $(CFLAGS)

libtest_ssd.so: main_ssd_memory.o FAAQueue.h DRAMAlloc.h
	$(CC) -shared -o libtest_ssd.so main_ssd_memory.o $(CFLAGS)

all: libtest.so libtest_ssd.so

clean:
	rm -rf *.o *.so
