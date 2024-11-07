

#include <iostream>
#include "FAAQueue.h"
#include "DRAMAlloc.h"

using namespace std;

int main(int argc, char **argv)
{
    DRAMAlloc dram;
    FAAArrayQueue<int> queue;
    queue.enqueue(0, 0);
    int res = queue.dequeue(0);
    cout << res << endl;
    res = queue.dequeue(0);
    cout << res << endl;

    dram.alloc(1, 1);
    dram.initialize(1, 1);
    float *add = dram.get_add(0);
    cout << add << endl;
    // float* add1 = dram.get_add(0);
    // cout << "add" << endl;
    dram.put_add(add, 0);
    add = dram.get_add(0);
    cout << add << endl;
    return 0;
}
