
#ifndef _DRAM_ALLOCATION_H_
#define _DRAM_ALLOCATION_H_

#include <atomic>
#include <iostream>
#include <cuda_runtime.h>
#include "FAAQueueAdd.h"

class DRAMAlloc
{
public:
    // size should be correlated to the size of the batch
    void alloc(size_t num, size_t size)
    {
        // each element needs to hold the parallel_item and the batch_num?
        //  TODO: is num the number of batches, and size the batch size (in floats)?
        // size_t total_element = size + 2*sizeof (int);

        cudaError_t err = cudaMallocHost((void **)&array, num * size * 4);
        assert(err == cudaSuccess);
    }

    // insert all free elements to the queue
    void initialize(size_t num, size_t size)
    {
        // size_t element_size = size + 2*sizeof (int);
        for (int i = 0; i < num; i++)
        {
            free_add.enqueue(array + (i * size), 0);
        }
    }

    // get free address
    float *get_add(size_t id)
    {
        while (true)
        {
            float *add = free_add.dequeue(id);
            if (add != nullptr)
                return add;
        }
    }

    // put free address
    void put_add(float *add, size_t id)
    {
        free_add.enqueue(add, 0);
    }

private:
    FAAArrayQueueAdd<float *> free_add;
    float *array;
};

#endif
