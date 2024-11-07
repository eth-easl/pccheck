#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <cuda.h>
#include "checkp_func.h"

// extern void checkpoint_util(char* filename, float* ar, size_t ckp_size);

void checkpoint()
{
        checkpoint_func();
}

void end_checkp()
{
        finish_checkpoint();
}

void start_checkp(const char *filename, unsigned long ckp_size, void **ar_ptrs, size_t *sizes, int num_to_register)
{
        init_checkpoint(filename, ckp_size, ar_ptrs, sizes, num_to_register);
}

extern "C"
{
        void save_gpm()
        {
                checkpoint();
        }

        void finish()
        {
                end_checkp();
        }

        void init(const char *filename, unsigned long ckp_size, void **ar_ptrs, size_t *sizes, int num_to_register)
        {
                start_checkp(filename, ckp_size, ar_ptrs, sizes, num_to_register);
        }
}
int main(int argc, char *argv[])
{

        int num_floats = 1000;
        float *ar = NULL;

        size_t tot_size = num_floats * sizeof(float);
        float tot_size_gb = tot_size * 1.0 / 1000000000;

        printf("Size is: %ld bytes, %f GB\n", tot_size, tot_size_gb);

        // checkpoint("test2", ar, tot_size);

        // checkpoint_func("test2", ar, tot_size);
}
