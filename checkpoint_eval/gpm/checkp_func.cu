#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>

#define PERSIST_TIME
double persist_time = 0;

#include "libgpmcp.cuh"
//#include "bandwidth_analysis.cuh"
#include "checkp_func.h"
///////////////////////////////////////////////////////////////////////////////////////////
// Definitions and helper utilities

// Block width for CUDA kernels
#define BW 128
#ifndef CP_ITER
#define CP_ITER 5
#endif
#define NTHREADS_PER_CPBLOCK 512

#ifdef USE_GFLAGS
#include <gflags/gflags.h>

#ifndef _WIN32
#define gflags google
#endif
#else

// Constant versions of gflags
#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag((default_value))
#endif

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s)                                     \
    do                                                    \
    {                                                     \
        std::stringstream _where, _message;               \
        _where << __FILE__ << ':' << __LINE__;            \
        _message << std::string(s) + "\n"                 \
                 << __FILE__ << ':' << __LINE__;          \
        std::cerr << _message.str() << "\nAborting...\n"; \
        cudaDeviceReset();                                \
        exit(1);                                          \
    } while (0)

#define checkCUDNN(status)                                              \
    do                                                                  \
    {                                                                   \
        std::stringstream _error;                                       \
        if (status != CUDNN_STATUS_SUCCESS)                             \
        {                                                               \
            _error << "CUDNN failure: " << cudnnGetErrorString(status); \
            FatalError(_error.str());                                   \
        }                                                               \
    } while (0)

#define checkCudaErrors(status)                   \
    do                                            \
    {                                             \
        std::stringstream _error;                 \
        if (status != 0)                          \
        {                                         \
            _error << "Cuda failure: " << status; \
            FatalError(_error.str());             \
        }                                         \
    } while (0)

///////////////////////////////////////////////////////////////////////////////////////////
// Command-line flags

// Application parameters
// DEFINE_int32(gpu, 0, "The GPU ID to use");
// DEFINE_int32(iterations, 1, "Number of iterations for training");
gpmcp *cp_dnn;

void checkpoint(gpmcp *cp_dnn)
{

    auto start_ts = std::chrono::high_resolution_clock::now();
    long long int timings = 0;

    // sleep here for a while
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    gpmcp_checkpoint(cp_dnn, 0);
    checkCudaErrors(cudaDeviceSynchronize());
    timings += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
    printf("Checkpoint took %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count());

    // auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_ts).count();
    // printf("Total time is %f us, %lld ns ---- \n", dur_us*1.0, timings);
}

void init_checkpoint(const char *filename, unsigned long ckp_size, void **ar_ptrs, size_t *sizes, int num_to_register)
{

    checkCudaErrors(cudaSetDevice(0));
    // ddio_off();
    cp_dnn = gpmcp_create(filename, ckp_size, num_to_register, 1);
    // ddio_on();
    register_many(ar_ptrs, sizes, num_to_register);
}

void register_many(void **ar_ptrs, size_t *sizes, int num_to_register)
{

    for (int i = 0; i < num_to_register; i++)
    {
        //printf("----------------- Register: %d, %p, %d\n", i, ar_ptrs[i], sizes[i]);
        //gpmcp_print((float *)(ar_ptrs[i]));
        gpmcp_register(cp_dnn, ar_ptrs[i], sizes[i], 0);
    }
    // for (int i=0; i<num_to_register; i++) {
    //     printf("%d, %d\n", i, cp_dnn->node_size[i]);
    // }
}

void checkpoint_func()
{

    checkpoint(cp_dnn);
}

void finish_checkpoint()
{
    // ddio_on();
    gpmcp_close(cp_dnn);
}

void dummy()
{
}
