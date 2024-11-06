#pragma once
extern "C"
{
//#include "change-ddio.h"
}
#include "libgpm.cuh"
#include <stdio.h>
#include <chrono>
#include <string>
#include <sys/mman.h>
#include <errno.h>

#define CACHE_LINE_SIZE 64

float total_map_time = 0.0;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

int fd;

static void FLUSH(void *p)
{
    asm volatile("clwb (%0)" ::"r"(p));
}

static void SFENCE()
{
    asm volatile("sfence" ::: "memory");
}

// Non-volatile metadata
struct gpmcp_nv
{
    long long elements;   // Maximum number of elements per group
    long long partitions; // Number of groups for the cp
    size_t size;          // Total size of data being cp
};

// Volatile metadata
struct gpmcp
{
    const char *path; // File path

    char *index; // Set of non-volatile indices
    void *start; // Pointer to start of non-volatile region

    void *start1;
    void *start2;
    void *mapped_addr;

    size_t tot_size;        // Total size, including shadow space and metadata
    size_t checkpoint_size; // Checkpoint size
    long long elements;     // Maximum number of elements per group
    long long partitions;   // Number of groups for the cp

    gpmcp_nv *cp; // Pointer to non-volatile metadata

    // Checkpoint entries
    void **node_addr;  // Set of starting addresses for different elements
    size_t *node_size; // Set of sizes of each element

    // Partition info
    long long *part_byte_size; // Set of cp starting addresses for each group
    long long *part_bytes;     // Size of contents in partition
    long long *part_elem_size; // Set indicating number of elements in each group
};

static __global__ void setup_cp(gpmcp *cp, long long size, long long elements, long long partitions)
{
    cp->cp->elements = elements;
    cp->cp->partitions = partitions;
    cp->cp->size = size;
}

static __global__ void setup_partitions(long long *byte_size, long long partitions, size_t size)
{
    long long ID = threadIdx.x + blockDim.x * blockIdx.x;
    for (long long i = ID; i < partitions; i += gridDim.x * blockDim.x)
    {
        byte_size[i] = i * size / partitions;
    }
}

static __host__ gpmcp *gpmcp_create(const char *path, size_t size, long long elements, long long partitions)
{
    printf("At create1, file name is %s, elements is %d\n", path, elements);
    gpmcp *cp;
    cudaMallocHost((void **)&cp, sizeof(gpmcp));
    cp->path = path;

    // Make all blocks of data equal sizes and 128-byte aligned
    // 4-byte alignment improves checkpoint throughput
    size += partitions - (size % partitions > 0 ? size % partitions : partitions);
    // 128 * elements to allow for 128-byte alignment
    size += (4 - (size / partitions % 4 > 0 ? size / partitions % 4 > 0 : 4) + 128 * elements) * partitions;
    // 128-byte align size
    size += 128 - (size % 128);

    // Header size + location bitmap
    size_t total_size = sizeof(gpmcp) + partitions;
    // Aligned 2 * Data size (2xsize for crash redundancy, (+size for extra metadata)
    total_size += 128 - total_size % 128 + 2 * size;

    total_size += 12288;
    cp->tot_size = total_size;

    // Map file
#ifdef NVM_ALLOC_GPU
    int is_pmem;
    size_t file_size;
    // Allocate metadata explicitly on CPU as gpm_map will allocate on GPU
    printf("Call pmem_map_file\n");
    char *cp_pointer = (char *)pmem_map_file(path, total_size - 2 * size, PMEM_FILE_CREATE, 0666, &file_size, &is_pmem);
    printf("Allocated CP metadata %ld size. Pmem? %d\n", file_size, is_pmem);
#else
    printf("Call gpm_map_file, with total_size %lu, size is %lu\n", total_size, size);
    void *mapped_addr;
    // Don't need to worry here, as gpm_map can be accessed from CPU
    gpm_map_file(path, total_size, 1, &mapped_addr);
    cp->mapped_addr = mapped_addr;
    char *cp_pointer = (char *)mapped_addr;
#endif

    gpmcp_nv *cp_nv = (gpmcp_nv *)cp_pointer;
    cp_pointer += sizeof(gpmcp_nv);
    cp->cp = cp_nv;

    void **node_addr;
    size_t *node_size;
    cudaMallocHost((void **)&node_addr, sizeof(void *) * elements * partitions);
    cudaMallocHost((void **)&node_size, sizeof(size_t) * elements * partitions);
    memset(node_addr, 0, sizeof(void *) * elements * partitions);
    memset(node_size, 0, sizeof(size_t) * elements * partitions);

    cp->node_addr = node_addr;
    cp->node_size = node_size;

    // align for disk msync
    cp_pointer += 4096 - (size_t)cp_pointer % 4096;
    cp->index = (char *)cp_pointer;
    cp_pointer += partitions;

    cp->cp->size = size;
    printf("Set size to %lu\n", cp->cp->size);
    cp->checkpoint_size = size;

#ifdef NVM_ALLOC_GPU
    size_t cp_size = 2 * size;
    void *addr;
    cp->start = (char *)gpm_map_file((std::string(path) + "_gpu").c_str(), cp_size, 1, &addr);
#else
    cp->start1 = cp_pointer;
#endif
    cp->elements = elements;

    // align for disk msync
    cp->start1 = (char *)(cp->start1) + 4096 - ((size_t)(cp->start1) % 4096);
    cp->start2 = (char *)cp->start1 + cp->cp->size;
    cp->start2 = (char *)(cp->start2) + 4096 - ((size_t)(cp->start2) % 4096);

    printf("%p, %p\n", cp->start1, cp->start2);

    cudaMallocHost((void **)&cp->part_byte_size, sizeof(long long) * partitions);
    cudaMallocHost((void **)&cp->part_elem_size, sizeof(long long) * partitions);
    cudaMallocHost((void **)&cp->part_bytes, sizeof(long long) * partitions);
    memset(cp->part_elem_size, 0, sizeof(long long) * partitions);
    memset(cp->part_bytes, 0, sizeof(long long) * partitions);
    setup_partitions<<<(partitions + 1023) / 1024, 1024>>>(cp->part_byte_size, partitions, size);
    printf("Size is %lu\n", cp->cp->size);

    cp->cp->elements = elements;
    cp->cp->partitions = partitions;
    cp->partitions = partitions;
    printf("Size is %lu\n", cp->cp->size);
    return cp;
}

static __host__ gpmcp *gpmcp_open(const char *path)
{

    printf("Inside gpmcp_open, for path %s\n", path);
    gpmcp *cp;
    cudaMallocHost((void **)&cp, sizeof(gpmcp));
    cp->path = path;

    size_t len = 0;
    void *addr;
    char *cp_pointer = (char *)gpm_map_file(path, len, false, &addr);

    printf("After map, cp_pointer is %p\n", cp_pointer);

    gpmcp_nv *cp_nv = (gpmcp_nv *)cp_pointer;
    cp_pointer += sizeof(gpmcp_nv);

    cp->tot_size = len;
    cp->cp = cp_nv;

    void **node_addr;
    size_t *node_size;
    cudaMallocHost((void **)&node_addr, sizeof(void *) * cp_nv->elements * cp_nv->partitions);
    cudaMallocHost((void **)&node_size, sizeof(size_t) * cp_nv->elements * cp_nv->partitions);
    memset(node_addr, 0, sizeof(void *) * cp_nv->elements * cp_nv->partitions);
    memset(node_size, 0, sizeof(size_t) * cp_nv->elements * cp_nv->partitions);

    cp->node_addr = node_addr;
    cp->node_size = node_size;
    cp->index = (char *)cp_pointer;
    cp_pointer += cp_nv->partitions;
    cp_pointer += 4096 - (size_t)cp_pointer % 4096;
    cp->start = cp_pointer;
    cp->elements = cp_nv->elements;

    cudaMallocHost((void **)&cp->part_byte_size, sizeof(long long) * cp_nv->partitions);
    cudaMallocHost((void **)&cp->part_elem_size, sizeof(long long) * cp_nv->partitions);
    cudaMallocHost((void **)&cp->part_bytes, sizeof(long long) * cp_nv->partitions);
    cudaMemset(cp->part_elem_size, 0, sizeof(long long) * cp_nv->partitions);
    cudaMemset(cp->part_bytes, 0, sizeof(long long) * cp_nv->partitions);

    setup_partitions<<<(cp_nv->partitions + 1023) / 1024, 1024>>>(cp->part_byte_size, cp_nv->partitions, cp_nv->size);
    return cp;
}

static __host__ void gpmcp_close(gpmcp *cp)
{
    printf("Total MMAP/UMAP time was %f ms\n", total_map_time);
    // gpm_unmap(cp->path, cp->cp, cp->tot_size);
    // cudaFreeHost(cp->node_addr);
    // cudaFreeHost(cp->node_size);
    // cudaFreeHost(cp->part_byte_size);
    // cudaFreeHost(cp->part_elem_size);
    // cudaFreeHost(cp->part_bytes);
    // cudaFreeHost(cp);
}

static __host__ cudaError_t gpmcp_unregister(gpmcp *cp)
{
    printf("UNMAP!\n");
    cudaError_t err = cudaHostUnregister(cp->mapped_addr);
    printf("Address %p, cudaHostUnregister of size %lu, Error is %d\n", cp->mapped_addr, cp->tot_size, err);
    return err;
}

static __host__ cudaError_t gpmcp_reregister(gpmcp *cp)
{
    printf("Reregister with size %lu\n", cp->tot_size);
    cudaError_t err = cudaHostRegister(cp->mapped_addr, cp->tot_size, cudaHostRegisterDefault);
    CHECK_CUDA_ERROR(err);

    char *cp_pointer = (char *)(cp->mapped_addr);
    gpmcp_nv *cp_nv = (gpmcp_nv *)cp_pointer;
    cp_pointer += sizeof(gpmcp_nv);
    cp->cp = cp_nv;

    printf("MMAP, addr is %p\n", cp_pointer);

    // align for disk msync
    cp_pointer += 4096 - (size_t)cp_pointer % 4096;
    cp->index = (char *)cp_pointer;
    cp_pointer += cp->partitions;

    cp->cp->size = cp->checkpoint_size; // TODO

    cp->start1 = cp_pointer;
    // align for disk msync
    cp->start1 = (char *)(cp->start1) + 4096 - ((size_t)(cp->start1) % 4096);
    cp->start2 = (char *)cp->start1 + cp->cp->size;
    cp->start2 = (char *)(cp->start2) + 4096 - ((size_t)(cp->start2) % 4096);

    printf("%p, %p\n", cp->start1, cp->start2);

    cp->cp->elements = cp->elements;
    cp->cp->partitions = cp->partitions;

    return err;
}

static __host__ long long gpmcp_register(gpmcp *cp, void *addr, size_t size, long long partition)
{
    long long val = 0;
    /*#if defined(__CUDA_ARCH__)
        long long start = cp->part_elem_size[partition];
        if(start >= cp->elements)
            return -1;
        // Device code here
        cp->node_addr[start + cp->elements * partition] = (long long *)addr;
        cp->node_size[start + cp->elements * partition] = size;
        cp->part_elem_size[partition]++;
        cp->part_bytes[partition] += size;

    #else*/

    long long start = cp->part_elem_size[partition];
    cp->node_addr[start + cp->elements * partition] = addr;
    cp->node_size[start + cp->elements * partition] = size;
    //printf("%d, SIZE OF %d IS %d\n", cp->elements, start + cp->elements * partition, cp->node_size[start + cp->elements * partition]);
    ++cp->part_elem_size[partition];
    cp->part_bytes[partition] += size;
    // #endif
    return val;
}

static __global__ void checkpointKernel(void *start, void *addr, size_t size)
{
    // Find where this thread should write
    size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * 8;

    for (; offset < size; offset += blockDim.x * gridDim.x * 8)
        gpm_memcpy_nodrain((char *)start + offset,
                           (char *)addr + offset,
                           min((size_t)8, size - offset), cudaMemcpyDeviceToHost);

    gpm_drain();
}
static __global__ void checkpointKernel_wdp(void *start, void *addr, size_t size)
{
    // Find where this thread should write
    size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * 8;

    for (; offset < size; offset += blockDim.x * gridDim.x * 8)
        gpm_memcpy_nodrain((char *)start + offset,
                           (char *)addr + offset,
                           min((size_t)8, size - offset), cudaMemcpyDeviceToHost);
}

static __global__ void printKernel(float *ar)
{
    printf("Num is %f\n", ar[0]);
}

static __host__ void gpmcp_print(float *ar)
{
    printKernel<<<1, 1>>>(ar);
}

static __host__ long long gpmcp_checkpoint(gpmcp *cp, long long partition)
{

    /*#if defined(__CUDA_ARCH__)
        // Device code
        size_t start = cp->part_byte_size[partition];

        PMEM_READ_OP( char ind = cp->index[partition] , sizeof(char) )
        ind = (ind != 0 ? 0 : 1);
        PMEM_READ_OP( size_t cp_size = cp->cp->size , sizeof(size_t) )
        start += ind * cp_size;

        long long elem_size = cp->part_elem_size[partition];
        PMEM_READ_OP( long long elems = cp->cp->elements , sizeof(long long) )
        for(long long i = 0; i < elem_size; ++i)
        {
            if(start >= 2 * cp_size)
                return -1;
            void *addr = (long long *)cp->node_addr[partition * elems + i];
            size_t size = cp->node_size[partition * elems + i];
            gpm_memcpy_nodrain((char *)cp->start + start, addr, size, cudaMemcpyDeviceToDevice);
            start += cp->node_size[partition * elems + i];
        }
        gpm_drain();
        // Update index once complete
        PMEM_READ_WRITE_OP( cp->index[partition] ^= 1; , sizeof(char) )
        gpm_drain();
        return 0;
    #else*/
    auto start_checkp_time = std::chrono::high_resolution_clock::now();
#ifdef GPM_WDP
    char *ptr = getenv("PMEM_THREADS");
    size_t pm_threads;
    if (ptr != NULL)
        pm_threads = atoi(ptr);
    else
        pm_threads = 1;
#endif
    size_t element_offset = 0;
    void *start_addr;
    char prev_ind;
    size_t all_part_size = 0;
    for (long long i = 0; i < cp->part_elem_size[partition]; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        void *addr = cp->node_addr[i + cp->elements * partition];
        size_t size = cp->node_size[i + cp->elements * partition];
        all_part_size += size;

        size_t start = cp->part_byte_size[partition] + element_offset;

        //printf("i is %d, index is %d, addr is %p, size is %lu, offset is %d\n", i, i + cp->elements * partition, addr, size, size % 128);

        // Based on index move to working copy
        prev_ind = (cp->index[partition] == 0 ? 0 : 1);
        char ind = (cp->index[partition] != 0 ? 0 : 1);
        if (ind == 0)
        {
            start_addr = cp->start1;
        }
        else
        {
            start_addr = cp->start2;
        }
        //printf("start_addr is %p\n", start_addr);

        // Host code
        const long long threads = 1024;
        long long blocks = 1;
        // Have each threadblock persist a single element
        // Threads within a threadblock persist at 4-byte offsets
#ifdef NVM_ALLOC_CPU
        auto t1 = std::chrono::high_resolution_clock::now();
        checkpointKernel<<<blocks, threads>>>((void *)((char *)start_addr + start), addr, size);
#endif
#ifdef NVM_ALLOC_GPU
        checkpointKernel<<<120, threads>>>((void *)((char *)cp->start + start), addr, size);
#endif
        assert(cudaDeviceSynchronize()==cudaSuccess);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        //printf("%d/%d, COPY TOOK %f ms\n", i, cp->part_elem_size[partition], ms_double);

#ifdef CHECKPOINT_TIME
        checkpoint_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
#endif
#ifdef GPM_WDP
        start_time = std::chrono::high_resolution_clock::now();
        size_t GRAN = (size + pm_threads - 1) / pm_threads;
        printf("Call persist 1\n");
#pragma omp parallel for num_threads(pm_threads)
        for (size_t ind = 0; ind < pm_threads; ++ind)
            pmem_persist((void *)((char *)cp->start + start + ind * GRAN), min(GRAN, size - ind * GRAN));
#ifdef PERSIST_TIME
        persist_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
#endif
#endif
        element_offset += size + 128 - (size % 128);
    }

#ifdef GPM_WDP
    // pmem_drain();
#endif
    auto start_time = std::chrono::high_resolution_clock::now();
    // Update index
    cp->index[partition] ^= 1;
    // pmem_persist(&cp->index[partition], sizeof(cp->index[partition]));
    char *addr = &cp->index[partition];
    int msync_err = msync(addr, sizeof(cp->index[partition]), MS_SYNC);
    printf("Persist error at address %p, partition %d, is %s\n", addr, partition, strerror(errno));
    assert(msync_err == 0);

    auto t1 = std::chrono::high_resolution_clock::now();
    //flush cache lines
    SFENCE();
    for (size_t j = 0; j < cp->cp->size; j += CACHE_LINE_SIZE)
    {
        FLUSH((char *)(start_addr) + j);
    }
    SFENCE();
    msync_err = msync(start_addr, cp->cp->size, MS_SYNC);
    SFENCE();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    printf("---------- MSYNC 2 TOOK %f ms\n", ms_double.count());

    printf("Persist error at address %p, size %lu, is %s\n", start_addr, cp->cp->size, strerror(errno));
    assert(msync_err == 0);

    auto start_map_time = std::chrono::high_resolution_clock::now();
    start_addr = (prev_ind == 0 ? cp->start1 : cp->start2);
    printf("prev_ind is %d, all_part_size is %lu\n", prev_ind, cp->cp->size);

    // not taken into account, just used to ensure msync is correct
    memset((void *)((char *)start_addr), 0, cp->cp->size);

    float time_one_checkp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_checkp_time).count();
    float time_map = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_map_time).count();
    float time_one_checkp_no_map = time_one_checkp - time_map;

    total_map_time += time_map;

    printf("Total time: %f ms, Map time: %f ms, Checkpoint time: %f ms\n", time_one_checkp, time_map, time_one_checkp_no_map);

#ifdef CHECKPOINT_TIME
    checkpoint_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
#endif
    return 0;
    // #endif
    // #endif
}

static __device__ long long gpmcp_checkpoint_start(gpmcp *cp, long long partition, long long element, size_t offset, size_t size)
{
    // Device code
    size_t start = cp->part_byte_size[partition];

    PMEM_READ_OP(char ind = cp->index[partition], sizeof(char))
    ind = (ind != 0 ? 0 : 1);
    PMEM_READ_OP(size_t cp_size = cp->cp->size, sizeof(size_t))
    start += ind * cp_size;

    PMEM_READ_OP(long long elems = cp->cp->elements, sizeof(long long))
    for (long long i = 0; i < element; ++i)
        start += cp->node_size[partition * elems + i];

    if (start >= 2 * cp_size)
        return -1;

    void *addr = (char *)cp->node_addr[partition * elems + element] + offset;
    gpm_memcpy((char *)cp->start + start + offset, addr, size, cudaMemcpyDeviceToDevice);
    return 0;
}

static __device__ long long gpmcp_checkpoint_value(gpmcp *cp, long long partition, long long element, size_t offset, size_t size, void *addr)
{
    // Device code
    size_t start = cp->part_byte_size[partition];

    PMEM_READ_OP(char ind = cp->index[partition], sizeof(char))
    ind = (ind != 0 ? 0 : 1);
    PMEM_READ_OP(size_t cp_size = cp->cp->size, sizeof(size_t))
    start += ind * cp_size;

    PMEM_READ_OP(long long elems = cp->cp->elements, sizeof(long long))
    for (long long i = 0; i < element; ++i)
        start += cp->node_size[partition * elems + i];

    if (start >= 2 * cp_size)
        return -1;

    gpm_memcpy((char *)cp->start + start + offset, addr, size, cudaMemcpyDeviceToDevice);
    return 0;
}

static __device__ long long gpmcp_checkpoint_finish(gpmcp *cp, long long partition)
{
    // Update index once complete
    PMEM_READ_WRITE_OP(cp->index[partition] ^= 1;, sizeof(char))
    gpm_drain();
    return 0;
}

__global__ void restoreKernel(void *start, void *addr, size_t size)
{
    // Find where this thread should write
    size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * 8;

    for (; offset < size; offset += blockDim.x * gridDim.x * 8)
        gpm_memcpy_nodrain((char *)start + offset,
                           (char *)addr + offset,
                           min((size_t)8, size - offset), cudaMemcpyDeviceToHost);
}

static __host__ long long gpmcp_restore(gpmcp *cp, long long partition)
{
    size_t element_offset = 0;
    for (long long i = 0; i < cp->part_elem_size[partition]; ++i)
    {
        void *addr = cp->node_addr[i + cp->elements * partition];
        size_t size = cp->node_size[i + cp->elements * partition];

        size_t start = cp->part_byte_size[partition] + element_offset;

        // Based on index move to working copy
        char ind = (cp->index[partition] != 0 ? 1 : 0);
        start += ind * cp->cp->size;

        // Host code
        const long long threads = 1024;
        long long blocks = 1;
        // Have each threadblock persist a single element
        // Threads within a threadblock persist at 4-byte offsets
        restoreKernel<<<blocks, threads>>>(addr, (void *)((char *)cp->start + start), size);
        element_offset += size + 128 - (size % 128);
    }
    cudaDeviceSynchronize();
    return 0;
}