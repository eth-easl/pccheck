#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <atomic>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <pthread.h>
#include <sys/time.h>
#include <chrono>
#include <assert.h>
#include <stdio.h>
#include <linux/mman.h>
#include <sys/types.h>
#include <fcntl.h> /* Definition of AT_* constants */
#include <sys/stat.h>
#include <sys/mman.h>
#include <libpmem.h>
#include <sys/stat.h>
#include <thread>
#include <atomic>
#include "MSQueue.h"

using namespace std;
#define CACHELINES_IN_1G 16777216
#define BYTES_IN_1G 1073741824
#define CACHELINE_SIZE 64
#define FLOAT_IN_CACHE 16
#define REGION_SIZE 10146834022ULL
#define PR_FILE_NAME "/mnt/pmem0/file_1_1"
#define PR_DATA_FILE_NAME "/mnt/pmem0/file_1_2"
#define MAX_ITERATIONS 4

// In NMV: checkpoint * ; checkpoint init ; checkpoint area 1 ; ... ; checkpint area MAX_ITERATIONS;
// Then all the tensors data begin. TODO: check if better to put checkpoint near data
// First, there is a checkpoint pointer. It points to the metadata of the current active chekpoint.
// Then, there is the metadata of the initial checkpoint. Used only once for initialization.
// Afterwards, there are MAX_ITERATIONS checkpoint metadatas that are updated one at a time
// when a new checkpoint is registered.
// After all the metadata, the real tensors data begin. The metadata points to its relevant data.
//=================== || ==================== || ========== ... ========== || =======================
//                    ||                      ||                           ||
//    Checkpoint *    || Checkpoint Metadata  ||    Checkpoint Metadata    ||   Checkpoint Metadata
//                    ||        init          || 0, ... MAX_ITERATIONS - 2 ||    MAX_ITERATIONS - 1
//                    ||                      ||                           ||
//=================== || ==================== || ========== ... ========== || =======================
//=================== || ==================== || ========== ... ========== || =======================
//                    ||                      ||                           ||
//     Checkpoint 0   ||     Checkpoint 1     ||      Checkpoint 2, ...    ||        Checkpoint
//                    ||                      ||     MAX_ITERATIONS - 2    ||      MAX_ITERATIONS - 1
//                    ||                      ||                           ||
//=================== || ==================== || ========================= || =======================

static uint8_t *PR_ADDR;
static uint8_t PADDING[64];
static uint8_t *PR_ADDR_DATA;

static uint8_t Cores[] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95};

struct thread_data
{
    uint32_t id;
    float *arr;
    float *pr_arr;
    uint32_t size;
} __attribute__((aligned(64)));

struct checkpoint
{
    long area;
    long counter;
} __attribute__((aligned(64)));

// NUM_THREADS = # of parallel threads that works on a single checkpoint
// ASYNC_CHECK = # of maximal parallel checkpoints
// SIZE = writing size of a single write
// TEST_TYPE = use flushes (FLUSH_FENCE) or non-temporal stores (default)
// curr_parall_iter = # of current  parallel checkpoints
// counter = upgraded within each checkpoint. Tracks the newest one
static int NUM_THREADS = 16;
static int ASYNC_CHECK = 1;
static int SIZE = 512;
static string TEST_TYPE = "";
static atomic<int> curr_parall_iter __attribute__((aligned(64)));
static atomic<long> counter __attribute__((aligned(64)));
static MSQueue<int> free_space;

//===================================================================

/* Allocate one core per thread */
static inline void set_cpu(int cpu)
{
    assert(cpu > -1);
    int n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu < n_cpus)
    {
        int cpu_use = Cores[cpu];
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu_use, &mask);
        pthread_t thread = pthread_self();
        if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &mask) != 0)
        {
            fprintf(stderr, "Error setting thread affinity\n");
        }
    }
}

//===================================================================

static void printHelp()
{
    cout << "  -T     test type" << endl;
    cout << "  -N     thread num" << endl;
    cout << "  -C     asynchronous checkpoints number" << endl;
    cout << "  -S     writing size" << endl;
}

//===================================================================

static bool parseArgs(int argc, char **argv)
{
    int arg;
    while ((arg = getopt(argc, argv, "T:N:C:S:H")) != -1)
    {
        switch (arg)
        {
        case 'T':
            TEST_TYPE = string(optarg);
            break;
        case 'N':
            NUM_THREADS = atoi(optarg);
            break;
        case 'C':
            ASYNC_CHECK = atoi(optarg);
            break;
        case 'S':
            SIZE = atoi(optarg);
            break;
        case 'H':
            printHelp();
            return false;
        default:
            return false;
        }
    }
    return true;
}

//====================================================================

static void mapPersistentRegion(const char *filename, uint8_t *regionAddr, const uint64_t regionSize, bool data, int fd)
{

    size_t mapped_len;
    int is_pmem;

    /*if (data) {
        if ((PR_ADDR_DATA = (uint8_t*)pmem_map_file(filename, regionSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem)) == NULL) {
            perror("pmem_map_file");
            exit(1);
        }
    } else {
        if ((PR_ADDR = (uint8_t*)pmem_map_file(filename, regionSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem)) == NULL) {
            perror("pmem_map_file");
            exit(1);
        }
    }*/

    if (data)
    {
        if ((PR_ADDR_DATA = (uint8_t *)mmap(NULL, regionSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED)
        {
            perror("mmap_file");
            exit(1);
        }
    }
    else
    {
        if ((PR_ADDR = (uint8_t *)mmap(NULL, regionSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED)
        {
            perror("mmap_file");
            exit(1);
        }
    }
}

//====================================================================

static void FLUSH(void *p)
{
    asm volatile("clwb (%0)" ::"r"(p));
}

static void SFENCE()
{
    asm volatile("sfence" ::: "memory");
}

static void BARRIER(void *p)
{
    FLUSH(p);
    SFENCE();
}

//====================================================================

static void initialize(const char *filename)
{

    struct stat buffer;
    bool newfile = (stat(filename, &buffer) == -1);

    printf("Inside init!\n");
    int fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, (mode_t)0666);
    ftruncate(fd, REGION_SIZE);

    mapPersistentRegion(filename, PR_ADDR_DATA, REGION_SIZE, true, fd);
    mapPersistentRegion(filename, PR_ADDR, REGION_SIZE, false, fd);
    // initialize only when file does not exist

    printf("File mmaped!\n");
    if (newfile)
    {
        /*float* w = (float*)(PR_ADDR_DATA);
        std::cout << "file does not exist!!" << std::endl;
            for (int i = 0; i < CACHELINES_IN_1G * 4; i++) {
                w += CACHELINE_SIZE * i;
                *w = 100;
                FLUSH(w);
                SFENCE();
            }*/
    }
    else
        std::cout << "file exists!" << std::endl;

    curr_parall_iter.store(0);
    counter.store(1);

    // write init checkpoint on NVMM - locted right next to the checkpoint *
    void *next_addr = PR_ADDR + CACHELINE_SIZE; // sizeof(checkpoint*) == CACHELINE_SIZE
    struct checkpoint check = {0, 0};
    // pmem_memcpy_persist(next_addr, &check, sizeof(struct checkpoint));
    // pmem_memcpy_persist(PR_ADDR, &next_addr, sizeof(struct checkpoint*));

    printf("Before copying and syncing\n");
    memcpy(next_addr, &check, sizeof(struct checkpoint));

    int res = msync((void *)PR_ADDR, CACHELINE_SIZE + sizeof(struct checkpoint), MS_SYNC);
    if (res == -1)
    {
        perror("msync - init, next_addr ");
        exit(1);
    }

    memcpy(PR_ADDR, &next_addr, sizeof(struct checkpoint *));

    res = msync((void *)PR_ADDR, sizeof(struct checkpoint *), MS_SYNC);
    if (res == -1)
    {
        perror("msync - init, PR_ADDR");
        exit(1);
    }

    printf("After copying and syncing\n");

    // insert the current free data slots in the file
    for (int i = 0; i < ASYNC_CHECK; i++)
    {
        free_space.enq(i);
    }

    printf("exit init!\n");
    // struct checkpoint* curr = *(struct checkpoint**)PR_ADDR;
    // cout << "PR_ADDR: " << (void*)PR_ADDR << endl;
    // cout << "NEXT_ADDR: " << next_addr << endl;
    // cout << "*PR_ADDR: " << curr << endl;
    // cout << "area: " << curr->area << endl;
    // cout << "counter: " << curr->counter << endl;
}

//====================================================================

/* Provides ways to write data to a dedicated address within PR_ADDR_DATA.
 * savenvm_thread_flush and savenvm_thread_nd writes and persists the
 * data to a dedicated address. These methods are called by every parallel
 * thread that writes within a single checkpoint (out of NUM_THREADS).
 * savenvm synchronizes the entire checkpoint. Called every time there is
 * a new checkpoint to be written. */

class NVM_write
{
public:
    static void savenvm_thread_flush(thread_data *data)
    {
        int id = data->id;
        float *arr = data->arr;
        float *add = data->pr_arr;
        int sz = data->size;
        set_cpu(id);
        for (int i = 0; i < sz;)
        {
            float *add_to_flush = add;
            for (int j = 0; j < FLOAT_IN_CACHE; j++)
            {
                *add = arr[i];
                i++;
                if (i == sz)
                    break;
                add++;
            }
            FLUSH(add_to_flush);
        }
        SFENCE();
    }

    static void savenvm_thread_nd(thread_data *data)
    {

        int id = data->id;
        float *arr = data->arr;
        float *add = (float *)data->pr_arr;
        size_t sz = data->size;
        // set_cpu(id);
        for (size_t i = 0; i < sz;)
        {
            // pmem_memcpy_nodrain((void*)add, (void*)arr, SIZE);
            memcpy((void *)add, (void *)arr, SIZE);
            arr += SIZE / sizeof(float);
            add += SIZE / sizeof(float);
            i += SIZE / sizeof(float);
        }
        // pmem_drain();
        // int res = msync((void*)(data->pr_arr), sz*sizeof(float), MS_SYNC);
        // if (res == -1) {
        //	perror("msync - savenvm_thread_nd ");
        //	exit(1);
        // }
    }

    static void savenvm(float *arr, size_t sz, int num_threads)
    {

        int parall_iter = 0;
        // check the last updated checkpoint. savenvm tries to change this value
        struct checkpoint *volatile last_check = *(struct checkpoint **)PR_ADDR;
        // get a new counter for the current checkpoint attempt
        long curr_counter = atomic_fetch_add(&counter, (long)1);
        // find free space to update the new checkpoint
        while (true)
        {
            parall_iter = free_space.deq();
            if (parall_iter == INT_MIN)
                continue;
            else
                break;
        }

        size_t size_for_thread = sz / num_threads;
        size_t reminder = sz % num_threads;
        float *curr_arr = arr;

        // get the metadata address of the new slot - TODO: change with a queue
        struct checkpoint *curr_checkpoint = (struct checkpoint *)(PR_ADDR + CACHELINE_SIZE * (parall_iter + 2));
        // get the data address of the new slot
        float *curr_pr_arr = (float *)(PR_ADDR_DATA + (parall_iter * sz));
        float *start_pr_arr = curr_pr_arr;

        thread *threads[num_threads];
        thread_data allThreadsData[num_threads];
        size_t num_floats_SIZE = SIZE / sizeof(float);
        size_t rem_floats_SIZE = size_for_thread % num_floats_SIZE;
        size_t curr_sz = 0;

        for (int i = 0; i < num_threads; i++)
        {
            // size_t size_for_thread_i = (reminder > i) ? (size_for_thread + 1) : size_for_thread;
            size_t size_for_thread_i = size_for_thread;
            // all should be multiple of SIZE
            if (rem_floats_SIZE != 0)
            {
                size_for_thread_i += num_floats_SIZE - rem_floats_SIZE;
            }

            size_for_thread_i = std::min(size_for_thread_i, sz - curr_sz);

            printf("Thread %d will save %lu\n", i, size_for_thread);
            thread_data &data = allThreadsData[i];
            // take into a consideration all the running threads in the system
            data.id = i + 1 + (num_threads * parall_iter);
            // the address to copy from
            data.arr = curr_arr;
            // the address to copy to
            data.pr_arr = curr_pr_arr + 4096;
            data.size = size_for_thread_i;
            threads[i] = new thread(&savenvm_thread_nd, &data);
            curr_arr += size_for_thread_i;
            curr_pr_arr += size_for_thread_i;
            curr_sz += size_for_thread_i;
        }

        for (int j = 0; j < num_threads; j++)
            threads[j]->join();

        // do a total msync here
        int res = msync((void *)(start_pr_arr), sz * sizeof(float), MS_SYNC);
        if (res == -1)
        {
            perror("msync");
            exit(1);
        }

        struct checkpoint curr_check = {parall_iter, curr_counter};
        memcpy(curr_checkpoint, &curr_check, sizeof(struct checkpoint));
        // FLUSH(curr_checkpoint);
        msync(curr_checkpoint, sizeof(struct checkpoint), MS_SYNC);

        while (true)
        {
            bool res = __sync_bool_compare_and_swap(PR_ADDR, last_check, curr_checkpoint);
            struct checkpoint *check = *(struct checkpoint **)PR_ADDR;
            if (res)
            {
                BARRIER(PR_ADDR);
                int free = (((uint8_t *)last_check - PR_ADDR) / CACHELINE_SIZE) - 2;
                if (free == -1)
                    continue;
                // cout << free << " " << curr_counter << endl;
                free_space.enq(free);
            }
            else if (check->counter < curr_counter)
            {
                last_check = check;
                continue;
            }
            else
            {
                free_space.enq(parall_iter);
            }
            return;
        }
        // struct checkpoint* curr = *(struct checkpoint**)PR_ADDR;
        // cout << curr << endl;
        // cout << curr->area << endl;
        // cout << curr->counter << endl;
        // cout << counter.load() << endl;
    }

    float *readfromnvm(float *ar, int size)
    {
        float *w = (float *)PR_ADDR_DATA;
        for (int i = 0; i < size; i++)
        {
            ar[i] = *w;
            w++;
        }
        return nullptr;
    }
};

//====================================================================

extern "C"
{

    NVM_write *writer(const char *filename)
    {
        NVM_write *nvmobj = new NVM_write();
        initialize(filename);
        return nvmobj;
    }

    void savenvm(NVM_write *t, float *ar, size_t sz, int num_threads)
    {
        t->savenvm(ar, sz, num_threads);
    }

    float *readfromnvm(NVM_write *t, float *ar, size_t sz)
    {
        return t->readfromnvm(ar, sz);
    }
}

int main(int argc, char **argv)
{
    return 0;
}
