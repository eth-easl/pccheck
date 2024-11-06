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
// #include "FAAQueue.h"

using namespace std;
#define CACHELINES_IN_1G 16777216
#define BYTES_IN_1G 1073741824
#define CACHELINE_SIZE 64
#define FLOAT_IN_CACHE 16
#define REGION_SIZE 53687091200ULL
#define PR_FILE_NAME "/mnt/pmem0/file_1_1"
#define PR_DATA_FILE_NAME "/mnt/pmem0/file_1_2"
#define MAX_ITERATIONS 8

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

static int curr_running = 0;

static int *PR_ADDR;
static int PADDING[64];
static int *PR_ADDR_DATA;

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
// static FAAArrayQueue<int> free_space;

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

static void mapPersistentRegion(const char *filename, int *regionAddr, const uint64_t regionSize, bool data)
{

    size_t mapped_len;
    int is_pmem;
    if (data)
    {
        if ((PR_ADDR_DATA = (int *)pmem_map_file(filename, regionSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem)) == NULL)
        {
            perror("pmem_map_file");
            exit(1);
        }
    }
    else
    {
        if ((PR_ADDR = (int *)pmem_map_file(filename, regionSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem)) == NULL)
        {
            perror("pmem_map_file");
            exit(1);
        }
    }
    assert(is_pmem > 0);
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

static void initialize(const char *filename, int max_async)
{

    struct stat buffer;
    bool newfile = (stat(filename, &buffer) == -1);
    mapPersistentRegion(filename, PR_ADDR_DATA, REGION_SIZE, true);
    mapPersistentRegion(filename, PR_ADDR, REGION_SIZE, false);

    curr_parall_iter.store(0);
    counter.store(1);

    // write init checkpoint on NVMM - locted right next to the checkpoint *
    void *next_addr = PR_ADDR + CACHELINE_SIZE; // sizeof(checkpoint*) == CACHELINE_SIZE
    struct checkpoint check = {0, 0};
    pmem_memcpy_persist(next_addr, &check, sizeof(struct checkpoint));
    pmem_memcpy_persist(PR_ADDR, &next_addr, sizeof(struct checkpoint *));

    // insert the current free data slots in the file
    for (int i = 0; i <= max_async; i++)
    {
        // printf("--------------- init enqueue: %d\n", i);
        free_space.enq(i);
        // free_space.enqueue(i, i);
    }
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

        set_cpu(id);

        // printf("%d, %p, %p\n", id, arr, add);

        for (size_t i = 0; i < sz;)
        {

            pmem_memcpy_nodrain((void *)add, (void *)arr, SIZE);
            arr += SIZE / sizeof(float);
            add += SIZE / sizeof(float);
            i += SIZE / sizeof(float);
        }
        pmem_drain();
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
            // parall_iter = free_space.dequeue(parall_iter);
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

        thread *threads[num_threads];
        thread_data allThreadsData[num_threads];

        for (int i = 0; i < num_threads; i++)
        {
            size_t size_for_thread_i = (reminder > i) ? (size_for_thread + 1) : size_for_thread;
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
        }

        for (int j = 0; j < num_threads; j++)
        {
            threads[j]->join();
        }

        struct checkpoint curr_check = {parall_iter, curr_counter};
        memcpy(curr_checkpoint, &curr_check, sizeof(struct checkpoint));
        FLUSH(curr_checkpoint);

        while (true)
        {
            bool res = __sync_bool_compare_and_swap(PR_ADDR, last_check, curr_checkpoint);
            struct checkpoint *check = *(struct checkpoint **)PR_ADDR;
            if (res)
            {
                BARRIER(PR_ADDR);
                int free = (((int *)last_check - PR_ADDR) / CACHELINE_SIZE) - 2;
                if (free == -1)
                    return;
                // cout << free << " " << curr_counter << endl;
                free_space.enq(free);
                // free_space.enqueue(free, free);
            }
            else if (check->counter < curr_counter)
            {
                last_check = check;
                continue;
            }
            else
            {
                free_space.enq(parall_iter);
                // free_space.enqueue(parall_iter, parall_iter);
            }
            return;
        }
    }

    static int registerCheck()
    {
        int parall_iter = 0;
        // get a new counter for the current checkpoint attempt
        long curr_counter = atomic_fetch_add(&counter, (long)1);
        // find free space to update the new checkpoint

        // TODO: fix MAX_ASYNC
        while (true)
        {

            // cout << "looking for free space: ";
            parall_iter = free_space.deq();
            // parall_iter = free_space.dequeue(parall_iter);
            // printf("--------------- dequeue register: %d\n", parall_iter);
            // cout << parall_iter << endl;
            if (parall_iter == INT_MIN)
                continue;
            else
                break;
        }

        // get the metadata address of the new slot
        struct checkpoint *curr_checkpoint = (struct checkpoint *)(PR_ADDR + CACHELINE_SIZE * (parall_iter + 2));
        struct checkpoint curr_check = {parall_iter, curr_counter};
        // cout << "before memcpy--------------------" << endl;
        memcpy(curr_checkpoint, &curr_check, sizeof(struct checkpoint));
        // cout << "after memcpy---------------------" << endl;
        // printf("from register, parallel_iter is %d, curr_counter is %ld, addr is %p\n", parall_iter, curr_counter, curr_checkpoint);
        return parall_iter;
    }

    static void savenvmNew(float *arr, size_t total_size, int num_threads, int parall_iter, int batch_num, size_t batch_size, bool last_batch)
    {
        // check the last updated checkpoint. Tries to change this value only in the last batch

        struct checkpoint *checkp_info_new = (struct checkpoint *)(PR_ADDR + CACHELINE_SIZE * (parall_iter + 2));
        // struct checkpoint* checkp_info_new = (struct checkpoint*)checkp_info;

        // int parallel_iteration = checkp_info_new->area;
        int counter_num = checkp_info_new->counter;

        struct checkpoint *volatile last_check = *(struct checkpoint *volatile *)PR_ADDR;
        // cout << "--------------IN SAVE NVM-------------" << endl;
        long curr_counter = counter_num;
        // int parall_iter = parallel_iteration;
        if (last_check->counter > curr_counter)
        { // Room for optimization
            if (last_batch)
            {
                BARRIER(PR_ADDR);
                // printf("--------------- a enqueue: %d\n", parall_iter);
                free_space.enq(parall_iter);
                // free_space.enqueue(parall_iter, parall_iter);
            }
            return;
        }

        float *curr_arr = arr; // address of the current batch
        size_t size_for_thread = batch_size / num_threads;
        size_t reminder = batch_size % num_threads;

        // get the metadata address of the new slot - already filled in the register function
        struct checkpoint *curr_checkpoint = checkp_info_new;
        //(struct checkpoint*) (PR_ADDR + CACHELINE_SIZE * (parall_iter + 2));
        // get the data address of the new slot - batches start from 1
        float *curr_pr_arr = (float *)(PR_ADDR_DATA) + (parall_iter * total_size) + (batch_size * (batch_num - 1));

        thread *threads[num_threads];
        thread_data allThreadsData[num_threads];

        size_t num_floats_SIZE = SIZE / sizeof(float);
        size_t rem_floats_SIZE = size_for_thread % num_floats_SIZE;
        size_t curr_sz = 0;

        size_t total_batches = total_size / batch_size;
        int thread_offset = (num_threads * (batch_num - 1)) + (parall_iter * total_batches * num_threads);

        // cout << "before savenvm_thread_nd" << endl;
        for (int i = 0; i < num_threads; i++)
        {
            // cout << "initiate threads: " << parall_iter << " " << i << endl;
            size_t size_for_thread_i = size_for_thread;
            // all should be multiple of SIZE
            size_for_thread_i += num_floats_SIZE - rem_floats_SIZE;
            size_for_thread_i = std::min(size_for_thread_i, batch_size - curr_sz);

            thread_data &data = allThreadsData[i];
            // printf("Thread %d will save %lu\n", i, size_for_thread);

            // take into a consideration all the running threads in the system
            // data.id =(i + 1 + thread_offset) % 46; //TODO: fix this - need to actually verify how many threads are currently running
            data.id = parall_iter * num_threads + i + 1;
            // printf("%d\n", data.id);
            //  the address to copy from
            data.arr = curr_arr;
            // the address to copy to
            // printf("curr_pr_arr is %p\n", curr_pr_arr);
            data.pr_arr = curr_pr_arr + 4096;
            data.size = size_for_thread_i;
            threads[i] = new thread(&savenvm_thread_nd, &data);
            curr_arr += size_for_thread_i;
            curr_pr_arr += size_for_thread_i;
            curr_sz += size_for_thread_i;
        }

        for (int j = 0; j < num_threads; j++)
        {
            // cout << "finalize threads: " << parall_iter << " " << j << endl;
            threads[j]->join();
        }

        // cout << "--------------FINISH SAVE NVM-------------" << endl;

        if (last_batch)
        {
            // cout << "in last batch" << endl;
            FLUSH(curr_checkpoint);
            while (true)
            {
                // std::cout << "PR_ADDR before!!!!!!!!!!!!!!!!!!!" << *(struct checkpoint**)PR_ADDR << std::endl;
                // printf("%p, %p, %p\n", *(struct checkpoint**)PR_ADDR, last_check, curr_checkpoint);
                bool res = __sync_bool_compare_and_swap(PR_ADDR, last_check, curr_checkpoint);
                // SFENCE();
                // std::cout << "PR_ADDR after!!!!!!!!!!!!!!!!!!!" << *(struct checkpoint**)PR_ADDR << std::endl;
                // cout << "The checkpoint happened:: " << res << endl;
                struct checkpoint *volatile check = *(struct checkpoint *volatile *)PR_ADDR;
                // cout << "The checkpoint for next time: " << check << endl;
                if (res)
                {
                    BARRIER(PR_ADDR);
                    int free = (((int *)last_check - PR_ADDR) / CACHELINE_SIZE) - 2;
                    if (free == -1)
                        return;
                    // cout << "The checkpoint address: " << last_check << endl;
                    // cout << "The PR_ADDR address: " << (struct checkpoint**)PR_ADDR << endl;
                    // printf("--------------- b enqueue: %d\n", free);
                    free_space.enq(free);
                    // free_space.enqueue(free, free);
                }
                else if (check->counter < curr_counter)
                {
                    last_check = check;
                    continue;
                }
                else
                {
                    BARRIER(PR_ADDR);
                    // printf("--------------- c enqueue: %d\n", parall_iter);
                    free_space.enq(parall_iter);
                    // free_space.enqueue(parall_iter, parall_iter);
                }
                // cout << "finished last batch" << endl;
                return;
            }
        }
        return;
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

    NVM_write *writer(const char *filename, int max_async)
    {
        NVM_write *nvmobj = new NVM_write();
        // printf("%s\n", filename);
        initialize(filename, max_async);
        return nvmobj;
    }

    void savenvm(NVM_write *t, float *ar, int sz, int num_threads)
    {
        t->savenvm(ar, sz, num_threads);
    }

    float *readfromnvm(NVM_write *t, float *ar, int sz)
    {
        return t->readfromnvm(ar, sz);
    }

    int registerCheck(NVM_write *t)
    {
        // printf("nvm object address is %p\n", t);
        return t->registerCheck();
        // printf("ret is %p\n", ret);
        // return ret;
    }

    // registerCheck() {}
    void savenvm_new(NVM_write *t, float *arr, size_t total_size, int num_threads, int parall_iter, int batch_num, size_t batch_size, bool last_batch)
    {

        // printf("received: %p\n", checkp_info);
        t->savenvmNew(arr, total_size, num_threads, parall_iter, batch_num, batch_size, last_batch);
    }
}

int main(int argc, char **argv)
{
    return 0;
}
