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
#include "FAAQueue.h"
#include "DRAMAlloc.h"
#include "socket_work.h"


using namespace std;
#define CACHELINES_IN_1G 16777216
#define BYTES_IN_1G 1073741824
#define CACHELINE_SIZE 64
#define OFFSET_SIZE 4096
#define FLOAT_IN_CACHE 16
#define REGION_SIZE 113406487152ULL
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
static int *PEER_CHECK_ADDR;
static int PADDING[64];
static int *PR_ADDR_DATA;

static uint8_t Cores[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

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
// static MSQueue<int> free_space;
static FAAArrayQueue<int> free_space;

int fd;
DRAMAlloc dram;

// for distributed
bool is_distributed;
int my_rank;
int world_size;
struct sockaddr_in socket_address;
int comm_fd;
int comm_socket;
vector<int> client_sockets;
int port = 1235;

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

static void mapPersistentRegion(const char *filename, int *regionAddr, const uint64_t regionSize, bool data, int fd)
{

    size_t mapped_len;
    int is_pmem;
    /*if (data) {
        if ((PR_ADDR_DATA = (int*)pmem_map_file(filename, regionSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem)) == NULL) {
            perror("pmem_map_file");
            exit(1);
        }
    } else {
        if ((PR_ADDR = (int*)pmem_map_file(filename, regionSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem)) == NULL) {
            perror("pmem_map_file");
            exit(1);
        }
    }
    assert (is_pmem > 0);*/
    if (data)
    {
        if ((PR_ADDR_DATA = (int *)mmap(NULL, regionSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED)
        {
            perror("mmap_file");
            exit(1);
        }
    }
    else
    {
        if ((PR_ADDR = (int *)mmap(NULL, regionSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED)
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

static void initialize(const char *filename, int max_async, size_t batch_size_floats, size_t num_batches, bool dist, int dist_rank, int wsize)
{

    struct stat buffer;
    bool newfile = (stat(filename, &buffer) == -1);

    fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, (mode_t)0666);
    ftruncate(fd, REGION_SIZE);

    //mapPersistentRegion(filename, PR_ADDR_DATA, REGION_SIZE, true, fd);
    mapPersistentRegion(filename, PR_ADDR, REGION_SIZE, false, fd);
    PEER_CHECK_ADDR = PR_ADDR + OFFSET_SIZE;
    PR_ADDR_DATA = PR_ADDR + (max_async+3)*OFFSET_SIZE;

    printf("PR_ADDR_DATA is %p, PR_ADDR is %p, PEER_CHECK_ADDR is %p\n", PR_ADDR_DATA, PR_ADDR, PEER_CHECK_ADDR);

    curr_parall_iter.store(0);
    counter.store(1);

    // write init checkpoint on NVMM - locted right next to the checkpoint *
    void *next_addr = PR_ADDR + 2*OFFSET_SIZE; // sizeof(checkpoint*) == CACHELINE_SIZE
    struct checkpoint check = {0, 0};
    // pmem_memcpy_persist(next_addr, &check, sizeof(struct checkpoint));
    // pmem_memcpy_persist(PR_ADDR, &next_addr, sizeof(struct checkpoint*));

    memcpy(next_addr, &check, sizeof(struct checkpoint));
    int res = msync((void *)PR_ADDR, 2*OFFSET_SIZE + sizeof(struct checkpoint), MS_SYNC);
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

    memcpy(PEER_CHECK_ADDR, &next_addr, sizeof(struct checkpoint *));
    res = msync((void *)PEER_CHECK_ADDR, sizeof(struct checkpoint *), MS_SYNC);
    if (res == -1)
    {
        perror("msync - init, PEER_CHECK_ADDR");
        exit(1);
    }

    // insert the current free data slots in the file
    for (int i = 0; i <= max_async; i++)
    {
        printf("--------------- init enqueue: %d\n", i);
        free_space.enqueue(i, 0);
    }

    printf("Call dram.alloc, num_batches is %lu, batch_size_floats is %lu\n", num_batches, batch_size_floats);
    dram.alloc(num_batches, batch_size_floats);
    dram.initialize(num_batches, batch_size_floats);

    is_distributed = dist;
    my_rank = dist_rank;
    world_size = wsize;

    printf("------------------------- is_distributed is %d, rank is %d\n", is_distributed, my_rank);
    if (is_distributed) {

        char const* tmp = getenv("PCCHECK_COORDINATOR");
        std::string server_ip(tmp);

        printf("My rank is %d\n", my_rank);
        if (my_rank==0) {
            setup_rank0_socket(port, &comm_fd, &socket_address, world_size-1, client_sockets);
        }
        else {
            setup_other_socket(&comm_socket, &socket_address, server_ip, port);
        }
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
    static void *take_cpu_address(size_t tid)
    {
        // TODO: what should the id be?
        void *ret = (void *)(dram.get_add(tid));
        return ret;
    }

    static void savenvm_thread_flush(thread_data *data)
    {
        int id = data->id;
        float *arr = data->arr;
        float *add = data->pr_arr;
        int sz = data->size;
        // set_cpu(id);
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
        printf("At savenvm_thread_nd id is %d, sz is %lu!\n", id, sz);
        for (size_t i = 0; i < sz;)
        {

            // pmem_memcpy_nodrain((void*)add, (void*)arr, SIZE);
            memcpy((void *)add, (void *)arr, SIZE);
            arr += SIZE / sizeof(float);
            add += SIZE / sizeof(float);
            i += SIZE / sizeof(float);
        }
        // pmem_drain();
    }

    static int registerCheck()
    {
        int parall_iter = 0;
        // get a new counter for the current checkpoint attempt
        long curr_counter = atomic_fetch_add(&counter, (long)1);
        // find free space to update the new checkpoint

        while (true)
        {
            parall_iter = free_space.dequeue(parall_iter);
            if (parall_iter == INT_MIN)
                continue;
            else
                break;
        }

        // get the metadata address of the new slot
        struct checkpoint *curr_checkpoint = (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (parall_iter + 3));
        struct checkpoint curr_check = {parall_iter, curr_counter};
        printf("Parall_iter %d, Write new metadata at address %p\n", parall_iter, curr_checkpoint);
        memcpy(curr_checkpoint, &curr_check, sizeof(struct checkpoint));
        return parall_iter;
    }

    static void savenvmNew(size_t tid, float *arr, size_t total_size, int num_threads, int parall_iter, int batch_num, size_t batch_size, bool last_batch)
    {

        printf("------------------------- savenvmNew, is_distributed is %d\n", is_distributed);

        // check the last updated checkpoint. Tries to change this value only in the last batch
        struct checkpoint *checkp_info_new = (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (parall_iter + 3));

        // int parallel_iteration = checkp_info_new->area;
        int counter_num = checkp_info_new->counter;
        struct checkpoint *volatile last_check = *(struct checkpoint *volatile *)PR_ADDR;
        long curr_counter = counter_num;
        // int parall_iter = parallel_iteration;

        printf("At savenvmNew, tid is %d, parall_iter is %d, num_threads is %d, last counter is %d, curr_counter is %d\n", tid, parall_iter, num_threads, last_check->counter, curr_counter);
        if (last_check->counter > curr_counter)
        { // Room for optimization
            if (last_batch)
            {
                BARRIER(PR_ADDR);
                printf("Return!\n");
                free_space.enqueue(parall_iter, parall_iter);
            }
            return;
        }

        float *curr_arr = arr; // address of the current batch
        size_t size_for_thread = batch_size / num_threads;
        size_t reminder = batch_size % num_threads;

        // get the metadata address of the new slot - already filled in the register function
        struct checkpoint *curr_checkpoint = checkp_info_new;
        // get the data address of the new slot - batches start from 1

        // make sure the start address is aligned at 4KB
        size_t offset = 4096 - (total_size * sizeof(float)) % 4096;
        float *start_pr_arr = NULL;
        // printf("offset is %ld\n", offset);
        start_pr_arr = (float *)PR_ADDR_DATA + (parall_iter * total_size) + (parall_iter * offset) / 4;
        float *curr_pr_arr = start_pr_arr + (batch_size * (batch_num - 1));

        thread *threads[num_threads];
        thread_data allThreadsData[num_threads];

        size_t num_floats_SIZE = SIZE / sizeof(float);
        size_t rem_floats_SIZE = size_for_thread % num_floats_SIZE;
        size_t curr_sz = 0;

        size_t total_batches = total_size / batch_size;
        int thread_offset = (num_threads * (batch_num - 1)) + (parall_iter * total_batches * num_threads);

        printf("Save checkpoint - create threads!\n");

        for (int i = 0; i < num_threads; i++)
        {
            size_t size_for_thread_i = size_for_thread;
            // all should be multiple of SIZE
            size_for_thread_i += num_floats_SIZE - rem_floats_SIZE;
            size_for_thread_i = std::min(size_for_thread_i, batch_size - curr_sz);

            thread_data &data = allThreadsData[i];

            // take into a consideration all the running threads in the system
            // data.id =(i + 1 + thread_offset) % 46; //TODO: fix this - need to actually verify how many threads are currently running
            data.id = parall_iter * num_threads + i + 1;
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
        {
            threads[j]->join();
        }

        //printf("AFTER ALL JOINED, tid is %d, parall_iter is %d, num_threads is %d\n", tid, parall_iter, num_threads);

        // free mem copy
        // TODO: what should id be?
        dram.put_add(arr, tid);
        //printf("AFTER PUT_ADD, tid is %d, parall_iter is %d, num_threads is %d\n", tid, parall_iter, num_threads);

        if (last_batch)
        {
            // do a total msync here
            auto t1 = std::chrono::high_resolution_clock::now();
            int res = msync((void *)(start_pr_arr), total_size * sizeof(float), MS_SYNC);
            if (res == -1)
            {
                printf("[ERROR] Proc %d msync during model persisting addr %p, size %lu\n", parall_iter, start_pr_arr, total_size * sizeof(float));
                exit(1);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            printf("MSYNC TOOK %f ms\n", ms_double.count());
        }

        //printf("AFTER MSYNC, tid is %d, parall_iter is %d, num_threads is %d\n", tid, parall_iter, num_threads);

        // cout << "--------------FINISH SAVE NVM-------------" << endl;

        if (last_batch)
        {
            int res = msync(curr_checkpoint, sizeof(struct checkpoint), MS_SYNC);
            if (res == -1)
            {
                printf("[ERROR] Proc %d msync during checkpoint sync, with addr %p\n", parall_iter, curr_checkpoint);
                exit(1);
            }
            while (true)
            {
                bool res = __sync_bool_compare_and_swap(PR_ADDR, last_check, curr_checkpoint);
                struct checkpoint *volatile check = *(struct checkpoint *volatile *)PR_ADDR;
                if (res)
                {
                    printf("CAS was successful! new counter is %ld, is_distributed is %d\n", check->counter, is_distributed);
                    BARRIER(PR_ADDR);
                    if (is_distributed) {
                        if (my_rank==0) {
                            wait_to_receive(client_sockets, world_size-1);
                        }
                        else {
                            send_and_wait(&comm_socket, curr_counter);
                        }

                        memcpy(PEER_CHECK_ADDR, curr_checkpoint, sizeof(struct checkpoint *));
                        res = msync((void *)PEER_CHECK_ADDR, sizeof(struct checkpoint *), MS_SYNC);
                        if (res == -1) {
                            perror("msync - init, PEER_CHECK_ADDR");
                            exit(1);
                        }
                    }

                    int free = (((int *)last_check - PR_ADDR) / OFFSET_SIZE) - 3;
                    if (free == -1)
                        return;
                    free_space.enqueue(free, free);
                }
                else if (check->counter < curr_counter)
                {
                    last_check = check;
                    continue;
                }
                else
                {
                    BARRIER(PR_ADDR);
                    free_space.enqueue(parall_iter, parall_iter);
                }
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

    NVM_write *writer(const char *filename, int max_async, size_t batch_size_floats, size_t num_batches, bool dist, int dist_rank, int world_size)
    {
        NVM_write *nvmobj = new NVM_write();
        // printf("%s\n", filename);
        initialize(filename, max_async, batch_size_floats, num_batches, dist, dist_rank, world_size);
        return nvmobj;
    }

    float *readfromnvm(NVM_write *t, float *ar, int sz)
    {
        return t->readfromnvm(ar, sz);
    }

    int registerCheck(NVM_write *t)
    {
        return t->registerCheck();
    }

    void *take_cpu_address(NVM_write *t, size_t tid)
    {
        return t->take_cpu_address(tid);
    }

    // registerCheck() {}
    void savenvm_new(NVM_write *t, size_t tid, float *arr, size_t total_size, int num_threads, int parall_iter, int batch_num, size_t batch_size, bool last_batch)
    {

        t->savenvmNew(tid, arr, total_size, num_threads, parall_iter, batch_num, batch_size, last_batch);
    }
}

int main(int argc, char **argv)
{
    return 0;
}
