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
//#include <libpmem.h>
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

/* Set by initialize() when it reattaches to an existing checkpoint file.
 * Exposed to Python via the accessors at the bottom so the loader can tell
 * whether there was anything to recover, and from which slot. */
static bool recovered_existing = false;
static long recovered_counter = 0;
static int recovered_slot = INT_MIN;

/* Slot count this file was initialized with (max_async + 1 usable slots).
 * Needed to bounds-check writes against REGION_SIZE. */
static int max_async_slots = 0;

/* Counter of the checkpoint currently being written into each slot.
 *
 * Deliberately volatile process memory, NOT the mapped file: the counter must
 * not become visible on disk until the payload is durable, otherwise a crash
 * mid-write leaves a slot advertising a checkpoint whose data is torn, and
 * recovery (which picks the highest counter) would select it. The persistent
 * counter is written only after the data msync succeeds -- see registerCheck()
 * and the last_batch block of savenvmNew(). */
static vector<long> pending_counters;

static int *PR_ADDR;
static int *PEER_CHECK_ADDR;
static int PADDING[64];
static int *PR_ADDR_DATA;

static uint8_t Cores[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

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

/* The head: which checkpoint is currently published.
 *
 * Stored as a single 64-bit word at PR_ADDR, packing the checkpoint's counter
 * with the slot holding it:  (counter << 8) | (slot + 1).
 *
 * It used to be a raw `struct checkpoint *`, which was unsound in two ways.
 * The CAS compared *addresses*, and slot addresses are recycled -- so after a
 * slot was freed and re-acquired by a newer checkpoint, a stale publisher's
 * CAS could match the address and succeed, regressing the head to an older
 * checkpoint and enqueueing an already-freed slot a second time (two
 * checkpoints then write the same region). Classic ABA. And a virtual address
 * is meaningless after a restart, since mmap hands back a different base.
 *
 * Packing the counter in removes both: counters come from a monotonic
 * fetch_add, so an expected value can never recur, and the slot is recorded
 * directly instead of being recovered by pointer arithmetic.
 *
 * slot + 1 is stored so the init metadata (slot -1) encodes as 0, keeping the
 * low byte unsigned. 8 bits covers 254 slots; max_async is single digits. */
typedef uint64_t head_t;

static inline head_t head_pack(long counter, int slot)
{
    return ((head_t)counter << 8) | (head_t)(uint8_t)(slot + 1);
}

static inline long head_counter(head_t h) { return (long)(h >> 8); }
static inline int head_slot(head_t h) { return (int)(h & 0xFF) - 1; }

static inline head_t head_load()
{
    return __atomic_load_n((head_t *)PR_ADDR, __ATOMIC_ACQUIRE);
}

static inline void head_store(head_t v)
{
    __atomic_store_n((head_t *)PR_ADDR, v, __ATOMIC_RELEASE);
}

static inline bool head_cas(head_t expected, head_t desired)
{
    return __atomic_compare_exchange_n((head_t *)PR_ADDR, &expected, desired,
                                       false, __ATOMIC_ACQ_REL,
                                       __ATOMIC_ACQUIRE);
}

//===================================================================

/* Hazard-pointer thread ids for the lock-free queues.
 *
 * FAAArrayQueue's enqueue/dequeue take a `tid` that indexes straight into
 * HazardPointers::hp[tid][...] with no bounds check, and the whole point of
 * that array is that each *live* thread owns a distinct entry: two threads
 * sharing one overwrite each other's protection, and one thread's clear()
 * cancels the other's, so a queue node can be retired and deleted while still
 * in use.
 *
 * The previous callers passed neither a thread id nor a bounded value --
 * registerCheck passed the slot index (0 for every concurrent caller), and on
 * an empty queue it fed the INT_MIN sentinel back in, indexing hp[INT_MIN].
 *
 * Ids are claimed on a thread's first use and released when it exits, so they
 * are reused across the short-lived worker threads the pipeline spawns per
 * checkpoint without two live threads ever sharing one. */
#define HP_TID_MAX 128 /* must match HazardPointers::HP_MAX_THREADS */

static std::atomic<bool> hp_tid_used[HP_TID_MAX];

class HpTidSlot
{
public:
    HpTidSlot()
    {
        for (int i = 0; i < HP_TID_MAX; i++)
        {
            bool expected = false;
            if (hp_tid_used[i].compare_exchange_strong(expected, true))
            {
                id_ = i;
                return;
            }
        }
        fprintf(stderr, "[pccheck] out of hazard-pointer thread slots "
                        "(>%d live threads)\n", HP_TID_MAX);
        abort();
    }
    ~HpTidSlot()
    {
        if (id_ >= 0)
            hp_tid_used[id_].store(false);
    }
    int id() const { return id_; }

private:
    int id_ = -1;
};

static int hp_tid()
{
    thread_local HpTidSlot slot;
    return slot.id();
}

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

    /* No O_TRUNC: opening the file must not destroy an existing checkpoint,
     * otherwise recovery is impossible by construction. Set PCCHECK_FRESH=1
     * to get the old discard-everything behaviour (useful for benchmarking,
     * where a stale checkpoint from a previous run is just noise). */
    bool force_fresh = (getenv("PCCHECK_FRESH") != NULL &&
                        atoi(getenv("PCCHECK_FRESH")) != 0);

    fd = open(filename, O_CREAT | O_RDWR | (force_fresh ? O_TRUNC : 0), (mode_t)0666);
    if (fd < 0)
    {
        perror("open checkpoint file");
        exit(1);
    }

    /* Only grow. ftruncate() to a smaller size would silently drop data, and
     * on an existing full-size file this is a no-op anyway. */
    if (force_fresh || newfile || (uint64_t)buffer.st_size < REGION_SIZE)
    {
        if (ftruncate(fd, REGION_SIZE) != 0)
        {
            perror("ftruncate");
            exit(1);
        }
    }

    //mapPersistentRegion(filename, PR_ADDR_DATA, REGION_SIZE, true, fd);
    mapPersistentRegion(filename, PR_ADDR, REGION_SIZE, false, fd);
    PEER_CHECK_ADDR = PR_ADDR + OFFSET_SIZE;
    /* Metadata occupies slots 0..max_async+3: the head pointer, the peer
     * pointer, the init metadata, then one per usable slot (0..max_async).
     * The data region must start *after* all of them -- with (max_async+3)
     * the last slot's metadata landed exactly on the first bytes of the data
     * region, and only survived because every write skipped 4096 floats. */
    PR_ADDR_DATA = PR_ADDR + (max_async+4)*OFFSET_SIZE;

    max_async_slots = max_async;
    pending_counters.assign(max_async + 1, 0);

    printf("PR_ADDR_DATA is %p, PR_ADDR is %p, PEER_CHECK_ADDR is %p\n", PR_ADDR_DATA, PR_ADDR, PEER_CHECK_ADDR);

    curr_parall_iter.store(0);

    /* Find the newest checkpoint already in the file, if any.
     *
     * We cannot follow the pointer stored at PR_ADDR[0]: it is a raw virtual
     * address from whichever process wrote it, and mmap(NULL, ...) hands us a
     * different base every run. So scan the metadata slots instead and take
     * the highest counter. Slot -1 is the init metadata at 2*OFFSET_SIZE;
     * slots 0..max_async are the rotating ones at (i+3)*OFFSET_SIZE.
     *
     * counter == 0 everywhere means "nothing here": registerCheck() hands out
     * counters starting at 1, and an unwritten (sparse) region reads as zero.
     * That distinction is why no separate magic/header is needed. */
    recovered_existing = false;
    recovered_counter = 0;
    recovered_slot = INT_MIN;

    if (!newfile && !force_fresh)
    {
        for (int i = -1; i <= max_async; i++)
        {
            struct checkpoint *c =
                (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (i + 3));
            if (c->counter > recovered_counter)
            {
                recovered_counter = c->counter;
                recovered_slot = i;
            }
        }
        recovered_existing = (recovered_counter > 0);
    }

    if (recovered_existing)
    {
        printf("[pccheck] reattached to existing checkpoint: slot %d, counter %ld\n",
               recovered_slot, recovered_counter);

        /* Continue the counter sequence rather than restarting it, so new
         * checkpoints still compare as newer than the recovered one. */
        counter.store(recovered_counter + 1);

        /* Re-point the head at the recovered checkpoint. Unlike the old raw
         * pointer, this value is meaningful across restarts. */
        head_t h = head_pack(recovered_counter, recovered_slot);
        head_store(h);
        if (msync((void *)PR_ADDR, sizeof(head_t), MS_SYNC) == -1)
        {
            perror("msync - recover, PR_ADDR");
            exit(1);
        }
        memcpy(PEER_CHECK_ADDR, &h, sizeof(head_t));
        if (msync((void *)PEER_CHECK_ADDR, sizeof(head_t), MS_SYNC) == -1)
        {
            perror("msync - recover, PEER_CHECK_ADDR");
            exit(1);
        }

        /* Every slot is reusable except the one holding the checkpoint we just
         * recovered -- handing that one out would let the next checkpoint
         * overwrite the only copy we have before it is superseded. */
        for (int i = 0; i <= max_async; i++)
        {
            if (i == recovered_slot)
                continue;
            printf("--------------- recover enqueue: %d\n", i);
            free_space.enqueue(i, hp_tid());
        }
    }
    else
    {
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

        /* Head points at the init metadata: counter 0, slot -1. Nothing has
         * been checkpointed yet, so no data slot is live. */
        head_t h = head_pack(0, -1);
        head_store(h);
        res = msync((void *)PR_ADDR, sizeof(head_t), MS_SYNC);
        if (res == -1)
        {
            perror("msync - init, PR_ADDR");
            exit(1);
        }

        memcpy(PEER_CHECK_ADDR, &h, sizeof(head_t));
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
            free_space.enqueue(i, hp_tid());
        }
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
        // Use our own hazard-pointer id, not the caller-supplied tid: Python
        // passes get_native_id()%128, which two live threads can collide on.
        (void)tid;
        void *ret = (void *)(dram.get_add(hp_tid()));
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

        // The queue's argument is a hazard-pointer THREAD id, not the slot we
        // are looking for. Keeping them in one variable meant every concurrent
        // caller shared entry 0, and an empty queue fed INT_MIN back in.
        const int tid = hp_tid();
        while (true)
        {
            parall_iter = free_space.dequeue(tid);
            if (parall_iter == INT_MIN)
                continue;
            else
                break;
        }

        // get the metadata address of the new slot
        struct checkpoint *curr_checkpoint = (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (parall_iter + 3));

        /* Mark the slot invalid *before* touching its data, and make that
         * durable. We are about to overwrite whatever payload it holds; if we
         * left the old counter in place and crashed mid-write, recovery would
         * find a valid-looking counter attached to a half-overwritten
         * checkpoint. counter == 0 is the "nothing here" value recovery skips.
         *
         * The real counter is published only once the data is durable, at the
         * end of savenvmNew. Until then it lives in process memory only. */
        struct checkpoint invalid = {parall_iter, 0};
        memcpy(curr_checkpoint, &invalid, sizeof(struct checkpoint));
        if (msync(curr_checkpoint, sizeof(struct checkpoint), MS_SYNC) == -1)
        {
            perror("msync - registerCheck, invalidate slot");
            exit(1);
        }

        pending_counters[parall_iter] = curr_counter;

        printf("Parall_iter %d, Write new metadata at address %p (counter %ld pending)\n",
               parall_iter, curr_checkpoint, curr_counter);
        return parall_iter;
    }

    static void savenvmNew(size_t tid, float *arr, size_t total_size, int num_threads, int parall_iter, int batch_num, size_t batch_size, bool last_batch)
    {

        printf("------------------------- savenvmNew, is_distributed is %d\n", is_distributed);

        // check the last updated checkpoint. Tries to change this value only in the last batch
        struct checkpoint *checkp_info_new = (struct checkpoint *)(PR_ADDR + OFFSET_SIZE * (parall_iter + 3));

        /* Take the counter from process memory, not from the mapped file: the
         * on-disk counter for this slot is deliberately 0 until the payload is
         * durable. Reading it as `long` also avoids the old long->int
         * truncation, which sat on the "which checkpoint is newest" path. */
        head_t last_head = head_load();
        long curr_counter = pending_counters[parall_iter];

        printf("At savenvmNew, tid is %lu, parall_iter is %d, num_threads is %d, last counter is %ld, curr_counter is %ld\n", tid, parall_iter, num_threads, head_counter(last_head), curr_counter);
        if (head_counter(last_head) > curr_counter)
        { // Room for optimization
            if (last_batch)
            {
                BARRIER(PR_ADDR);
                printf("Return!\n");
                free_space.enqueue(parall_iter, hp_tid());
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

        /* Nothing else validates that the mapping is big enough: REGION_SIZE
         * is a hardcoded constant while the requirement grows with total_size
         * and max_async. Running past the end of the mapping gives SIGBUS or
         * silent corruption, so fail loudly and early instead. */
        {
            uint64_t meta_bytes = (uint64_t)(max_async_slots + 4) * OFFSET_SIZE * sizeof(int);
            uint64_t slot_stride = (uint64_t)total_size * sizeof(float) + offset;
            uint64_t need = meta_bytes + (uint64_t)(max_async_slots + 1) * slot_stride;
            if (need > REGION_SIZE)
            {
                printf("[ERROR] checkpoint does not fit: need %lu bytes "
                       "(%d slots x %lu floats), REGION_SIZE is %llu. "
                       "Reduce max_async or the model size.\n",
                       need, max_async_slots + 1, (unsigned long)total_size, REGION_SIZE);
                exit(1);
            }
        }

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
            // the address to copy to. No +4096 skip: that existed only to keep
            // writes off the last metadata slot, which used to overlap the
            // start of the data region. PR_ADDR_DATA now starts past all
            // metadata, so the payload begins exactly at start_pr_arr -- which
            // is also what the msync below covers.
            data.pr_arr = curr_pr_arr;
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
        dram.put_add(arr, hp_tid());
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
            /* The payload is durable now, so it is safe to publish the
             * counter. Ordering matters: until this point the slot's on-disk
             * counter is 0, so a crash anywhere above leaves the slot ignored
             * by recovery rather than advertising torn data. */
            struct checkpoint done_check = {parall_iter, curr_counter};
            memcpy(curr_checkpoint, &done_check, sizeof(struct checkpoint));

            int res = msync(curr_checkpoint, sizeof(struct checkpoint), MS_SYNC);
            if (res == -1)
            {
                printf("[ERROR] Proc %d msync during checkpoint sync, with addr %p\n", parall_iter, curr_checkpoint);
                exit(1);
            }
            const head_t desired = head_pack(curr_counter, parall_iter);
            while (true)
            {
                /* Compare on (counter, slot), not on an address. The counter
                 * comes from a monotonic fetch_add, so `last_head` can never
                 * recur once superseded -- the ABA that let a stale publisher
                 * overwrite a newer head, and double-free its slot, is gone by
                 * construction. */
                bool res = head_cas(last_head, desired);
                head_t now = head_load();
                if (res)
                {
                    printf("CAS was successful! new counter is %ld, is_distributed is %d\n", head_counter(now), is_distributed);
                    BARRIER(PR_ADDR);
                    if (is_distributed) {
                        if (my_rank==0) {
                            wait_to_receive(client_sockets, world_size-1);
                        }
                        else {
                            send_and_wait(&comm_socket, curr_counter);
                        }

                        /* Record the same packed value. The previous code
                         * copied 8 bytes *from* the metadata struct here while
                         * initialize() wrote a pointer, so the two writers
                         * disagreed about the field's meaning. */
                        memcpy(PEER_CHECK_ADDR, &desired, sizeof(head_t));
                        res = msync((void *)PEER_CHECK_ADDR, sizeof(head_t), MS_SYNC);
                        if (res == -1) {
                            perror("msync - init, PEER_CHECK_ADDR");
                            exit(1);
                        }
                    }

                    /* The slot we just superseded is now free. Read straight
                     * out of the old head instead of recovering it by pointer
                     * arithmetic; -1 is the init metadata, which is not a data
                     * slot and must not be enqueued. */
                    int free = head_slot(last_head);
                    if (free == -1)
                        return;
                    free_space.enqueue(free, hp_tid());
                }
                else if (head_counter(now) < curr_counter)
                {
                    last_head = now;
                    continue;
                }
                else
                {
                    BARRIER(PR_ADDR);
                    free_space.enqueue(parall_iter, hp_tid());
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

    /* ---------------------------------------------------------------- *
     * Loader.
     *
     * Deliberately standalone: no writer object, no initialize(), no global
     * state, and no 105GB mapping. It must be callable *before* a writer
     * exists, because constructing one allocates the DRAM staging area,
     * re-points the head and rebuilds the free-slot queue -- none of which a
     * reader should trigger. pread() rather than mmap() keeps it cheap and
     * avoids faulting in the whole region.
     *
     * These mirror chk_load.py; the two are cross-checked against each other.
     * ---------------------------------------------------------------- */

    /* Byte offset of a slot's metadata. Slot -1 is the init metadata.
     * PR_ADDR is an int*, so OFFSET_SIZE is in ints. */
    static uint64_t pccheck_meta_offset(int slot)
    {
        return (uint64_t)(slot + 3) * OFFSET_SIZE * sizeof(int);
    }

    /* Byte offset of a slot's payload. Must match savenvmNew's start_pr_arr
     * and PR_ADDR_DATA in initialize(). */
    static uint64_t pccheck_data_offset(int slot, size_t total_size, int max_async)
    {
        size_t pad = 4096 - (total_size * sizeof(float)) % 4096;
        uint64_t start_floats =
            (uint64_t)slot * total_size + ((uint64_t)slot * pad) / 4;
        return (uint64_t)(max_async + 4) * OFFSET_SIZE * sizeof(int)
               + start_floats * sizeof(float);
    }

    /* Slot holding the newest complete checkpoint, or INT_MIN if there is
     * none. Writes its counter to out_counter when non-NULL. Same rule as the
     * recovery path in initialize(): highest counter wins, 0 means empty. */
    int pccheck_find_latest(const char *filename, int max_async, long *out_counter)
    {
        if (out_counter)
            *out_counter = 0;

        int f = open(filename, O_RDONLY);
        if (f < 0)
            return INT_MIN;

        int best_slot = INT_MIN;
        long best_counter = 0;
        for (int i = -1; i <= max_async; i++)
        {
            struct checkpoint c;
            ssize_t r = pread(f, &c, sizeof(c), pccheck_meta_offset(i));
            if (r != (ssize_t)sizeof(c))
                break;
            if (c.counter > best_counter)
            {
                best_counter = c.counter;
                best_slot = i;
            }
        }
        close(f);

        if (best_counter <= 0)
            return INT_MIN;
        if (out_counter)
            *out_counter = best_counter;
        return best_slot;
    }

    /* Read the newest checkpoint into out (total_size floats, caller-owned).
     * Returns its counter (> 0) on success, 0 if the file holds no complete
     * checkpoint, -1 on I/O error. Writes the source slot to out_slot when
     * non-NULL. */
    long pccheck_load_latest(const char *filename, int max_async,
                             size_t total_size, float *out, int *out_slot)
    {
        long counter = 0;
        int slot = pccheck_find_latest(filename, max_async, &counter);
        if (slot == INT_MIN)
            return 0;

        int f = open(filename, O_RDONLY);
        if (f < 0)
        {
            perror("pccheck_load_latest: open");
            return -1;
        }

        uint64_t off = pccheck_data_offset(slot, total_size, max_async);
        uint64_t need = off + (uint64_t)total_size * sizeof(float);

        struct stat st;
        if (fstat(f, &st) == 0 && (uint64_t)st.st_size < need)
        {
            printf("[ERROR] pccheck_load_latest: file is %lu bytes, need %lu "
                   "for slot %d of %lu floats -- wrong total_size/max_async?\n",
                   (uint64_t)st.st_size, need, slot, (unsigned long)total_size);
            close(f);
            return -1;
        }

        /* Loop: a single pread is capped well below a large checkpoint. */
        size_t remaining = total_size * sizeof(float);
        char *dst = (char *)out;
        while (remaining > 0)
        {
            ssize_t r = pread(f, dst, remaining, off);
            if (r < 0)
            {
                perror("pccheck_load_latest: pread");
                close(f);
                return -1;
            }
            if (r == 0)
            {
                printf("[ERROR] pccheck_load_latest: short read, %lu bytes left\n",
                       (unsigned long)remaining);
                close(f);
                return -1;
            }
            dst += r;
            off += r;
            remaining -= r;
        }
        close(f);

        if (out_slot)
            *out_slot = slot;
        return counter;
    }

    /* Recovery introspection. Valid after writer()/initialize() has run.
     * pccheck_recovered() == 0 means the file held no checkpoint and a fresh
     * one was initialized; == 1 means we reattached, and the slot/counter
     * accessors identify which checkpoint is live. */
    int pccheck_recovered()
    {
        return recovered_existing ? 1 : 0;
    }

    long pccheck_latest_counter()
    {
        return recovered_counter;
    }

    int pccheck_latest_slot()
    {
        return recovered_slot;
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
