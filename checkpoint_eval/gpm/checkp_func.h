#ifndef CHECKP_FUNC
#define CHECKP_FUNC

void checkpoint_func();
void dummy();
void init_checkpoint(const char *filename, unsigned long ckp_size, void **ar_ptrs, size_t *sizes, int num_to_register);
void finish_checkpoint();
void register_many(void **ar_ptrs, size_t *sizes, int num_to_register);
#endif
