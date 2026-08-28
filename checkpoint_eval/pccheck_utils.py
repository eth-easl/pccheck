import torch


# =====================================================================
# Original layout: parameters, then gradients.
# =====================================================================


def set_storage(model, optimizer_list, gpu_ar):
    start_idx = 0
    for name, ref in model.named_parameters():
        end_idx = start_idx + ref.numel()
        my_ar = gpu_ar[start_idx:end_idx]
        prev_shape = ref.size()
        with torch.no_grad():
            temp = ref.clone()
            ref.set_(my_ar, 0, tuple(prev_shape))
            ref.copy_(temp)
            # print(prev_shape, ref.shape, ref.data_ptr(), type(ref))
        start_idx += ref.numel()

    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    end_idx = start_idx + p.grad.numel()
                    my_ar = gpu_ar[start_idx:end_idx]
                    prev_shape = p.grad.size()
                    p.grad.set_(my_ar, 0, tuple(prev_shape))
                    start_idx += p.grad.numel()


def initialize(model, optimizer_list, do_opt_step=True):
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.state_dict()

    # initialize optimizer for realistic setups
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        if len(opt_state['state']) == 0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.grad = p.data.new_zeros(p.size()) # updated new to new_zeros
        if do_opt_step:
            optimizer.step()

    model_size = 0
    for name, ref in model_state.items():
        if (torch.is_tensor(ref)):
            model_size += ref.numel()
        elif (type(ref) == int or type(ref) == float):
            model_size += 1

    opt_size = 0
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        for name, _ in opt_state['state'].items():
            for k, ref in opt_state['state'][name].items():
                # print(k, ref.dtype)
                if (torch.is_tensor(ref)):
                    opt_size += ref.numel()
                elif (type(ref) == int or type(ref) == float):
                    opt_size += 1
    total_size = model_size + opt_size
    gpu_ar = torch.zeros(total_size).cuda()

    return gpu_ar, total_size


def get_total_size(model, optimizer_list):
    model_state = model.state_dict()
    model_size = 0
    for name, ref in model_state.items():
        if (torch.is_tensor(ref)):
            model_size += ref.numel()
        elif (type(ref) == int or type(ref) == float):
            model_size += 1

    opt_size = 0
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        for name, _ in opt_state['state'].items():
            for k, ref in opt_state['state'][name].items():
                # print(k, ref.dtype)
                if (torch.is_tensor(ref)):
                    opt_size += ref.numel()
                elif (type(ref) == int or type(ref) == float):
                    opt_size += 1

    return model_size + opt_size


# =====================================================================
# Updated layout: parameters, optimizer state, model buffers.
# =====================================================================


def _optimizer_state_tensors(optimizer_list):
    """Optimizer state tensors, in a stable order.

    Keyed by position (group/param index) rather than by parameter identity,
    so the order is reproducible across processes -- a resumed run must walk
    the buffer in exactly the order the saving run wrote it.
    """
    for oi, optimizer in enumerate(optimizer_list):
        for gi, group in enumerate(optimizer.param_groups):
            for pi, p in enumerate(group["params"]):
                state = optimizer.state.get(p, {})
                for k in sorted(state):
                    v = state[k]
                    if torch.is_tensor(v):
                        yield f"opt{oi}.g{gi}.p{pi}.{k}", v


def iter_checkpoint_state(model, optimizer_list):
    """Canonical layout of the updated flat buffer.

    This includes everything needed to resume training exactly.

      1. model parameters
      2. optimizer state (SGD momentum_buffer, Adam exp_avg/exp_avg_sq, ...)
      3. model buffers (BatchNorm running_mean/running_var/num_batches_tracked)

    Gradients are deliberately absent.
    """
    for name, p in model.named_parameters():
        yield f"param.{name}", p
    for name, v in _optimizer_state_tensors(optimizer_list):
        yield name, v
    for name, b in model.named_buffers():
        yield f"buffer.{name}", b


def set_storage_updated(model, optimizer_list, gpu_ar):
    """Re-point every checkpointed tensor into the flat buffer.

    After this, writing gpu_ar to disk captures the whole training state, and
    copying a checkpoint back into gpu_ar restores it -- no scatter/gather on
    either side. Returns the number of floats used.

    Caveat: do not call optimizer.load_state_dict() afterwards. It replaces
    the state tensors with fresh allocations, silently breaking their aliasing
    to gpu_ar so the buffer's optimizer region goes stale.
    """
    modules = dict(model.named_modules())
    start_idx = 0

    for name, ref in iter_checkpoint_state(model, optimizer_list):
        numel = ref.numel()
        my_ar = gpu_ar[start_idx:start_idx + numel]

        with torch.no_grad():
            if ref.dtype == torch.float32:
                # alias in place, preserving the current value
                temp = ref.clone()
                ref.set_(my_ar, 0, tuple(ref.size()))
                ref.copy_(temp)
            elif name.startswith("buffer."):
                # A non-float32 tensor cannot alias a float32 storage, so swap
                # the module's registered buffer for a float32 view. Affects
                # BatchNorm.num_batches_tracked (int64); it is only ever
                # incremented and divided, so float32 behaves identically.
                view = my_ar.view(ref.size())
                view.copy_(ref.to(torch.float32))
                parent, _, attr = name[len("buffer."):].rpartition(".")
                setattr(modules[parent], attr, view)
            else:
                raise TypeError(
                    f"cannot map {name} of dtype {ref.dtype} into the float32 "
                    f"checkpoint buffer. Only module buffers support dtype "
                    f"conversion; optimizer state must be float32."
                )

        start_idx += numel

    return start_idx


def initialize_updated(model, optimizer_list, do_opt_step=True):
    """Materialise optimizer state and allocate the flat buffer.

    The optimizer step exists only to force lazily-created state (momentum
    buffers etc) into existence so it can be sized and mapped. Unlike
    initialize(), gradients are zeroed first, so that step does not apply
    uninitialised memory to the weights.
    """
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        if len(opt_state['state']) == 0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.grad = p.data.new_zeros(p.size())
        if do_opt_step:
            optimizer.step()

    total_size = get_total_size_updated(model, optimizer_list)
    gpu_ar = torch.zeros(total_size).cuda()

    return gpu_ar, total_size


def get_total_size_updated(model, optimizer_list):
    return sum(t.numel() for _, t in iter_checkpoint_state(model, optimizer_list))
