import torch


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
                    p.grad = p.data.new(p.size())
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
