import re
import torch
from torch import nn


from deepspeed.pipe import PipelineModule, LayerSpec
from bloom_ds import get_bloom_causal_lm_specs
from opt_ds import get_opt_causal_lm_specs

class LMCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, is_opt, *args, **kwargs):
        self.is_opt = is_opt
        super().__init__(*args, **kwargs, ignore_index=-100)

    def forward(self, lm_logits, labels):
        if not self.is_opt:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            return super().forward(
                    shift_logits.view(batch_size * seq_length, vocab_size),
                    shift_labels.view(batch_size * seq_length)
            )
        else:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., :].contiguous()
            print(shift_logits.shape, shift_labels.shape)
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss


class LMLoss(nn.Module):

    def __init__(self, is_opt):
        super().__init__()
        self.crit_ce = LMCrossEntropyLoss(is_opt)

    def forward(self, lm_logits, labels):
        loss = self.crit_ce(lm_logits, labels)
        return loss


def convert_bloom(model, config, num_stages):
    state = model.state_dict()
    for k,v in state.items():
        print(k)

    res = {
        0: {
            'word_embeddings.weight': state['transformer.word_embeddings.weight'],
            'word_embeddings_layernorm.weight': state['transformer.word_embeddings_layernorm.weight'],
            'word_embeddings_layernorm.bias': state['transformer.word_embeddings_layernorm.bias'],
        },
    }

    ind_last = -1
    for k,v in state.items():
        if not re.search('^transformer.h.', k): continue
        k = re.sub('^transformer.h.', '', k)
        ind = int(re.search('^\d+', k).group())
        k = re.sub('^\d+\.', '', k)
        ind += 1
        if not ind in res: res[ind] = {}
        res[ind][k] = v
        ind_last = max(ind_last, ind)

    ind_last += 1
    last = {
        ind_last: {
            'word_embeddings.weight': state['transformer.word_embeddings.weight'],
            'word_embeddings_layernorm.weight': state['transformer.ln_f.weight'],
            'word_embeddings_layernorm.bias': state['transformer.ln_f.bias'],
        },
    }
    if 'lm_head.weight' in state.keys():
        last[ind_last]['word_embeddings.weight'] = state['lm_head.weight']
    res.update(last)

    # convert
    layers = get_bloom_causal_lm_specs(config, res)

    model = PipelineModule(layers, loss_fn=LMLoss(False), num_stages=num_stages)

    return model

def convert_opt(model, config, num_stages):
    print(model)
    state = model.state_dict()
    for k,v in state.items():
        print(k)

    res = {
        0: {
            'embed_tokens.weight': state['model.decoder.embed_tokens.weight'],
            'embed_positions.weight': state['model.decoder.embed_positions.weight'],
            #'project_out.weight': state['model.decoder.project_out.weight'],
            #'project_in.weight': state['model.decoder.project_in.weight'],
        },
    }

    ind_last = -1
    for k,v in state.items():
        if not re.search('^model.decoder.layers.', k): continue
        k = re.sub('^model.decoder.layers.', '', k)
        ind = int(re.search('^\d+', k).group())
        k = re.sub('^\d+\.', '', k)
        ind += 1
        if not ind in res: res[ind] = {}
        res[ind][k] = v
        ind_last = max(ind_last, ind)

    ind_last += 1
    last = {
        ind_last: {
            'embed_tokens.weight': state['lm_head.weight'],
        },
    }
    res.update(last)

    layers = get_opt_causal_lm_specs(config, res)
    model = PipelineModule(layers, loss_fn=LMLoss(True), num_stages=num_stages)
    return model

def convert(model_name, model, config, num_stages):
    if 'bloom' in model_name:
        return convert_bloom(model, config, num_stages)
    elif 'opt' in model_name:
        return convert_opt(model, config, num_stages)
