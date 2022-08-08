# import os, time, gc, json, pickle, argparse, math
# import torch
# import torch.nn as nn
# import torch.utils.data as data
# import torch.distributed as dist
# import torch.multiprocessing as mp
import numpy as np
# from data.util import *
import copy

def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def init_para_frompretrained(m, pm, share_para=False):
    m.wte.weight = pm.wte.weight
    m.wpe.weight = pm.wpe.weight

    for i in range(min(len(m.h), len(pm.h))):
        m.h[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.copy(pm.h[i].ln_1.weight)
        m.h[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.copy(pm.h[i].ln_1.bias)
        m.h[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.copy(pm.h[i].attn.c_attn.weight)
        m.h[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.copy(pm.h[i].attn.c_attn.bias)
        m.h[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.copy(pm.h[i].attn.c_proj.weight)
        m.h[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.copy(pm.h[i].attn.c_proj.bias)
        m.h[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.copy(pm.h[i].ln_2.weight)
        m.h[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.copy(pm.h[i].ln_2.bias)
        m.h[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.copy(pm.h[i].mlp.c_fc.weight)
        m.h[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.copy(pm.h[i].mlp.c_fc.bias)
        m.h[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.copy(pm.h[i].mlp.c_proj.weight)
        m.h[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.copy(pm.h[i].mlp.c_proj.bias)

    m.ln_f.weight = pm.ln_f.weight if share_para else copy.copy(pm.ln_f.weight)
    m.ln_f.bias = pm.ln_f.bias if share_para else copy.copy(pm.ln_f.bias)

    print('finish')



def init_para_frompretrained_roberta(m, pm, share_para=True):
    m.embeddings.position_embeddings.weight = pm.embeddings.position_embeddings.weight
    m.embeddings.word_embeddings.weight = pm.embeddings.word_embeddings.weight
    m.embeddings.token_type_embeddings.weight = pm.embeddings.token_type_embeddings.weight

    for i in range(max(len(m.encoder.layer), len(pm.encoder.layer))): # RobertaLayer
        m.encoder.layer[i].attention.self.key.weight = pm.encoder.layer[i].attention.self.key.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.key.weight)
        m.encoder.layer[i].attention.self.query.weight = pm.encoder.layer[i].attention.self.query.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.query.weight)
        m.encoder.layer[i].attention.self.value.weight = pm.encoder.layer[i].attention.self.value.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.value.weight)
        m.encoder.layer[i].attention.output.dense.weight = pm.encoder.layer[i].attention.output.dense.weight if share_para else copy.copy(pm.encoder.layer[i].attention.output.dense.weight)

        m.encoder.layer[i].output.dense.weight = pm.encoder.layer[i].output.dense.weight if share_para else copy.copy(pm.encoder.layer[i].output.dense.weight)
        # m.encoder.layer[i].output.LayerNorm.weight = pm.encoder.layer[i].output.LayerNorm.weight if share_para else copy.copy(pm.encoder.layer[i].output.LayerNorm.weight)

        m.encoder.layer[i].intermediate.dense.weight = pm.encoder.layer[i].intermediate.dense.weight if share_para else copy.copy(pm.encoder.layer[i].intermediate.dense.weight)

        # m.encoder.layer[i].attention.output.LayerNorm.weight = pm.encoder.layer[i].attention.output.LayerNorm.weight if share_para else copy.copy(pm.encoder.layer[i].attention.output.LayerNorm.weight)
        # m.encoder.layer[i].attention.output.dropout.weight = pm.encoder.layer[i].attention.output.dropout.weight if share_para else copy.copy(pm.encoder.layer[i].attention.dropout.dense.weight)

    m.pooler.dense.weight = pm.pooler.dense.weight


def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """
    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s
    return f

def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)

    return f