from network import Attention, MultiheadSelfAttention, FeedForwardBlock, SelfAttentionNet, Value1d, Policy1d, PVNet
import torch
import torch.nn as nn
from torch.quantization import per_channel_dynamic_qconfig
import numpy as np
from PyBGEnv import qubic
from alphazero import AlphaZeroConfig, AlphaZero, PVMCTS

import os
from main import eval_func

a = torch.FloatTensor(np.zeros((41, 64, 25)))
shape = (-1, 64, 2)
att = MultiheadSelfAttention(32, 4, 0.9, 64)
ffb = FeedForwardBlock(32, (64, 32))
sa = SelfAttentionNet(
    length=64, 
    hidden_size=32, 
    layers=[att, ffb, att, ffb, att, ffb, att, ffb], 
    in_ch=25, out_ch=33)
print(sa(a).shape)

input_net = SelfAttentionNet(
    in_ch=2, out_ch=8,
    hidden_size=32, layers=[att, ffb, att, ffb], length=64
)

val = Value1d(
    in_f=8,
    out_f=2,
    in_fc=128
)

pol = Policy1d(
    in_feature=8,
    out_feature=2,
    in_fc=128,
    out_fc=16
)

def transform(state):
    state = state.reshape([-1, 2, 64])
    return torch.transpose(state, dim0=1, dim1=2)

pv = PVNet(input_layer=input_net, policy_layer=pol, value_layer=val, transform=transform)

env = qubic

# ts_model = torch.jit.script(pv)
# q_pv = torch.quantization.quantize_dynamic_jit(ts_model, {'': per_channel_dynamic_qconfig})
q_pv = torch.quantization.quantize_dynamic(pv, {nn.Linear, nn.SiLU})

eval_func(
    PVMCTS(q_pv, alpha=0.35, env=env, epsilon=0, num_sims=50)
)
exit(0)

# config = AlphaZeroConfig(
#     buffer_size=40000,
#     log_name="alphazero_qubic_test",
#     prob_action_th=4,
#     selfplay_n=300,
#     episode=1000,
#     batch_size=128,
#     epoch=10,
#     sim_n=50,
#     save_dir="checkpoint",
#     load=False,
#     load_path="checkpoint/alphazero/latest.pth",
#     quantize=False,
# )

if __name__ == "__main__":
    if config.load:
        path = config.load_path
        assert os.path.exists(path)
        pv.load_state_dict(torch.load(path))
    
    alphazero = AlphaZero(pv, config, env, eval_func=eval_func)

    alphazero.train()