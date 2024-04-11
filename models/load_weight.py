import torch
import numpy as np
from os.path import join as pjoin

from torch import nn



a = ["token_mixing/Dense_0","token_mixing/Dense_1","channel_mixing/Dense_0","channel_mixing/Dense_1"]

def load_weights(model, weights):
    weights = np.load(weights, allow_pickle=True)
    with torch.no_grad():
        if model.zero_head:  ### remove this if add it in training loop
            nn.init.zeros_(model.head.weight)
            nn.init.zeros_(model.head.bias)
        else:
            model.head.weight.copy_(torch.from_numpy(weights["head/kernel"]).t())
            model.head.bias.copy_(torch.from_numpy(weights["head/bias"]).t())
        model.stem.weight.copy_(torch.from_numpy(weights["stem/kernel"]).permute(3, 2, 0, 1))
        model.stem.bias.copy_(torch.from_numpy(weights["stem/bias"]))
        model.pre_head_ln.weight.copy_(torch.from_numpy(weights["pre_head_layer_norm/scale"]))
        model.pre_head_ln.bias.copy_(torch.from_numpy(weights["pre_head_layer_norm/bias"]))

        for bname, block in enumerate(model.mixer_layers):
            ROOT = f"MixerBlock_{bname}"
           # print(f'ROOT is {ROOT}')
           # print(f'layers length is : {len(block.token_mlp_block.layers)}')
            with torch.no_grad():
                # print(f'without intialization')
                # for k,i in enumerate(block.token_mlp_block.layers):
                #     if k !=1:
                #         print(type(i))
                #         print(i.weight)
                
                for i in range(0, len(block.token_mlp), 2):
                    k = i
                    if k == 2:
                        k = 1
                    block.token_mlp[i].weight.copy_(torch.from_numpy(weights[ROOT+"/"+ a[k]+"/"+ "kernel"]).t())
                    block.token_mlp[i].bias.copy_(torch.from_numpy(weights[ROOT+"/"+ a[k]+"/"+ "bias"]).t())
                    block.channel_mlp[i+1].weight.copy_(torch.from_numpy(weights[ROOT+"/"+ a[k + 2]+"/"+ "kernel"]).t())
                    block.channel_mlp[i+1].bias.copy_(torch.from_numpy(weights[ROOT+"/"+ a[k + 2]+"/"+ "bias"]).t())

                # print(f'with intialization')
                # for k,i in enumerate(block.token_mlp_block.layers):
                #     if k !=1:
                #         print(type(i))
                #         print(i.weight)

                block.pre_norm.weight.copy_(torch.from_numpy(weights[ROOT+"/"+ "LayerNorm_0"+"/"+ "scale"]))
                block.pre_norm.bias.copy_(torch.from_numpy(weights[ROOT+"/"+ "LayerNorm_0"+"/"+ "bias"]))
                block.post_norm.weight.copy_(torch.from_numpy(weights[ROOT+"/"+ "LayerNorm_1"+"/"+ "scale"]))
                block.post_norm.bias.copy_(torch.from_numpy(weights[ROOT+"/"+ "LayerNorm_1"+"/"+ "bias"]))   
