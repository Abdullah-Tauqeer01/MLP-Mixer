import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import copy

class MixerBlock(nn.Module):
    def __init__(self, config):
        super(MixerBlock, self).__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(config['sequence_length'], config['tokens_mlp_dim']),
            nn.GELU(),
            nn.Linear(config['tokens_mlp_dim'], config['sequence_length'])
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(config['hidden_dim'], eps=1e-6),
            nn.Linear(config['hidden_dim'], config['channels_mlp_dim']),
            nn.GELU(),
            nn.Linear(config['channels_mlp_dim'], config['hidden_dim'])
        )
        self.pre_norm = nn.LayerNorm(config['hidden_dim'], eps=1e-6)
        self.post_norm = nn.LayerNorm(config['hidden_dim'], eps=1e-6)

    def forward(self, x):
        h = x
        x = self.pre_norm(x)
        x = x.transpose(-1, -2)
        x = self.token_mlp(x)
        x = x.transpose(-1, -2)
        x = x + h

        h = x
        x = self.post_norm(x)
        x = self.channel_mlp(x)
        x = x + h
        return x


class MlpMixer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, patch_size=16, zero_head=False):
        super(MlpMixer, self).__init__()
        self.zero_head = zero_head
        self.num_classes = num_classes
        patch_size = _pair(patch_size)
        n_patches = (img_size // patch_size[0]) * (img_size // patch_size[1])
        config['sequence_length'] = n_patches

        self.stem = nn.Conv2d(in_channels=3,
                              out_channels=config['hidden_dim'],
                              kernel_size=patch_size,
                              stride=patch_size)
        self.head = nn.Linear(config['hidden_dim'], num_classes, bias=True)
        self.pre_head_ln = nn.LayerNorm(config['hidden_dim'], eps=1e-6)

        self.mixer_layers = nn.ModuleList()
        for _ in range(config['num_blocks']):
            mixer_layer = MixerBlock(config)
            self.mixer_layers.append(copy.deepcopy(mixer_layer))

    def forward(self, x, labels=None):
        x = self.stem(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)

        x = self.pre_head_ln(x)
        x = torch.mean(x, dim=1)
        logits = self.head(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits
