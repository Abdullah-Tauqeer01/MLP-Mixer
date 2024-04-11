# MLP-Mixer Reproducibility Study

## Overview

This repository contains our reproducibility study for the "MLP-Mixer: An all-MLP Architecture for Vision" paper by Tolstikhin et al. The MLP-Mixer is a novel architecture that relies solely on multi-layer perceptrons (MLPs) without using convolutions or self-attention layers. Our study aimed to replicate the findings of the original paper under our computational constraints and available resources.

## Architecture

MLP-Mixer employs a series of MLP layers to process image patches (tokens) and mix features across channels. The architecture is designed without convolutions or self-attention, making it simpler yet effective for handling vision tasks.

## Dataset and Modifications

Due to computational limitations, instead of pretraining from scratch on ImageNet, we used pretrained weights and fine-tuned on:
- CIFAR-10
- ImageNet

## Results

Our findings show a decrease in performance compared to the original study:
- **Top-1 Accuracy on ImageNet**: 73.64% (Original: 76.44%)
- **Top-5 Accuracy on ImageNet**: 88.70% (Original: 93.63%)

The decrease in accuracy is attributed to various factors including limited computational resources and differences in training configurations.

## Reproduction Details

### Model Architecture

The MLP-Mixer model consists of token-mixing MLPs and channel-mixing MLPs applied sequentially in layers. Each type of MLP mixes information either spatially across tokens or across feature channels.

### Hardware Setup

Our experiments were conducted using a cluster of 2 RTX 8090 GPUs.

### Pre-training and Fine-tuning

We utilized pretrained models from the original authors, specifically the MLP-Mixer-B/16 and MLP-Mixer-L/16, pretrained on the ImageNet-21k dataset. These models were then fine-tuned on CIFAR-10 and ImageNet under the following configurations:
- **Optimizer**: Momentum SGD
- **Batch Size**: 512
- **Learning Rate**: Adjusted via a cosine schedule with linear warmup
- **Gradient Clipping**: At global norm 1
- **Weight Decay**: None

### Code

All code used for this study, including model training and evaluation scripts, is available in this repository. You can clone and follow the instructions to replicate our results or explore further:

```bash
git clone https://github.com/Abdullah-Tauqeer01/MLP-Mixer
cd MLP-Mixer
pip install -r requirements.txt
# Follow specific training and evaluation instructions
```

## Discussion

This study highlights challenges in reproducing the exact performance of novel architectures like MLP-Mixer, especially under resource constraints. Differences in hardware, training duration, and potentially unreported details in the original implementation may influence outcomes.


## Citation

If you find this study useful, please cite the original paper and our reproducibility study as follows:
```bibtex
@article{originalmlpmixer,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and others},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}
@misc{reproducibilitystudy,
  title={Reproducibility Study of MLP-Mixer},
  author={Tauqeera, Abdullah and Taherkhani, Hamed},
  year={2021},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/Abdullah-Tauqeer01/MLP-Mixer}}
}
```
