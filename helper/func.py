import logging
import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.model import MlpMixer
import models.configs as CONFIGS
from models.load_weight import load_weights
from utils.dataloader import get_loader
from torch.cuda import amp
from sklearn.metrics import accuracy_score
import math

def warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, cycles=0.5, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)

def setup_model(args):
    config = CONFIGS.CONFIGS[args.Mixer_type]
    model = MlpMixer(config, args.img_size, num_classes=args.num_classes, patch_size=16, zero_head=True)
    load_weights(model, args.weight_dir)
    model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model configuration: {config}")
    print(f"Training parameters: {args}")
    print(f"Total Parameters: {num_params / 1e6:.1f}M")
    return model

def setup_optimizer(args, model):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
    return optimizer

def setup_scheduler(args, optimizer):
    scheduler = warmup_cosine_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.num_steps)
    return scheduler

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.model_name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    print(f"Saved model checkpoint to [DIR: {args.output_dir}]")

def evaluate(args, model, test_loader):
    print("Running Validation...")
    print(f"Num steps: {len(test_loader)}")
    print(f"Batch size: {args.eval_batch_size}")
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    loss_fct = torch.nn.CrossEntropyLoss()
    eval_loss = 0
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        with torch.no_grad():
            logits = model(x)
            loss = loss_fct(logits, y)
            eval_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
        ...
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % (eval_loss / (step + 1)))
    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = accuracy_score(all_label, all_preds)
    print(f"Validation Loss: {eval_loss / len(test_loader):.5f}")
    print(f"Validation Accuracy: {accuracy:.5f}")
    return accuracy

def set_constant(args):
    args.output_dir = "output"
    args.train_batch_size = 64
    args.eval_batch_size = 64
    args.eval_every = 100
    args.learning_rate = 3e-2
    args.weight_decay = 0
    args.num_steps = 10000
    args.warmup_steps = 500
    args.max_grad_norm = 1.0
    args.seed = 42
    args.gradient_accumulation_steps = 1
    args.fp16 = False
    args.fp16_opt_level = 'O2'
    args.loss_scale = 0
    args.num_classes = 10
    return args
