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
from helper.func import set_constant,evaluate,setup_scheduler, save_model,setup_scheduler,setup_optimizer,set_seed,setup_model,setup_optimizer

def train(args, model, train_loader, test_loader):
    optimizer = setup_optimizer(args, model)
    scheduler = setup_scheduler(args, optimizer)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.model_name))
    os.makedirs(args.output_dir, exist_ok=True)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    print(f"Total optimization steps: {args.num_steps}")
    print(f"Instantaneous batch size per GPU: {args.train_batch_size}")
    print(f"Gradient Accumulation steps: {args.gradient_accumulation_steps}")
    model.zero_grad()
    global_step, best_acc = 0, 0
    while global_step < args.num_steps:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=False)
        train_loss = 0
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            train_loss += loss.item() * args.gradient_accumulation_steps
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, train_loss / (step + 1))
                )
                if global_step % args.eval_every == 0:
                    accuracy = evaluate(args, model, test_loader)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

        train_loss = 0

    writer.close()
    print(f"Best Accuracy: {best_acc:.5f}")
    print("End Training!")


def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--model_name", required=True,
                        help="Name of run used for saving model and monitoring later on")
    parser.add_argument("--Mixer_type", required=True, choices=['S/32','S/16','B/32','B/16','L/32','L/16','H/14'],
                        help="Model needed to be used. Model can be viewed in models.config file")
    parser.add_argument("--weight_dir", required=True,
                        help="location to load pretrained weights, "
                        "if they are not already downloaded script will download them to the given directory")
    args = parser.parse_args()
   
    num_gpus = torch.cuda.device_count()
    print(torch.cuda.is_available())
    # Specify the GPUs you want to use
    gpu_ids = list(range(num_gpus))
    
    # Set the device to the first GPU in the list
    device = torch.device(f"cuda:{gpu_ids[0]}")
    
    print(f"Using GPUs: {gpu_ids}")
    
    args.n_gpu = torch.cuda.device_count()
    args = set_constant(args)
    args.device = device
    args.img_size = 224
    set_seed(args)
    model = setup_model(args)
    train_loader, test_loader = get_loader(args)
    train(args, model, train_loader, test_loader)

if __name__ == "__main__":
    main()