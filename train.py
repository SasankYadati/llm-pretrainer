"""
uv run train.py
"""
import argparse
import os
from utils import set_all_seed
from transformers import AutoConfig
from model import Llama
import torch
from torch.optim import AdamW
import time
import wandb
from dataloader import MicroBatchDataLoader
import torch.nn.functional as F

# A100 80GB BF16 peak throughput
A100_PEAK_FLOPS = 312 * 10 ** 12

def calculate_mfu(tokens_per_second, num_params):
    """
    Calculate Model FLOPs Utilization (MFU) using 6*N*D as approximation for flops required.

    Args:
        tokens_per_second: Training throughput (tokens processed per second)
        num_params: Number of model parameters (N)

    Returns:
        mfu: Model FLOPs Utilization as a percentage (0-100)
    """
    actual_flops = 6 * num_params * tokens_per_second
    return 100 * actual_flops / A100_PEAK_FLOPS

def train_step(model, data_loader, device, dtype, grad_acc_steps):
    loss_acc = 0.0
    for i in range(grad_acc_steps):
        batch = next(data_loader)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        with torch.autocast(device_type='cuda', dtype=dtype):
            outputs = model(input_ids=input_ids)
            batch_size, seq_len = input_ids.shape
            target_ids = target_ids.reshape(-1)
            outputs = outputs.view(seq_len * batch_size, -1)
            loss = F.cross_entropy(outputs, target_ids, reduction='mean') / grad_acc_steps

        loss.backward()
        loss_acc += loss.item()
    return loss_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for Llama")


    # model config
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M")

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=512)
    # Global batch: 32 * 16 * 512 = 262K tokens per optimizer step
    # Also holds => 262K = 64 * 8 * 512
    parser.add_argument("--micro_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    # Batch size warmup: ramp grad_acc_steps from 1 to target over this many tokens (0 = disabled)
    parser.add_argument("--batch_size_warmup_tokens", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", type=str, default="sample-10BT")
    parser.add_argument("--n_tokens", type=int, default=10000000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_proc", type=int, default=48)


    # logging
    parser.add_argument("--run_name", type=str, default="no_parallelism")

    args = parser.parse_args()

    os.environ["DEVICE"] = "cuda"

    set_all_seed(args.seed)

    wandb.init(
        project="llm-pretrainer",
        name=f"{args.run_name}",
        config={
            "tensor_parallel_size": 1, # no parallelism
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "model": args.model_name,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
    )


    device = torch.device("cuda")
    dtype = torch.bfloat16

    model_config = AutoConfig.from_pretrained(args.model_name)

    model = Llama(model_config=model_config)
    model = torch.compile(model)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Create dataloader
    dataloader = MicroBatchDataLoader(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer_name=args.model_name,
        n_tokens=args.n_tokens,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
    )

    target_tokens_per_step = dataloader.global_batch_size * args.seq_len
    print(f"Target tokens per step: {target_tokens_per_step:,}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    trained_token, step = 0, 0

    # Warmup steps to exclude from MFU measurement (torch.compile overhead)
    warmup_steps = 5

    # Training loop
    while trained_token < args.n_tokens:

        # Batch size warmup: linearly ramp grad_acc_steps from 1 to target
        if args.batch_size_warmup_tokens == 0:
            current_grad_acc_steps = args.gradient_accumulation_steps
        else:
            current_grad_acc_steps = min(
                1 + (trained_token * (args.gradient_accumulation_steps - 1)) // args.batch_size_warmup_tokens,
                args.gradient_accumulation_steps
            )
        tokens_per_step = args.micro_batch_size * current_grad_acc_steps * args.seq_len

        step_start_time = time.time()
        optimizer.zero_grad()

        loss = train_step(model, dataloader, device, dtype, current_grad_acc_steps)

        optimizer.step()

        step_duration = time.time() - step_start_time
        trained_token += tokens_per_step
        step += 1

        tokens_per_second = tokens_per_step / step_duration

        is_warmup = step <= warmup_steps
        mfu = None if is_warmup else calculate_mfu(tokens_per_second, num_params)
        mfu_str = "warming up" if is_warmup else f"{mfu:.1f}%"
        print(
            f"Step: {step}, Loss: {loss:.4f}, "
            f"Tokens/s: {tokens_per_second:.0f}, "
            f"MFU: {mfu_str}, "
            f"Tokens: {trained_token}/{args.n_tokens}, "
            f"Memory: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
        )

        wandb.log({
            "loss": loss,
            "tokens_per_step": tokens_per_step,
            "tokens_per_second": tokens_per_second,
            "mfu": mfu,
            "grad_acc_steps": current_grad_acc_steps,
            "memory_usage": torch.cuda.memory_reserved() / 1e9,
            "trained_tokens": trained_token,
        })

    wandb.finish()
