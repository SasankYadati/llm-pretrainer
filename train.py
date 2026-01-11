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

def train_step(model, data_loader, device):
    loss_acc = 0.0
    grad_acc_steps = data_loader.grad_acc_steps
    for i in range(grad_acc_steps):
        batch = next(data_loader)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
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
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--n_tokens", type=int, default=int(1e6))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=4)


    # logging
    parser.add_argument("--run_name", type=str, default="default_run")

    args = parser.parse_args()

    os.environ["DEVICE"] = "cuda"

    set_all_seed(args.seed)

    wandb.init(
        project="llm-pretrainer",
        name=f"{args.run_name}_no_parallelism",
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
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_len

    model = Llama(model_config=model_config)
    model.to(dtype).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Create dataloader
    dataloader = MicroBatchDataLoader(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
        dataset_name=args.dataset_name,
        tokenizer_name=args.model_name,
        n_tokens=args.n_tokens,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
    )

    tokens_per_step = dataloader.global_batch_size * args.seq_len
    print("Tokens per step:", tokens_per_step)

    trained_token, step = 0, 0

    # Training loop
    while trained_token < args.n_tokens:

        step_start_time = time.time()
        optimizer.zero_grad()

        loss = train_step(model, dataloader, device)

        optimizer.step()

        step_duration = time.time() - step_start_time
        trained_token += tokens_per_step
        step += 1

        print(f"Step: {step}, Loss: {loss:.4f}, "
            f"Global batch size (with seq_len): {tokens_per_step}, "
            f"Tokens/s: {(tokens_per_step / step_duration)}, "
            f"Tokens: {(trained_token)}/{args.n_tokens}, "
            f"Memory usage: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
        )

        wandb.log({"loss": loss, "tokens_per_step": tokens_per_step, "tokens_per_second": tokens_per_step / step_duration,\
                "memory_usage": torch.cuda.memory_reserved() / 1e9, "trained_tokens": trained_token})

    wandb.finish()
