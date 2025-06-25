import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_constant_schedule_with_warmup
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm


def load_model_and_tokenizer(ckpt: str, mp: str = "bf16"):
    acc = Accelerator(mixed_precision=mp)
    config = AutoConfig.from_pretrained(ckpt)
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ckpt, 
        config=config,
        torch_dtype=torch.bfloat16 if mp == "bf16" 
               else torch.float16  if mp == "fp16" 
               else torch.float32,
    )
    model.gradient_checkpointing_enable()

    acc.print(f"Loaded model from {ckpt}")
    acc.print(f"Model config: {config}")
    config.use_cache = False  
    return acc, model, tok

def post_process(response: str) -> str:
    # find the first 'assistant' in the response and truncate it
    if "assistant" in response:
        response = response.split("assistant", 1)[1]
    else:
        raise ValueError("Response must contain 'assistant' to truncate.")
    
    return response

    ## (Optional) find the last \boxed{ }, and delete everything after it
    # last_boxed_pos = response.rfind(r'\boxed{')
    # if last_boxed_pos == -1:
    #     raise ValueError("Response must contain at least one \\boxed\{\}.")
    # start_pos = last_boxed_pos + 7  # len(r'\boxed{') = 7
    # brace_count = 0  
    # current_pos = start_pos
    
    # while current_pos < len(response) and brace_count >= 0:
    #     if response[current_pos] == '{':
    #         brace_count += 1
    #     elif response[current_pos] == '}':
    #         brace_count -= 1
    #     current_pos += 1
    
    # if brace_count == -1:
    #     return response[:current_pos]
    # else:
    #     raise ValueError("Response does not have a matching closing brace.")

def tokenize_function(example: Dict[str, str], tok):
    prompt = example["prompt"]
    response = post_process(example["response"])
    
    if tok.eos_token is None:
        raise ValueError("eos_token must be defined in the tokenizer configuration.")
    whole_string = tok.apply_chat_template(
        [{"role": "user",      "content": prompt}, 
         {"role": "assistant", "content": response}],
        add_generation_prompt=False,
        tokenize=False,
    )
    # print(f"Whole string: {whole_string}#############")

    encoded = tok(whole_string, 
                  truncation=True, 
                  max_length=tok.model_max_length,
                  return_tensors="pt")
    
    ids       = encoded["input_ids"][0]
    mask      = encoded["attention_mask"][0]  ## It should be all ones at this point
    labels    = ids.clone()
    start_id  = tok.convert_tokens_to_ids("<|im_start|>")  ## 151644
    assist_id = tok.convert_tokens_to_ids("assistant")     ## 77091

    ## Find the first start + assist, and mask everything until then
    idx = None
    for i in range(len(ids) - 1):
        if ids[i] == start_id and ids[i + 1] == assist_id:
            idx = i + 2
            break
    if idx is None:
        raise ValueError("Assistant role marker not found.")
    labels[:idx] = -100

    return {
        "input_ids":       ids,
        "attention_mask":  mask,
        "labels":          labels,
    }

def compute_loss(model, batch):
    outputs = model(**batch)
    return outputs.loss

def evaluate_model(model, eval_loader, accelerator):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            loss = compute_loss(model, batch)
            total_loss += accelerator.gather(loss).mean().item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else -1

def save_checkpoint(model, tokenizer, accelerator, output_dir, step, save_total_limit=-1):
    save_path = os.path.join(output_dir, f"step-{step}")
    os.makedirs(save_path, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    accelerator.print(f"Checkpoint saved to {save_path}")

    if save_total_limit > 0:
        # Remove old checkpoints if save_total_limit is set
        checkpoints = sorted(Path(output_dir).glob("step-*"), key=os.path.getmtime)
        if len(checkpoints) > save_total_limit:
            for checkpoint in checkpoints[:-save_total_limit]:
                accelerator.print(f"Removing old checkpoint: {checkpoint}")
                checkpoint.rmdir()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, default="train-data/pi1_r128_responses.parquet")
    parser.add_argument("--ckpt", type=str, default="/homes/gws/lxh22/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--output_dir", type=str, default="save/Qwen2.5-Math-1.5B")

    parser.add_argument("--effective_batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    # parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    acc, model, tok = load_model_and_tokenizer(args.ckpt, mp=args.precision)
    set_seed(args.seed, device_specific=True)
    aprint = acc.print

    # ----------------------------- dataset ------------------------------ #

    df = load_dataset("parquet", data_files=args.train_file, split="train")
    if args.val_fraction > 0.0:
        split = df.train_test_split(test_size=args.val_fraction, seed=args.seed)
        train_ds = split["train"]
        eval_ds  = split["test"]
    else:
        train_ds = df
        eval_ds  = None

    # print(f"Loaded dataset with {len(train_ds)} training examples \
        #    and {len(eval_ds) if eval_ds else 0} validation examples.")

    tokenize_fn = lambda ex: tokenize_function(ex, tok)
    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, num_proc=8)
    if eval_ds:
        eval_ds = eval_ds.map(tokenize_fn, remove_columns=eval_ds.column_names, num_proc=8)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        padding="longest",
        max_length=tok.model_max_length,
        pad_to_multiple_of=8 if args.precision == "bf16" else None,
    )
    per_device_batch_size = args.effective_batch // acc.num_processes
    actual_effective_batch = per_device_batch_size * acc.num_processes
    steps = args.steps
    epochs = math.ceil(steps / (len(train_ds) // actual_effective_batch)) ## ?
    
    train_loader = DataLoader(
        train_ds, batch_size=per_device_batch_size, shuffle=True, drop_last=True, collate_fn=data_collator
    )
    if eval_ds:
        eval_loader = DataLoader(
            eval_ds, batch_size=per_device_batch_size, shuffle=False, collate_fn=data_collator
        )
    else:
        eval_loader = None

    
    # ----------------------------- training ------------------------------ #

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.95), 
        eps=1e-8
    )
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps, 
    )
    model, optimizer, train_loader, lr_scheduler = acc.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    if eval_loader:
        eval_loader = acc.prepare(eval_loader)
    
    aprint(f"Training arguments:")
    aprint(f"  Num examples = {len(train_ds)}")
    aprint(f"  Per device batch size = {per_device_batch_size}")
    aprint(f"  Total train batch size = {actual_effective_batch}")
    aprint(f"  Total optimization steps = {steps}")
    aprint(f"  Number of epochs = {epochs}")
    aprint(f"  Expected effective batch size = {args.effective_batch}")

    
    ## Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    progress_bar = tqdm(range(steps), disable=not acc.is_main_process)
    completed_steps = 0
    total_loss: float = 0.0
    for __ in range(0, epochs):
        model.train()
        for _, batch in enumerate(train_loader):
            with acc.accumulate(model):
                loss = compute_loss(model, batch)
                acc.backward(loss)
                total_loss += loss.item()
                acc.clip_grad_norm_(model.parameters(), 1.0) ## enable gradient clipping
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            completed_steps += 1
                
            ## Logging
            if completed_steps % args.logging_steps == 0:
                avg_loss = total_loss / args.logging_steps
                learning_rate = lr_scheduler.get_last_lr()[0]
                
                aprint(f"Step {completed_steps}: loss={avg_loss:.4f}, lr={learning_rate:.2e}")
                
                if eval_loader and completed_steps % args.logging_steps == 0:
                    eval_loss = evaluate_model(model, eval_loader, acc)
                    aprint(f"Step {completed_steps}: eval_loss={eval_loss:.4f}")
                    model.train()  
                total_loss = 0.0
            
            ## Save checkpoint
            if args.save_steps > 0 and completed_steps % args.save_steps == 0:
                acc.wait_for_everyone()
                if acc.is_main_process:
                    save_checkpoint(
                        model, tok, acc, args.output_dir, 
                        str(completed_steps), args.save_total_limit
                    )
            
            if completed_steps >= steps:
                break
        
        if completed_steps >= steps:
            break

    acc.wait_for_everyone()
    if acc.is_main_process:
        save_checkpoint(model, tok, acc, args.output_dir, "final", args.save_total_limit)


if __name__ == "__main__":
    main()
