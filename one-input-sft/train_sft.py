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
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from tqdm.auto import tqdm


def load_model_and_tokenizer(ckpt: str, mp: str = "bf16", ds_config = None):
    acc = Accelerator(mixed_precision=mp, deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config))
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
    model.config.use_cache = False  

    acc.print(f"Loaded model from {ckpt}")
    acc.print(f"Model config: {config}")
    config.use_cache = False  
    return acc, model, tok

def post_process(filter_option: int, response: str) -> str:
    # find the first 'assistant' in the response and truncate it
    if "assistant" in response:
        response = response.split("assistant", 1)[1]
    else:
        raise ValueError("Response must contain 'assistant' to truncate.")
    
    if filter_option == 0:
        return response
    elif filter_option == 1:
        ## (Optional, filter 1) find the last \boxed{ }, and delete everything after it
        last_boxed_pos = response.rfind(r'\boxed{')
        if last_boxed_pos == -1:
            raise ValueError("Response must contain at least one \\boxed\{\}.")
        start_pos = last_boxed_pos + 7  # len(r'\boxed{') = 7
        brace_count = 0  
        current_pos = start_pos
        
        while current_pos < len(response) and brace_count >= 0:
            if response[current_pos] == '{':
                brace_count += 1
            elif response[current_pos] == '}':
                brace_count -= 1
            current_pos += 1
        
        if brace_count == -1:
            return response[:current_pos]
        else:
            raise ValueError("Response does not have a matching closing brace.")
    else:
        raise ValueError(f"Unknown filter option {filter_option}")

def tokenize_function(filter_option: int, example: Dict[str, str], tok):
    prompt = example["prompt"]
    response = post_process(filter_option, example["response"])
    
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
                  max_length=2048,
                  return_tensors="pt")
    
    if filter_option == 0:
        ids   = encoded["input_ids"][0]
        mask  = encoded["attention_mask"][0]  ## It should be all ones at this point
    elif filter_option == 1:
        ## Filter 1: remove the last <|im_end|> and '\n' tokens
        ids   = encoded["input_ids"][0][:-2]
        mask  = encoded["attention_mask"][0][:-2]  ## It should be all ones at this point
    else:
        raise ValueError(f"Unknown filter option {filter_option}")
    
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
    labels[mask == 0] = -100

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
    parser.add_argument("--ckpt", type=str, default="/homes/gws/lxh22/models/Qwen2.5-Math-7B")
    parser.add_argument("--output_dir", type=str, default="/local1/lxh/save/Qwen2.5-Math-7B-filter1") ## change this

    parser.add_argument("--effective_batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=150)                                   ## change this
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    ## save every save_steps, or save_steps_before if completed_steps < save_before
    parser.add_argument("--save_steps", type=int, default=10)                               ## change this  
    parser.add_argument("--save_before", type=int, default=0)                               ## change this    
    parser.add_argument("--save_steps_before", type=int, default=1)                         ## change this
    parser.add_argument("--save_total_limit", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=5)                             ## change this   

    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-option", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_device_batch_size = args.per_device_batch_size
    gradient_accumulation_steps = max(1, args.effective_batch // (per_device_batch_size * world_size))
    # if args.effective_batch % (per_device_batch_size * world_size) != 0:
        # raise ValueError(f"Effective batch size {args.effective_batch} must be divisible by "
                        #  f"per_device_batch_size {per_device_batch_size} * world_size {world_size}.")
    actual_effective_batch = per_device_batch_size * world_size * gradient_accumulation_steps
    steps = args.steps
    ds_config = {
        "train_micro_batch_size_per_gpu": per_device_batch_size,
        "train_batch_size": actual_effective_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": {"enabled": args.precision == "bf16"},
        "zero_optimization": {
                              "stage": 2, 
                              "offload_optimizer": {"device": "cpu"}, 
                              "offload_param": {"device": "none"}
                             },
        "gradient_clipping": 1.0,
    }
    acc, model, tok = load_model_and_tokenizer(args.ckpt, mp=args.precision, ds_config=ds_config)
    set_seed(args.seed, device_specific=True)
    aprint = acc.print

    ## sleep 30s for debugging
    # import time
    # aprint("Sleeping for 30 seconds for debugging...")
    # time.sleep(30)

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

    filter_option = args.filter_option
    tokenize_fn = lambda ex: tokenize_function(filter_option, ex, tok)
    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, num_proc=8)
    if eval_ds:
        eval_ds = eval_ds.map(tokenize_fn, remove_columns=eval_ds.column_names, num_proc=8)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        padding="longest",
        max_length=2048,
        pad_to_multiple_of=8 if args.precision == "bf16" else None,
    )
    # aprint(f"model max length: {tok.model_max_length}") ## 131072
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

    epochs = math.ceil(steps / (len(train_ds) // actual_effective_batch)) ## ?
    ## epochs = math.ceil(steps / (len(train_ds) // (per_device_batch_size * world_size)))
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
            ##------------------------------- debug -------------------------------##
            # print(f"batch:", batch)
            # with open("debug.log", "w") as f:
            #     print(f"batch input_id shape: {batch['input_ids'].shape}", file=f)
            #     print(f"batch attention_mask size: {batch['attention_mask'].shape}", file=f)
            #     print(f"batch labels size: {batch['labels'].shape}", file=f)
            #     ## print the whole input_ids to file
            #     ids = batch['input_ids'][1].cpu()
            #     print(f"batch input_ids: {ids}", file=f)
            #     ## print the eos_token id
            #     print(f"eos_token id: {tok.eos_token_id}", file=f)
            #     print(f"pad_token id: {tok.pad_token_id}", file=f)
            #     len_ids = len(ids)
            #     for i in range(len_ids):
            #         if ids[len_ids-1-i] != 151643:
            #             start = len_ids - 10 - i
            #             end = len_ids - i + 5
            #             print(f"position {start} to {end-1}: {ids[start:end]}", file=f)
            #             print(f"mask {start} to {end-1}: {batch['attention_mask'][1][start:end]}", file=f)
            #             print(f"labels {start} to {end-1}: {batch['labels'][1][start:end]}", file=f)
            #             break
            #     print(f"decode 151644: {tok.decode(151644)}", file=f)       ## <|im_start|>
            #     print(f"decode 151643: {tok.decode(151643)}", file=f)       ## <|endoftext|>
            #     print(f"decode 151645: {tok.decode(151645)}", file=f)       ## <|im_end|>
            #     print(f"decode 198: f{tok.decode(198)}f", file=f)           ## \n
            #     print(f"decode all: #{tok.decode([ 11520,  79075,     90,     16,     17,     13,     23,     92, 151645, 198, 151643, 151643, 151643, 151643, 151643])}#", file=f)
            
            # exit(0)
            ##------------------------------- debug -------------------------------##

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
            should_save = (completed_steps <= args.save_before and completed_steps % args.save_steps_before == 0) or \
                          (completed_steps >  args.save_before and completed_steps % args.save_steps == 0)
            if should_save:            
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
    # if acc.is_main_process:
    #     save_checkpoint(model, tok, acc, args.output_dir, "final", args.save_total_limit)


if __name__ == "__main__":
    main()
