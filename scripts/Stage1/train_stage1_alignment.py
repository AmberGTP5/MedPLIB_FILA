# file: train_stage1_alignment.py
# THE ONE SCRIPT TO RULE THEM ALL
# This is the final, consolidated version containing all fixes for all previously encountered errors:
# 1. Complete ArgumentParser to fix `unrecognized arguments`.
# 2. Enriched `config` object to fix all `AttributeError` on startup.
# 3. Lightweight checkpoint saving logic to fix hanging during training.
# 4. Robust collate_fn and training loop to handle data errors gracefully.

import argparse
import os
import sys
import json
import warnings
from pathlib import Path
from functools import partial

import torch
import deepspeed
from tqdm import tqdm
import transformers
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    CLIPImageProcessor,
    logging as transformers_logging
)

# This pathing is important to ensure it finds your model and datasets directories
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model.fila_lisa import FILAForCausalLM
from datasets.alignment_dataset import AlignmentDataset

# Suppress unnecessary warnings for a cleaner log
warnings.filterwarnings("ignore", category=UserWarning)
transformers_logging.set_verbosity_error()


def setup_model_and_tokenizer(args):
    """Initializes the model, tokenizer, and all necessary configurations."""
    print("🚀 Setting up model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.version, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.version)
    
    # --- Enrich the config with all required custom parameters ---
    config.mm_vision_tower = args.vision_tower
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_hidden_size = 1024
    # For MedPLIB base architecture compatibility
    config.max_sample_point = 4096 
    
    model = FILAForCausalLM.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True
    )
    
    if tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]}) > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    # Attach tokenizer for internal access
    model.get_model().vision_tower.vision_tower.tokenizer = tokenizer

    if not model.get_model().vision_tower.vision_tower.is_loaded:
        model.get_model().vision_tower.vision_tower.load_model()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled.")
        
    print("✅ Model and tokenizer setup complete.")
    return model, tokenizer

def setup_training_parameters(model):
    """Freezes backbone and unfreezes only the alignment modules."""
    print("🔧 Configuring training parameters...")
    for p in model.parameters():
        p.requires_grad = False
        
    trainable_params_count = 0
    for name, p in model.named_parameters():
        if "cvfm_modules" in name or "mm_projector" in name:
            p.requires_grad = True
            trainable_params_count += p.numel()
            
    print(f"✅ Trainable parameters: {trainable_params_count:,}")
    if trainable_params_count == 0:
        raise ValueError("No trainable parameters found!")

def create_collate_fn(tokenizer):
    """Creates the data collator function for batching."""
    def collate_fn(batch):
        batch = [b for b in batch if isinstance(b, dict) and 'input_ids' in b and 'images_vit' in b]
        if not batch:
            return {}
        
        images_vit_list = [item['images_vit'] for item in batch]
        images_convnext_list = [item['images_convnext'] for item in batch]
        input_ids_list = [item['input_ids'] for item in batch]
        labels_list = [item['labels'] for item in batch]
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, True, tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels_list, True, -100)
        
        return {
            'images_vit': torch.stack(images_vit_list),
            'images_convnext': torch.stack(images_convnext_list),
            'input_ids': padded_input_ids,
            'labels': padded_labels,
        }
    return collate_fn

def main():
    parser = argparse.ArgumentParser(description="FILA Stage 1 Alignment Training")
    # All arguments the launch script provides are now defined.
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accumulation_steps", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action='store_true', default=False)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = setup_model_and_tokenizer(args)
    setup_training_parameters(model)
    dataset = AlignmentDataset(args.data_path, args.image_folder, tokenizer)
    
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {"type": "AdamW", "params": {"lr": args.lr}},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "gradient_clipping": 1.0,
        "scheduler": { "type": "WarmupDecayLR", "params": { "total_num_steps": args.max_steps, "warmup_num_steps": args.warmup_steps } }
    }
    
    model_engine, _, train_loader, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        training_data=dataset,
        collate_fn=create_collate_fn(tokenizer)
    )

    print("🔥 Starting training loop...")
    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(tqdm(train_loader, disable=model_engine.global_rank != 0)):
            if global_step >= args.max_steps:
                break
            
            if not batch:
                if model_engine.global_rank == 0: print(f"⚠️  Skipping empty batch at step {global_step}.")
                continue
            
            # This logic correctly handles the unpacked data from your collate_fn
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            loss = model_engine(**batch).loss
            model_engine.backward(loss)
            model_engine.step()
            global_step += 1

            if model_engine.global_rank == 0:
                if global_step % 10 == 0:
                    print(f"Step {global_step}, Loss: {loss.item():.4f}")
                
                # This handles intermediate checkpoint saving
                if global_step > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    with deepspeed.zero.GatheredParameters(model_engine.module.parameters(recurse=True), modifier_rank=0):
                        if model_engine.global_rank == 0:
                            state_to_save = { name: param.cpu().data for name, param in model_engine.module.named_parameters() if param.requires_grad }
                            torch.save(state_to_save, os.path.join(checkpoint_dir, "stage1_projector_cvfm.bin"))
                            print(f"\n✅ Lightweight checkpoint saved to {checkpoint_dir}")
        
        if global_step >= args.max_steps:
            break

    # 🔥 FIX: Add a final save after the training loop finishes.
    # This ensures that the model at the final step is always saved.
    if model_engine.global_rank == 0 and global_step > 0:
        print(f"\n🎉 Training finished at step {global_step}. Saving final model...")
        final_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}-final")
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        
        with deepspeed.zero.GatheredParameters(model_engine.module.parameters(recurse=True), modifier_rank=0):
            if model_engine.global_rank == 0:
                state_to_save = { name: param.cpu().data for name, param in model_engine.module.named_parameters() if param.requires_grad }
                torch.save(state_to_save, os.path.join(final_checkpoint_dir, "stage1_projector_cvfm.bin"))
                print(f"✅ Final lightweight model saved to {final_checkpoint_dir}")

if __name__ == "__main__":
    main()