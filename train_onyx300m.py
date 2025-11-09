#!/usr/bin/env python3
"""
Training script for Onyx 125M with packed sequences.
Simplified for training on small datasets.

Author: Marvin Tutt, Caia Tech
"""

import os
import sys
import json
import glob
import time
import math
import shutil
import signal
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Import our model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_onyx125m import Onyx125M, OnyxConfig, build_block_causal_mask

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("W&B not available for logging", stacklevel=2)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Data
    data_glob: str = "/Users/owner/Desktop/caiatech/models/onyx-001/data/huggingface/cosmopedia_filtered/stories_filtered.jsonl"
    tokenizer: str = "NousResearch/Hermes-2-Pro-Llama-3-8B"  # Llama 3 tokenizer
    eval_glob: Optional[str] = None

    # Training
    tokens_per_step: int = 32_000  # Smaller for CPU
    max_steps: Optional[int] = 1000
    train_tokens_target: Optional[int] = 100_000_000  # 100M tokens
    max_seq_len: int = 2048  # Fixed context length
    fill_ratio: float = 0.9
    block_cross_doc_attention: bool = False

    # Model
    compile_model: bool = False  # Disabled for CPU
    gradient_checkpointing: bool = False

    # Optimization
    learning_rate: float = 6e-4  # Higher LR for smaller model
    min_lr: float = 6e-5
    warmup_steps: int = 300      # ~10% of training (for 3125 total steps)
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # Precision
    bf16: bool = True   # Use BF16 on GPU (H100/A100/4090) - auto-disabled on CPU
    fp16: bool = False  # BF16 preferred over FP16

    # Checkpointing
    save_dir: str = "./checkpoints_125m"
    save_every_steps: int = 500
    eval_every_steps: int = 100
    resume: Optional[str] = None  # "auto" or path

    # Logging
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    log_every: int = 1  # Log every step for detailed monitoring
    loss_ema_alpha: float = 0.1  # EMA smoothing factor (0.1 = 10% new, 90% old)

    # System
    num_workers: int = 0  # Single-threaded for simplicity
    pin_memory: bool = False
    seed: int = 1337
    dry_run: bool = False


# ============================================================================
# Dataset and Data Loading
# ============================================================================

class PackedDocument:
    """Single packed sequence with multiple documents"""
    def __init__(self, input_ids: List[int], doc_spans: List[Tuple[int, int]], seq_len: int):
        self.input_ids = input_ids
        self.doc_spans = doc_spans
        self.seq_len = seq_len


class StreamingPackedDataset(IterableDataset):
    """
    Streaming dataset that packs documents into sequences.
    Emits variable-length sequences; we pad in collate_fn to LOCAL max.
    """

    def __init__(
        self,
        file_pattern: str,
        tokenizer,
        max_seq_len: int = 2048,
        fill_ratio: float = 0.9,
        eod_token_id: int = 3,
        pad_token_id: int = 0,
        seed: int = 42
    ):
        self.file_pattern = file_pattern
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.fill_ratio = fill_ratio
        self.eod_token_id = eod_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        # Files
        self.files = sorted(glob.glob(file_pattern, recursive=True))
        if not self.files:
            raise ValueError(f"No files found matching {file_pattern}")

        # Stats
        self.stats = defaultdict(int)
        self._epoch = 0  # Track iterations for reproducible shuffling

    def _pack_documents(self, doc_iterator, target_len: int) -> Optional[PackedDocument]:
        """Pack multiple *tokenized* documents (lists of ints) into one sequence."""
        packed_ids, doc_spans = [], []
        current_pos = 0
        target_fill = int(target_len * self.fill_ratio)

        for tokens in doc_iterator:
            try:
                if not tokens:
                    continue

                new_len = current_pos + len(tokens) + (1 if packed_ids else 0)  # +1 for EOD
                if new_len > target_len:
                    if current_pos >= target_fill:
                        break
                    if len(tokens) > target_len - current_pos:
                        continue

                if packed_ids:
                    packed_ids.append(self.eod_token_id)
                    current_pos += 1

                start = current_pos
                packed_ids.extend(tokens)
                current_pos += len(tokens)
                doc_spans.append((start, current_pos))

                if current_pos >= target_fill:
                    break

            except Exception:
                continue

        # Add end token
        if packed_ids and (current_pos + 1) <= target_len and packed_ids[-1] != self.eod_token_id:
            packed_ids.append(self.eod_token_id)
            current_pos += 1

        if not packed_ids:
            return None

        return PackedDocument(packed_ids, doc_spans, len(packed_ids))

    def _read_jsonl_file(self, filepath: str):
        """Read and yield documents from JSONL file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        self.stats["json_decode_errors"] += 1
                        if self.stats["json_decode_errors"] <= 3:  # Log first few errors
                            warnings.warn(f"JSON decode error in {filepath} line {line_num}", stacklevel=2)
                        continue
        except Exception as e:
            warnings.warn(f"Error reading file {filepath}: {e}", stacklevel=2)

    def _batched_token_stream(self, files, batch_size: int = 32):
        """
        Read raw docs, batch-tokenize, and yield token sequences.
        """
        buf = []
        def _flush(buf):
            if not buf:
                return []
            enc = self.tokenizer(
                buf,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False
            )
            return enc["input_ids"]

        for fp in files:
            for doc in self._read_jsonl_file(fp):
                text = doc if isinstance(doc, str) else (doc.get("text", "") or doc.get("content", "")) if isinstance(doc, dict) else ""
                if not text.strip():
                    continue
                buf.append(text)
                if len(buf) >= batch_size:
                    for tokens in _flush(buf):
                        yield tokens
                    buf.clear()
        # tail
        for tokens in _flush(buf):
            yield tokens

    def __iter__(self):
        """Iterate over packed sequences"""
        files = list(self.files)
        # Use epoch counter for reproducible but varied shuffling
        random.Random(self.seed + self._epoch).shuffle(files)
        self._epoch += 1

        def token_stream():
            for filepath in files:
                yield from self._batched_token_stream([filepath])

        doc_iter = token_stream()

        while True:
            packed = self._pack_documents(doc_iter, self.max_seq_len)
            if packed is None:
                # Rewind stream
                doc_iter = token_stream()
                continue

            # Labels (shift by 1, -100 on last token)
            labels = packed.input_ids[1:] + [-100]

            self.stats["total_sequences"] += 1
            self.stats["total_tokens"] += packed.seq_len

            yield {
                "input_ids": torch.tensor(packed.input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "seq_len": packed.seq_len,
                "doc_spans": packed.doc_spans
            }


def create_dataloader(
    config: TrainingConfig,
    tokenizer,
    is_eval: bool = False
) -> DataLoader:
    """Create dataloader for training or evaluation"""

    file_pattern = config.eval_glob if is_eval else config.data_glob
    if not file_pattern:
        return None

    dataset = StreamingPackedDataset(
        file_pattern=file_pattern,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        fill_ratio=config.fill_ratio,
        eod_token_id=tokenizer.convert_tokens_to_ids("<eod>"),
        pad_token_id=tokenizer.pad_token_id,
        seed=config.seed
    )

    # Pad to LOCAL max per batch
    def collate_fn(batch):
        seqs = [x["input_ids"] for x in batch]
        labs = [x["labels"] for x in batch]
        seq_lens = torch.tensor([x["seq_len"] for x in batch], dtype=torch.long)
        doc_spans = [x["doc_spans"] for x in batch]

        S = max(int(t.size(0)) for t in seqs)
        pad_id = tokenizer.pad_token_id

        def pad_to(t: torch.Tensor, length: int, pad_value: int):
            if t.size(0) == length:
                return t
            out = torch.full((length,), pad_value, dtype=t.dtype)
            out[: t.size(0)] = t
            return out

        input_ids = torch.stack([pad_to(t, S, pad_id) for t in seqs])
        labels = torch.stack([pad_to(t, S, -100) for t in labs])

        return {"input_ids": input_ids, "labels": labels, "seq_lens": seq_lens, "doc_spans": doc_spans}

    # Batch size > 1 enables varlen FlashAttention for better GPU efficiency
    # Each sequence is already packed with multiple docs (up to max_seq_len tokens)
    # Gradient accumulation is token-based, so batch_size affects GPU util but not training dynamics
    effective_batch_size = 8 if not is_eval else 4

    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )


# ============================================================================
# Training Utilities
# ============================================================================

def prepare_attention_inputs(
    batch: Dict[str, Any],
    config: TrainingConfig,
    device: torch.device
) -> Dict[str, Any]:
    """Build attention mask if needed."""
    seq_lens = batch["seq_lens"].to(device)
    doc_spans = batch.get("doc_spans")
    if config.block_cross_doc_attention:
        attn_mask = build_block_causal_mask(
            seq_lens=seq_lens,
            doc_spans=doc_spans,
            block_cross_doc=True,
            device=device
        )
    else:
        attn_mask = None
    return {"seq_lens": seq_lens, "attn_mask": attn_mask, "doc_spans": doc_spans}


class Trainer:
    """Main training orchestrator"""

    def __init__(self, model: Onyx125M, config: TrainingConfig, tokenizer, device: torch.device):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Mixed precision (disabled for CPU)
        self.use_amp = False
        self.scaler = None

        # Scheduler
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.global_tokens = 0
        self.accum_token_cap = self.config.tokens_per_step

        # Metrics
        self.metrics = defaultdict(list)
        self.loss_history = deque(maxlen=1000)  # Keep last 1000 losses
        self.step_times = deque(maxlen=100)  # Keep last 100 step times

        # Moving averages for stability
        self.loss_ema = None
        self.ema_alpha = config.loss_ema_alpha
        self.grad_norm_ema = None
        self.throughput_ema = None

        # Checkpointing
        self.best_eval_loss = float('inf')
        self.checkpoint_dir = Path(config.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        self.use_wandb = WANDB_AVAILABLE and config.wandb_project
        if self.use_wandb:
            wandb.init(project=config.wandb_project, name=config.wandb_run, config=asdict(config), resume="allow")

    def _create_optimizer(self) -> AdamW:
        param_groups = self.model.get_param_groups()

        # Try to use fused AdamW for 10-20% speedup (PyTorch 2.0+, CUDA only)
        optimizer_kwargs = {
            "lr": self.config.learning_rate,
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_eps
        }

        if self.device.type == "cuda":
            try:
                optimizer = AdamW(
                    [
                        {"params": param_groups["decay"], "weight_decay": self.config.weight_decay},
                        {"params": param_groups["no_decay"], "weight_decay": 0.0},
                    ],
                    fused=True,
                    **optimizer_kwargs
                )
                print("✅ Using fused AdamW optimizer")
                return optimizer
            except (TypeError, RuntimeError):
                print("⚠️  Fused AdamW not available, falling back to standard AdamW")

        # Fallback to standard AdamW
        return AdamW(
            [
                {"params": param_groups["decay"], "weight_decay": self.config.weight_decay},
                {"params": param_groups["no_decay"], "weight_decay": 0.0},
            ],
            **optimizer_kwargs
        )

    def _create_scheduler(self, num_training_steps: int):
        from torch.optim.lr_scheduler import LambdaLR
        warm = self.config.warmup_steps
        base = self.config.learning_rate
        eta_min = self.config.min_lr
        def lr_lambda(step: int) -> float:
            if step < warm:
                return float(step) / float(max(1, warm))
            progress = float(step - warm) / float(max(1, num_training_steps - warm))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (eta_min / base) + (1 - eta_min / base) * cos
        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        attention_kwargs = prepare_attention_inputs(batch, self.config, self.device)

        # Model is already in target dtype (BF16/FP16/FP32), no autocast needed
        outputs = self.model(input_ids=input_ids, labels=labels, **attention_kwargs)
        loss = outputs["loss"]
        effective_tokens = batch["seq_lens"].sum().item()

        # Backward pass - gradients accumulate naturally
        loss.backward()

        return {"loss": float(loss.item()), "tokens": effective_tokens, "seq_lens": batch["seq_lens"].tolist()}

    def optimizer_step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, num_steps: int = 50) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_tokens = [], 0

        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            attention_kwargs = prepare_attention_inputs(batch, self.config, self.device)

            outputs = self.model(input_ids=input_ids, labels=labels, **attention_kwargs)
            loss = outputs["loss"]

            total_loss.append(loss.item())
            total_tokens += batch["seq_lens"].sum().item()

        return {
            "eval_loss": float(np.mean(total_loss)),
            "eval_ppl": float(np.exp(np.mean(total_loss))),
            "eval_tokens": int(total_tokens)
        }

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        print("\n" + "="*80)
        print("🚀 Starting Onyx 125M Training")
        print(f"   Device: {self.device}")
        print(f"   Precision: fp32")
        print(f"   Model Parameters: {self.model.get_num_params():,}")
        print(f"   Tokens per step: {self.config.tokens_per_step:,}")
        print(f"   Target tokens: {self.config.train_tokens_target:,}")
        print(f"   Max steps: {self.config.max_steps}")
        print(f"   Learning rate: {self.config.learning_rate:.2e} → {self.config.min_lr:.2e}")
        print(f"   Checkpoint dir: {self.checkpoint_dir}")
        print("="*80 + "\n")

        accumulated_loss = 0.0
        accumulated_tokens = 0
        accumulated_steps = 0
        last_log_time = time.time()
        step_start_time = time.time()

        total_steps = self.config.max_steps or (self.config.train_tokens_target // self.config.tokens_per_step)

        # Create scheduler once at the beginning with total steps
        if self.scheduler is None:
            self.scheduler = self._create_scheduler(total_steps)
            print(f"📅 Learning rate scheduler initialized for {total_steps} steps\n")

        pbar = tqdm(total=total_steps, initial=self.global_step, desc="Training", ncols=120)

        def signal_handler(signum, frame):
            print("\n⚠️  Interrupt received, saving checkpoint...")
            self.save_checkpoint("interrupt")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            for batch_idx, batch in enumerate(train_dataloader):
                if self.global_tokens >= self.config.train_tokens_target:
                    print("\n✅ Reached target token count!")
                    break
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    print("\n✅ Reached maximum steps!")
                    break

                # Train step
                try:
                    step_metrics = self.train_step(batch)
                    accumulated_loss += step_metrics["loss"]
                    accumulated_tokens += step_metrics["tokens"]
                    accumulated_steps += 1

                    # Step when token budget is met
                    if accumulated_tokens >= self.accum_token_cap:
                        step_end_time = time.time()
                        step_time = step_end_time - step_start_time
                        self.step_times.append(step_time)

                        grad_norm = self.optimizer_step()

                        self.global_step += 1
                        self.global_tokens += accumulated_tokens

                        avg_loss = accumulated_loss / max(1, accumulated_steps)
                        tokens_per_sec = accumulated_tokens / max(1e-6, (time.time() - last_log_time))
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate

                        # Update EMA for all metrics (provides stability)
                        if self.loss_ema is None:
                            self.loss_ema = avg_loss
                            self.grad_norm_ema = grad_norm
                            self.throughput_ema = tokens_per_sec
                        else:
                            self.loss_ema = self.ema_alpha * avg_loss + (1 - self.ema_alpha) * self.loss_ema
                            self.grad_norm_ema = self.ema_alpha * grad_norm + (1 - self.ema_alpha) * self.grad_norm_ema
                            self.throughput_ema = self.ema_alpha * tokens_per_sec + (1 - self.ema_alpha) * self.throughput_ema

                        self.loss_history.append(avg_loss)

                        # Calculate statistics
                        avg_step_time = np.mean(self.step_times[-10:]) if len(self.step_times) > 0 else step_time
                        remaining_steps = total_steps - self.global_step
                        eta_seconds = remaining_steps * avg_step_time
                        eta_mins = eta_seconds / 60

                        # Calculate loss trend
                        loss_trend = ""
                        if len(self.loss_history) >= 5:
                            recent_avg = np.mean(self.loss_history[-5:])
                            older_avg = np.mean(self.loss_history[-10:-5]) if len(self.loss_history) >= 10 else self.loss_history[0]
                            if recent_avg < older_avg:
                                loss_trend = "↓"
                            elif recent_avg > older_avg:
                                loss_trend = "↑"
                            else:
                                loss_trend = "→"

                        if self.global_step % self.config.log_every == 0:
                            # Print detailed step information
                            print("\n" + "─" * 80, flush=True)
                            print(f"📊 Step {self.global_step}/{total_steps} | "
                                  f"Tokens: {self.global_tokens:,}/{self.config.train_tokens_target:,}", flush=True)
                            print(f"   Loss:          {avg_loss:.6f} (EMA: {self.loss_ema:.6f}) {loss_trend}", flush=True)
                            print(f"   Perplexity:    {np.exp(avg_loss):.2f} (EMA: {np.exp(self.loss_ema):.2f})", flush=True)
                            print(f"   Learning Rate: {current_lr:.6e}", flush=True)
                            print(f"   Grad Norm:     {grad_norm:.4f} (EMA: {self.grad_norm_ema:.4f})", flush=True)
                            print(f"   Tokens/sec:    {tokens_per_sec:.1f} (EMA: {self.throughput_ema:.1f})", flush=True)
                            print(f"   Step Time:     {step_time:.2f}s (avg: {avg_step_time:.2f}s)", flush=True)
                            print(f"   Batch Tokens:  {accumulated_tokens:,} tokens in {accumulated_steps} micro-batches", flush=True)
                            print(f"   ETA:           {eta_mins:.1f} minutes ({remaining_steps} steps remaining)", flush=True)

                            # Show loss history sparkline (last 20 steps)
                            if len(self.loss_history) > 1:
                                recent_losses = self.loss_history[-min(20, len(self.loss_history)):]
                                min_loss = min(recent_losses)
                                max_loss = max(recent_losses)
                                loss_range = max_loss - min_loss if max_loss > min_loss else 1
                                sparkline = "".join(["▁▂▃▄▅▆▇█"[min(7, int((l - min_loss) / loss_range * 8))] for l in recent_losses])
                                print(f"   Loss Trend:    {sparkline} ({min_loss:.4f} - {max_loss:.4f})", flush=True)

                            print("─" * 80, flush=True)

                            log_dict = {
                                "loss": avg_loss,
                                "loss_ema": self.loss_ema,
                                "ppl": float(np.exp(avg_loss)),
                                "ppl_ema": float(np.exp(self.loss_ema)),
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                                "grad_norm_ema": self.grad_norm_ema,
                                "tokens/sec": tokens_per_sec,
                                "throughput_ema": self.throughput_ema,
                                "step_time": step_time,
                                "global_tokens": self.global_tokens,
                                "batch_tokens": accumulated_tokens,
                            }

                            pbar.set_postfix({
                                "loss": f"{avg_loss:.4f}",
                                "ema": f"{self.loss_ema:.4f}",
                                "ppl": f"{np.exp(avg_loss):.1f}",
                                "lr": f"{current_lr:.1e}",
                                "tok/s": f"{tokens_per_sec:.0f}",
                            })
                            pbar.update(1)

                            if self.use_wandb:
                                wandb.log(log_dict, step=self.global_step)

                        step_start_time = time.time()

                        accumulated_loss = 0.0
                        accumulated_tokens = 0
                        accumulated_steps = 0
                        last_log_time = time.time()

                        # Eval
                        if eval_dataloader and self.global_step % self.config.eval_every_steps == 0:
                            print("\n" + "="*80, flush=True)
                            print("🔍 Running Evaluation...", flush=True)
                            eval_metrics = self.evaluate(eval_dataloader)
                            print(f"   Eval Loss:       {eval_metrics['eval_loss']:.6f}", flush=True)
                            print(f"   Eval Perplexity: {eval_metrics['eval_ppl']:.2f}", flush=True)
                            print(f"   Eval Tokens:     {eval_metrics['eval_tokens']:,}", flush=True)
                            if eval_metrics["eval_loss"] < self.best_eval_loss:
                                improvement = self.best_eval_loss - eval_metrics["eval_loss"]
                                print(f"   🎉 New best! (improved by {improvement:.6f})", flush=True)
                                self.best_eval_loss = eval_metrics["eval_loss"]
                                self.save_checkpoint("best")
                            else:
                                print(f"   Best Loss:       {self.best_eval_loss:.6f}", flush=True)
                            print("="*80 + "\n", flush=True)
                            if self.use_wandb:
                                wandb.log(eval_metrics, step=self.global_step)

                        # Periodic checkpoint
                        if self.global_step % self.config.save_every_steps == 0:
                            print(f"\n💾 Saving checkpoint at step {self.global_step}...", flush=True)
                            self.save_checkpoint("step")
                            print(f"   ✅ Checkpoint saved\n", flush=True)

                except Exception as e:
                    print(f"⚠️  Error at step {self.global_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                if self.config.dry_run and self.global_step >= 10:
                    print("\n✅ Dry run completed successfully!")
                    break

        finally:
            pbar.close()
            self.save_checkpoint("final")

            # Print final summary
            total_time = sum(self.step_times)
            avg_time_per_step = np.mean(self.step_times) if self.step_times else 0
            final_loss = self.loss_history[-1] if self.loss_history else 0
            initial_loss = self.loss_history[0] if self.loss_history else 0
            loss_improvement = initial_loss - final_loss

            print("\n" + "="*80)
            print("✅ Training Completed!")
            print("="*80)
            print("\n📈 Training Statistics:")
            print(f"   Total Steps:        {self.global_step:,}")
            print(f"   Total Tokens:       {self.global_tokens:,}")
            print(f"   Total Time:         {total_time/3600:.2f} hours")
            print(f"   Avg Time/Step:      {avg_time_per_step:.2f}s")
            print(f"   Avg Throughput:     {self.global_tokens/max(1, total_time):.1f} tokens/sec")
            print(f"\n📊 Loss Metrics:")
            print(f"   Initial Loss:       {initial_loss:.6f}")
            print(f"   Final Loss:         {final_loss:.6f}")
            print(f"   Loss Improvement:   {loss_improvement:.6f}")
            print(f"   Best Eval Loss:     {self.best_eval_loss:.6f}")
            print(f"   Final Perplexity:   {np.exp(final_loss):.2f}")
            print(f"\n💾 Checkpoints:")
            print(f"   Directory:          {self.checkpoint_dir}")
            print(f"   Latest:             checkpoint_final_{self.global_step}.pt")
            print("="*80 + "\n")

            if self.use_wandb:
                try:
                    wandb.finish()
                except Exception:
                    pass

    def save_checkpoint(self, tag: str = "step"):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "global_step": self.global_step,
            "global_tokens": self.global_tokens,
            "best_eval_loss": self.best_eval_loss,
            "metrics": dict(self.metrics),
            # Save EMA values for smooth resumption
            "loss_ema": self.loss_ema,
            "grad_norm_ema": self.grad_norm_ema,
            "throughput_ema": self.throughput_ema,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        path = Path(self.checkpoint_dir) / f"checkpoint_{tag}_{self.global_step}.pt"
        torch.save(checkpoint, path)
        shutil.copy(path, Path(self.checkpoint_dir) / "checkpoint_latest.pt")
        with open(Path(self.checkpoint_dir) / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        print(f"💾 Saved checkpoint: {path}")
        return path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if both checkpoint has it and scheduler exists
        if "scheduler_state_dict" in checkpoint:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                warnings.warn("Checkpoint contains scheduler state but no scheduler initialized", stacklevel=2)

        self.global_step = checkpoint.get("global_step", 0)
        self.global_tokens = checkpoint.get("global_tokens", 0)
        self.best_eval_loss = checkpoint.get("best_eval_loss", float('inf'))
        self.metrics = defaultdict(list, checkpoint.get("metrics", {}))

        # Restore EMA values for smooth continuation
        self.loss_ema = checkpoint.get("loss_ema", None)
        self.grad_norm_ema = checkpoint.get("grad_norm_ema", None)
        self.throughput_ema = checkpoint.get("throughput_ema", None)

        print(f"✅ Resumed from checkpoint: step={self.global_step}, tokens={self.global_tokens:,}")
        if self.loss_ema is not None:
            print(f"   Loss EMA: {self.loss_ema:.6f}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Onyx 125M")

    # Data
    parser.add_argument("--data_glob", type=str,
                       default="/Users/owner/Desktop/caiatech/models/onyx-001/data/huggingface/cosmopedia_filtered/stories_filtered.jsonl")
    parser.add_argument("--eval_glob", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")

    # Training
    parser.add_argument("--tokens_per_step", type=int, default=32_000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--train_tokens_target", type=int, default=100_000_000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--fill_ratio", type=float, default=0.9)

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./checkpoints_125m")
    parser.add_argument("--save_every_steps", type=int, default=500)
    parser.add_argument("--eval_every_steps", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)

    # System
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    # Seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 Using device: {device}")

    # Tokenizer
    print(f"📚 Loading tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True, trust_remote_code=True)

    # Add special tokens BEFORE creating model
    special_tokens = {}
    if "<eod>" not in tokenizer.get_vocab():
        special_tokens["additional_special_tokens"] = tokenizer.additional_special_tokens + ["<eod>"]

    # Add proper pad token instead of reusing EOS
    if tokenizer.pad_token is None:
        if "<pad>" not in tokenizer.get_vocab():
            special_tokens["pad_token"] = "<pad>"
        else:
            tokenizer.pad_token = "<pad>"

    if special_tokens:
        num_added = tokenizer.add_special_tokens(special_tokens)
        print(f"   Added {num_added} special tokens to vocabulary")

    eod_id = tokenizer.convert_tokens_to_ids("<eod>")
    print(f"   Vocab size: {len(tokenizer)}, PAD: {tokenizer.pad_token_id}, EOS: {tokenizer.eos_token_id}, EOD: {eod_id}")

    # Model
    print("🔨 Creating Onyx 256M model (Option B)...")
    model_config = OnyxConfig(
        vocab_size=len(tokenizer),
        d_model=960,                # Option B: 80-dim heads
        n_layers=12,
        n_heads=12,
        n_kv_heads=12,              # Full attention (no GQA)
        d_ff=3840,                  # 4x d_model
        max_seq_len=2048,
        rope_theta=10000.0,         # Standard for 2k context
        pad_token_id=tokenizer.pad_token_id,
        eod_token_id=eod_id,
        block_cross_doc_attention=config.block_cross_doc_attention,
        gradient_checkpointing=config.gradient_checkpointing,
        use_torch_compile=False,
        num_attention_sinks=0,      # Disabled (unnecessary)
        windowed_attention=False    # Disabled (not needed at 2k)
    )
    # Determine model dtype based on config
    if config.bf16 and device.type == "cuda":
        model_dtype = torch.bfloat16
        print("   Using BF16 precision for training")
    elif config.fp16 and device.type == "cuda":
        model_dtype = torch.float16
        print("   Using FP16 precision for training")
    else:
        model_dtype = torch.float32
        print("   Using FP32 precision for training")

    model = Onyx125M(model_config).to(device=device, dtype=model_dtype)

    print(f"✅ Model created: {model.get_num_params():,} parameters")
    print(f"   d_model={model_config.d_model}, head_dim={model_config.d_model//model_config.n_heads}")
    print(f"   dtype={model_dtype}")

    # Validate data files
    print("🔍 Validating data files...")
    train_files = sorted(glob.glob(config.data_glob, recursive=True))
    if not train_files:
        raise ValueError(f"No training files found matching pattern: {config.data_glob}")
    print(f"   Found {len(train_files)} training file(s)")

    # Check if files are readable and have content
    total_lines = 0
    for fpath in train_files[:min(5, len(train_files))]:  # Check first 5 files
        try:
            with open(fpath, 'r') as f:
                line_count = sum(1 for _ in f)
                total_lines += line_count
        except Exception as e:
            raise ValueError(f"Cannot read training file {fpath}: {e}")
    print(f"   Sample files contain ~{total_lines:,} lines")

    # Dataloaders
    print("📊 Creating dataloaders...")
    train_dataloader = create_dataloader(config, tokenizer, is_eval=False)

    # Create eval dataloader if eval data provided
    if config.eval_glob:
        print("📊 Creating eval dataloader...")
        eval_dataloader = create_dataloader(config, tokenizer, is_eval=True)
    else:
        eval_dataloader = None
        print("⚠️  No eval data - use --eval_glob to enable validation")

    # Trainer
    trainer = Trainer(model, config, tokenizer, device)

    # Resume
    if config.resume:
        if config.resume == "auto":
            ckpt = Path(config.save_dir) / "checkpoint_latest.pt"
            if ckpt.exists():
                trainer.load_checkpoint(ckpt)
        else:
            trainer.load_checkpoint(config.resume)

    # Train
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
