#!/usr/bin/env python3
"""
Production-grade training script for Onyx 7B with packed sequences, length curriculum,
and robust checkpointing/resume.

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
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Import our model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_onyx7b_marvin_tutt_caia_tech import Onyx7B, OnyxConfig, build_block_causal_mask

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("W&B not available for logging", stacklevel=2)


# ============================================================================
# Helpers
# ============================================================================

def stage_to_len(stage: str) -> int:
    return {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384}[stage]


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Data
    data_glob: str = "/data/**/*.jsonl"
    tokenizer: str = "meta-llama/Llama-3.2-1B"  # Or path
    eval_glob: Optional[str] = None

    # Training
    tokens_per_step: int = 2_000_000
    max_steps: Optional[int] = None
    train_tokens_target: Optional[int] = 1_000_000_000_000  # 1T tokens

    # Curriculum
    max_stage: str = "16k"  # {1k, 2k, 4k, 8k, 16k}
    stage_schedule: Dict[str, int] = None  # Tokens per stage
    stage_mix: Dict[str, float] = None  # Length distribution
    fill_ratio: float = 0.9
    block_cross_doc_attention: bool = False

    # Model
    compile_model: bool = True
    gradient_checkpointing: bool = False

    # Optimization
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps_per_stage: int = 2000
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.97
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    save_dir: str = "./checkpoints"
    save_every_steps: int = 5000
    eval_every_steps: int = 1000
    resume: Optional[str] = None  # "auto" or path

    # Logging
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    log_every: int = 10

    # System
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 1337
    dry_run: bool = False

    def __post_init__(self):
        # Default stage schedule (in tokens)
        if self.stage_schedule is None:
            self.stage_schedule = {
                "1k": 120_000_000_000,
                "2k": 140_000_000_000,
                "4k": 180_000_000_000,
                "8k": 230_000_000_000,
                "16k": 330_000_000_000,
            }

        # Default stage mix
        if self.stage_mix is None:
            self.stage_mix = {"cur": 0.7, "mid": 0.2, "short": 0.1}


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
        max_seq_len: int,
        fill_ratio: float = 0.9,
        eod_token_id: int = 3,
        pad_token_id: int = 0,
        seed: int = 42,
        stage_mix: Dict[str, float] = None,
        current_stage_max: int = 16384
    ):
        self.file_pattern = file_pattern
        self.tokenizer = tokenizer
        self.max_seq_len = min(max_seq_len, current_stage_max)
        self.fill_ratio = fill_ratio
        self.eod_token_id = eod_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed
        self.stage_mix = stage_mix or {"cur": 1.0}
        self.current_stage_max = current_stage_max

        # Files
        self.files = sorted(glob.glob(file_pattern, recursive=True))
        if not self.files:
            raise ValueError(f"No files found matching {file_pattern}")

        # Stats
        self.stats = defaultdict(int)

    def _get_target_length(self) -> int:
        """Sample target length based on stage mix"""
        r = random.random()
        if self.current_stage_max <= 1024:
            return 1024
        elif self.current_stage_max <= 2048:
            return 2048 if r < self.stage_mix.get("cur", 0.7) else 1024
        elif self.current_stage_max <= 4096:
            if r < self.stage_mix.get("cur", 0.7):
                return 4096
            elif r < self.stage_mix.get("cur", 0.7) + self.stage_mix.get("mid", 0.2):
                return 2048
            else:
                return 1024
        elif self.current_stage_max <= 8192:
            if r < self.stage_mix.get("cur", 0.6):
                return 8192
            elif r < self.stage_mix.get("cur", 0.6) + self.stage_mix.get("mid", 0.25):
                return 4096
            else:
                return random.choice([1024, 2048])
        else:
            if r < self.stage_mix.get("cur", 0.6):
                return 16384
            elif r < self.stage_mix.get("cur", 0.6) + self.stage_mix.get("mid", 0.25):
                return 8192
            else:
                return random.choice([1024, 2048, 4096])

    def _pack_documents(self, doc_iterator, target_len: int) -> Optional[PackedDocument]:
        """Pack multiple documents into a single sequence (no global padding)."""
        packed_ids, doc_spans = [], []
        current_pos = 0
        target_fill = int(target_len * self.fill_ratio)

        for doc in doc_iterator:
            try:
                text = doc if isinstance(doc, str) else (doc.get("text", "") or doc.get("content", "")) if isinstance(doc, dict) else ""
                if not text.strip():
                    continue

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
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

        if not packed_ids:
            return None

        return PackedDocument(packed_ids, doc_spans, len(packed_ids))

    def _read_jsonl_file(self, filepath: str):
        """Read and yield documents from JSONL file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    def __iter__(self):
        """Iterate over packed sequences"""
        files = list(self.files)
        random.Random(self.seed + int(time.time())).shuffle(files)

        def doc_stream():
            for filepath in files:
                yield from self._read_jsonl_file(filepath)

        doc_iter = doc_stream()

        while True:
            target_len = self._get_target_length()
            packed = self._pack_documents(doc_iter, target_len)
            if packed is None:
                doc_iter = doc_stream()
                continue

            # Labels (shift by 1, -100 on last token; padding handled in collate_fn)
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
    is_eval: bool = False,
    current_stage_max: int = 16384
) -> DataLoader:
    """Create dataloader for training or evaluation"""

    file_pattern = config.eval_glob if is_eval else config.data_glob
    if not file_pattern:
        return None

    max_len = stage_to_len(config.max_stage)

    dataset = StreamingPackedDataset(
        file_pattern=file_pattern,
        tokenizer=tokenizer,
        max_seq_len=max_len,
        fill_ratio=config.fill_ratio,
        eod_token_id=tokenizer.convert_tokens_to_ids("<eod>") if "<eod>" in tokenizer.get_vocab() else 3,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        seed=config.seed,
        stage_mix=config.stage_mix,
        current_stage_max=max_len
    )

    # Pad to LOCAL max per batch; mask pads in attention.
    def collate_fn(batch):
        seqs = [x["input_ids"] for x in batch]
        labs = [x["labels"] for x in batch]
        seq_lens = torch.tensor([x["seq_len"] for x in batch], dtype=torch.long)
        doc_spans = [x["doc_spans"] for x in batch]

        S = max(int(t.size(0)) for t in seqs)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        def pad_to(t: torch.Tensor, length: int, pad_value: int):
            if t.size(0) == length:
                return t
            out = torch.full((length,), pad_value, dtype=t.dtype)
            out[: t.size(0)] = t
            return out

        input_ids = torch.stack([pad_to(t, S, pad_id) for t in seqs])      # (B,S)
        labels = torch.stack([pad_to(t, S, -100) for t in labs])           # (B,S)

        return {"input_ids": input_ids, "labels": labels, "seq_lens": seq_lens, "doc_spans": doc_spans}

    return DataLoader(
        dataset,
        batch_size=1,  # token-budgeted accumulation; leave at 1 to keep masks small
        collate_fn=collate_fn,
        num_workers=0 if is_eval else config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(config.num_workers > 0 and not is_eval),
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
    """Always build a (B,S,S) block-causal mask; SDPA path in the model consumes it."""
    seq_lens = batch["seq_lens"].to(device)
    doc_spans = batch.get("doc_spans")

    attn_mask = build_block_causal_mask(
        seq_lens=seq_lens,
        doc_spans=doc_spans,
        block_cross_doc=config.block_cross_doc_attention,
        device=device
    )
    return {"seq_lens": seq_lens, "attn_mask": attn_mask, "doc_spans": doc_spans}


class Trainer:
    """Main training orchestrator"""

    def __init__(self, model: Onyx7B, config: TrainingConfig, tokenizer, device: torch.device):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Mixed precision
        self.use_amp = config.bf16 or config.fp16
        self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
        self.scaler = GradScaler() if config.fp16 else None

        # Scheduler (set per stage)
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.global_tokens = 0
        self.current_stage = "1k"
        self.stage_tokens = 0
        self.stage_step = 0

        # Accumulation estimate per stage (set in _update_stage)
        self.gradient_accumulation_steps = max(1, self.config.tokens_per_step // stage_to_len(self.current_stage))

        # Metrics
        self.metrics = defaultdict(list)

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
        return AdamW(
            [
                {"params": param_groups["decay"], "weight_decay": self.config.weight_decay},
                {"params": param_groups["no_decay"], "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps
        )

    def _create_scheduler(self, num_training_steps: int):
        from torch.optim.lr_scheduler import LambdaLR
        warm = self.config.warmup_steps_per_stage

        def lr_lambda(step: int) -> float:
            if step < warm:
                return float(step) / float(max(1, warm))
            progress = float(step - warm) / float(max(1, num_training_steps - warm))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _get_stage_max_len(self, stage: str) -> int:
        return stage_to_len(stage)

    def _update_stage(self):
        """Check/update current stage based on global_tokens; update scheduler + grad_accum."""
        stages = ["1k", "2k", "4k", "8k", "16k"]
        cumulative_tokens = 0
        for stage in stages:
            if stage not in self.config.stage_schedule:
                continue
            stage_budget = self.config.stage_schedule[stage]
            cumulative_tokens += stage_budget
            if self.global_tokens < cumulative_tokens:
                if stage != self.current_stage:
                    print(f"\n📈 Advancing to stage: {stage} (max_len={self._get_stage_max_len(stage)})")
                    self.current_stage = stage
                    self.stage_tokens = 0
                    self.stage_step = 0

                    if stage in ["8k", "16k"]:
                        self.model.config.gradient_checkpointing = True

                    remaining_tokens = cumulative_tokens - self.global_tokens
                    est_steps = max(1, remaining_tokens // self.config.tokens_per_step)
                    self.scheduler = self._create_scheduler(est_steps)

                    # Update accumulation estimate for this stage
                    self.gradient_accumulation_steps = max(1, self.config.tokens_per_step // self._get_stage_max_len(stage))
                break

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        attention_kwargs = prepare_attention_inputs(batch, self.config, self.device)

        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(input_ids=input_ids, labels=labels, **attention_kwargs)
            loss = outputs["loss"]

        # Scale loss by estimated grad accumulation steps for this stage
        if hasattr(self, 'gradient_accumulation_steps'):
            loss = loss / max(1, self.gradient_accumulation_steps)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        effective_tokens = batch["seq_lens"].sum().item()
        return {"loss": loss.item(), "tokens": effective_tokens, "seq_lens": batch["seq_lens"].tolist()}

    def optimizer_step(self):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, num_steps: int = 100) -> Dict[str, float]:
        self.model.eval()
        losses_by_bucket = defaultdict(list)
        total_loss, total_tokens = [], 0

        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            attention_kwargs = prepare_attention_inputs(batch, self.config, self.device)

            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(input_ids=input_ids, labels=labels, **attention_kwargs)
                loss = outputs["loss"]

            L = int(batch["seq_lens"][0].item())
            bucket = "≤1k" if L <= 1024 else "1-2k" if L <= 2048 else "2-4k" if L <= 4096 else "4-8k" if L <= 8192 else "8-16k"

            losses_by_bucket[bucket].append(loss.item())
            total_loss.append(loss.item())
            total_tokens += batch["seq_lens"].sum().item()

        results = {"eval_loss": float(np.mean(total_loss)), "eval_ppl": float(np.exp(np.mean(total_loss))), "eval_tokens": int(total_tokens)}
        for bucket, losses in losses_by_bucket.items():
            results[f"eval_ppl_{bucket}"] = float(np.exp(np.mean(losses)))
        return results

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        print("\n" + "="*60)
        print("🚀 Starting Onyx 7B Training")
        print(f"   Device: {self.device}")
        print(f"   Precision: {'bf16' if self.config.bf16 else 'fp16' if self.config.fp16 else 'fp32'}")
        print(f"   Tokens per step: {self.config.tokens_per_step:,}")
        print(f"   Target tokens: {self.config.train_tokens_target:,}")
        print("="*60 + "\n")

        accumulated_loss = 0.0
        accumulated_tokens = 0
        accumulated_steps = 0
        last_log_time = time.time()

        total_steps = self.config.max_steps or (self.config.train_tokens_target // self.config.tokens_per_step)
        pbar = tqdm(total=total_steps, initial=self.global_step, desc="Training")

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

                # Stage updates + sync dataloader stage cap
                self._update_stage()
                stage_len = self._get_stage_max_len(self.current_stage)
                if hasattr(train_dataloader, "dataset"):
                    train_dataloader.dataset.current_stage_max = stage_len
                    train_dataloader.dataset.max_seq_len = stage_len

                # Train step
                try:
                    step_metrics = self.train_step(batch)
                    accumulated_loss += step_metrics["loss"]
                    accumulated_tokens += step_metrics["tokens"]
                    accumulated_steps += 1

                    # If enough tokens accumulated or enough microbatches (fallback), step optimizer
                    if (accumulated_tokens >= self.config.tokens_per_step) or (accumulated_steps >= self.gradient_accumulation_steps):
                        grad_norm = self.optimizer_step()

                        self.global_step += 1
                        self.global_tokens += accumulated_tokens
                        self.stage_tokens += accumulated_tokens
                        self.stage_step += 1

                        avg_loss = accumulated_loss / max(1, accumulated_steps)
                        tokens_per_sec = accumulated_tokens / max(1e-6, (time.time() - last_log_time))
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate

                        if self.global_step % self.config.log_every == 0:
                            log_dict = {
                                "loss": avg_loss,
                                "ppl": float(np.exp(avg_loss)),
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                                "tokens/sec": tokens_per_sec,
                                "stage": self.current_stage,
                                "global_tokens": self.global_tokens,
                            }
                            pbar.set_postfix({
                                "loss": f"{avg_loss:.4f}",
                                "ppl": f"{np.exp(avg_loss):.2f}",
                                "lr": f"{current_lr:.2e}",
                                "tok/s": f"{tokens_per_sec:.0f}",
                                "stage": self.current_stage
                            })
                            pbar.update(1)
                            if self.use_wandb:
                                wandb.log(log_dict, step=self.global_step)

                        accumulated_loss = 0.0
                        accumulated_tokens = 0
                        accumulated_steps = 0
                        last_log_time = time.time()

                        # Eval
                        if eval_dataloader and self.global_step % self.config.eval_every_steps == 0:
                            eval_metrics = self.evaluate(eval_dataloader)
                            print(f"\n📊 Eval @ step {self.global_step}: loss {eval_metrics['eval_loss']:.4f} | ppl {eval_metrics['eval_ppl']:.2f}")
                            for b in ["≤1k", "1-2k", "2-4k", "4-8k", "8-16k"]:
                                k = f"eval_ppl_{b}"
                                if k in eval_metrics:
                                    print(f"   PPL {b}: {eval_metrics[k]:.2f}")
                            if self.use_wandb:
                                wandb.log(eval_metrics, step=self.global_step)
                            if eval_metrics["eval_loss"] < self.best_eval_loss:
                                self.best_eval_loss = eval_metrics["eval_loss"]
                                self.save_checkpoint("best")

                        # Periodic checkpoint
                        if self.global_step % self.config.save_every_steps == 0:
                            self.save_checkpoint("step")

                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️  OOM at step {self.global_step}, skipping batch")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                except Exception as e:
                    print(f"⚠️  Error at step {self.global_step}: {e}")
                    continue

                if self.config.dry_run and self.global_step >= 100:
                    print("\n✅ Dry run completed successfully!")
                    break

        finally:
            pbar.close()
            self.save_checkpoint("final")
            if self.use_wandb:
                wandb.finish()
            print("\n" + "="*60)
            print("✅ Training completed!")
            print(f"   Total steps: {self.global_step}")
            print(f"   Total tokens: {self.global_tokens:,}")
            print("="*60)

    # ---- checkpoint I/O ----
    def save_checkpoint(self, tag: str = "step"):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": asdict(self.config),
            "global_step": self.global_step,
            "global_tokens": self.global_tokens,
            "current_stage": self.current_stage,
            "stage_tokens": self.stage_tokens,
            "best_eval_loss": self.best_eval_loss,
            "metrics": dict(self.metrics)
        }
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
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.global_tokens = checkpoint.get("global_tokens", 0)
        self.current_stage = checkpoint.get("current_stage", "1k")
        self.stage_tokens = checkpoint.get("stage_tokens", 0)
        self.best_eval_loss = checkpoint.get("best_eval_loss", float('inf'))
        self.metrics = defaultdict(list, checkpoint.get("metrics", {}))
        print(f"✅ Resumed from checkpoint: step={self.global_step}, tokens={self.global_tokens:,}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Onyx 7B with packed sequences")

    # Data
    parser.add_argument("--data_glob", type=str, default="/data/**/*.jsonl")
    parser.add_argument("--eval_glob", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B")

    # Training
    parser.add_argument("--tokens_per_step", type=int, default=2_000_000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_tokens_target", type=int, default=1_000_000_000_000)

    # Curriculum
    parser.add_argument("--max_stage", type=str, default="16k", choices=["1k", "2k", "4k", "8k", "16k"])
    parser.add_argument("--fill_ratio", type=float, default=0.9)
    parser.add_argument("--block_cross_doc_attention", action="store_true")

    # Model
    parser.add_argument("--compile_model", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps_per_stage", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta2", type=float, default=0.97)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Precision
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every_steps", type=int, default=5000)
    parser.add_argument("--eval_every_steps", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)

    # System
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    # Seeds / CUDA
    random.seed(config.seed); np.random.seed(config.seed)
    torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")

    # Tokenizer
    print(f"📚 Loading tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True, trust_remote_code=True)

    # Ensure <eod> exists; ensure pad token
    if "<eod>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<eod>"]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")

    # Model
    print("🔨 Creating Onyx 7B model...")
    model_config = OnyxConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eod_token_id=eod_id,
        block_cross_doc_attention=config.block_cross_doc_attention,
        gradient_checkpointing=config.gradient_checkpointing,
        use_torch_compile=config.compile_model
    )
    model = Onyx7B(model_config).to(device=device, dtype=(torch.bfloat16 if config.bf16 else torch.float16))

    if config.compile_model:
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune", dynamic=True)

    print(f"✅ Model created: {model.get_num_params():,} parameters")

    # Dataloaders
    print("📊 Creating dataloaders...")
    train_dataloader = create_dataloader(config, tokenizer, is_eval=False, current_stage_max=stage_to_len(config.max_stage))
    eval_dataloader = create_dataloader(config, tokenizer, is_eval=True, current_stage_max=stage_to_len(config.max_stage)) if config.eval_glob else None

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