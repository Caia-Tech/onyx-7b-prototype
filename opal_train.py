# train_32x32_ar.py
# Training loop for 32x32 pixel-AR on CIFAR-10. Saves checkpoints and sample grids.

import argparse, os, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from model_32x32_ar import (
    ARConfig, PixelARBackbone, PixelCategoricalHead, pixel_ce_loss,
    imgs_to_seq_uint8, VOCAB, SOS_ID, PIXELS, sample_images, EMA
)

def bpd_from_ce(loss_ce: float) -> float:
    # CE (nats/token over pixels) -> bits per pixel-dimension
    # loss_ce is CE over all tokens predicted (T=PIXELS) averaged per token.
    # 1 token == 1 channel-dimension
    return (loss_ce / math.log(2.0))

def seed_everything(seed: int = 1337):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=18)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accum", type=int, default=1, help="grad accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="./runs/pxar32")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    seed_everything(args.seed)
    out = Path(args.out_dir)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    # ---- Data: CIFAR-10 (uint8, flip only) ----
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # float32 in [0,1]
    ])

    train_set = torchvision.datasets.CIFAR10(root=str(out / "data"), train=True, download=True, transform=tfm)
    val_set   = torchvision.datasets.CIFAR10(root=str(out / "data"), train=False, download=True, transform=tfm)

    def to_uint8(batch):
        # Convert float[0,1] -> uint8[0,255]
        return (batch * 255.0).round().to(torch.uint8)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- Model ----
    cfg = ARConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        vocab_size=VOCAB,
        max_len=PIXELS + 1
    )
    backbone = PixelARBackbone(cfg).to(args.device)
    head = PixelCategoricalHead(backbone.d_model).to(args.device)

    # Optional BF16
    amp_dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else torch.float32
    print(f"Using dtype={amp_dtype}")

    # ---- Optimizer / EMA / Scheduler ----
    params = list(backbone.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    ema_backbone = EMA(backbone, decay=args.ema_decay)
    ema_head = EMA(head, decay=args.ema_decay)

    def lr_schedule(step):
        # warmup + cosine
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        p = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        p = min(max(p, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    # ---- Training ----
    global_step = 0
    best_val = float("inf")

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16))  # likely off for bf16

    while global_step < args.max_steps:
        backbone.train()
        head.train()

        for imgs, _ in train_loader:
            global_step += 1
            lr_scale = lr_schedule(global_step)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_scale

            imgs = to_uint8(imgs).to(args.device, non_blocking=True)  # (B,3,32,32) uint8
            seq = imgs_to_seq_uint8(imgs).long()                      # (B,3072) long in [0..255]

            # prepend SOS
            sos = torch.full((seq.size(0), 1), SOS_ID, dtype=torch.long, device=seq.device)
            full = torch.cat([sos, seq], dim=1)                       # (B,3073)

            inp = full[:, :-1]   # (B,3072) [SOS, x0..x3070]
            tgt = full[:, 1:]    # (B,3072) [x0..x3071]  predict only pixel ids

            with torch.autocast(device_type=args.device.split(":")[0], dtype=amp_dtype, enabled=(amp_dtype!=torch.float32)):
                h = backbone(inp)          # (B,T,d)
                logits = head(h)           # (B,T,256)
                loss = pixel_ce_loss(logits, tgt)

            (loss / args.accum).backward()
            if global_step % args.accum == 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema_backbone.update(backbone)
                ema_head.update(head)

            if global_step % 100 == 0:
                bpd = bpd_from_ce(loss.item())
                print(f"step {global_step:6d} | lr {opt.param_groups[0]['lr']:.2e} | loss {loss.item():.4f} | bpd {bpd:.4f}")

            # samples
            if global_step % args.sample_every == 0:
                # use EMA for cleaner samples
                backbone_ema = PixelARBackbone(cfg).to(args.device)
                head_ema = PixelCategoricalHead(backbone.d_model).to(args.device)
                ema_backbone.copy_to(backbone_ema)
                ema_head.copy_to(head_ema)

                with torch.no_grad():
                    imgs_gen = sample_images(backbone_ema, head_ema, num_images=args.samples, temperature=1.0, top_p=1.0, device=args.device)
                grid = make_grid(imgs_gen.float() / 255.0, nrow=int(math.sqrt(args.samples)))
                save_image(grid, os.path.join(args.out_dir, "samples", f"sample_{global_step:06d}.png"))

                del backbone_ema, head_ema
                torch.cuda.empty_cache()

            # save
            if global_step % args.save_every == 0:
                ckpt = {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "opt": opt.state_dict(),
                    "step": global_step,
                    "cfg": vars(cfg),
                }
                torch.save(ckpt, os.path.join(args.out_dir, "ckpt", f"model_{global_step:06d}.pt"))

            if global_step >= args.max_steps:
                break

        # quick val pass (optional; CIFAR test set)
        backbone.eval()
        head.eval()
        val_loss_sum, val_tok = 0.0, 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = (imgs * 255.0).round().to(torch.uint8).to(args.device, non_blocking=True)
                seq = imgs_to_seq_uint8(imgs).long()
                sos = torch.full((seq.size(0), 1), SOS_ID, dtype=torch.long, device=seq.device)
                full = torch.cat([sos, seq], dim=1)
                inp = full[:, :-1]
                tgt = full[:, 1:]
                h = backbone(inp)
                logits = head(h)
                loss = pixel_ce_loss(logits, tgt)
                val_loss_sum += loss.item() * inp.numel() / inp.size(0)
                val_tok += inp.numel()

        val_loss = val_loss_sum / max(1, val_tok)
        val_bpd = bpd_from_ce(val_loss)
        print(f"[val] step {global_step:6d} | bpd {val_bpd:.4f}")

        if val_bpd < best_val:
            best_val = val_bpd
            torch.save({
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "step": global_step,
                "best_val_bpd": best_val,
            }, os.path.join(args.out_dir, "ckpt", "best.pt"))

if __name__ == "__main__":
    main()
