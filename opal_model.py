# model_32x32_ar.py
# Minimal 32x32 pixel-autoregressive Transformer with 2D pos embeddings.
# Exact 8-bit fidelity via 256-way categorical head.

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_H = 32
IMAGE_W = 32
IMAGE_C = 3
PIXELS = IMAGE_H * IMAGE_W * IMAGE_C  # 3072
PIXEL_VOCAB = 256                     # 0..255 exact 8-bit
SOS_ID = 256                          # special start token (outside 0..255)
VOCAB = PIXEL_VOCAB + 1               # 257 (0..255 + SOS)

# ---------- Utils: seq<->image ----------
def imgs_to_seq_uint8(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: (B, 3, 32, 32) uint8
    returns: (B, 3072) uint8, raster-major with [R,G,B] per pixel
    """
    assert imgs.dtype == torch.uint8
    b = imgs.size(0)
    seq = imgs.permute(0, 2, 3, 1).contiguous().view(b, PIXELS)
    return seq

def seq_to_imgs_uint8(seq: torch.Tensor) -> torch.Tensor:
    """
    seq: (B, 3072) uint8
    returns: (B, 3, 32, 32) uint8
    """
    b = seq.size(0)
    imgs = seq.view(b, IMAGE_H, IMAGE_W, IMAGE_C).permute(0, 3, 1, 2).contiguous()
    return imgs

# ---------- Positional Embeddings (2D row/col + channel) ----------
class Image2DPositional(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.row = nn.Embedding(IMAGE_H, d_model)
        self.col = nn.Embedding(IMAGE_W, d_model)
        self.chn = nn.Embedding(IMAGE_C, d_model)

    def forward(self, T: int, device=None) -> torch.Tensor:
        """
        Returns pos embedding for a sequence of length T where:
          t=0 is SOS, t>=1 are image tokens corresponding to positions p=t-1.
        Output: (T, d_model)
        """
        d = self.row.embedding_dim
        pe = torch.zeros(T, d, device=device)

        if T == 0:
            return pe

        t = torch.arange(T, device=device)
        p = (t - 1).clamp_min(0)  # p=0 for SOS

        # derive row/col/channel for t>=1 (p in [0..3071])
        rows = (p // (IMAGE_W * IMAGE_C)).clamp_max(IMAGE_H - 1)
        cols = ((p // IMAGE_C) % IMAGE_W).clamp_max(IMAGE_W - 1)
        chns = (p % IMAGE_C).clamp_max(IMAGE_C - 1)

        pe = self.row(rows) + self.col(cols) + self.chn(chns)
        # make SOS position unique by zeroing it (or you could add a learned SOS pos)
        pe[0, :] = 0.0
        return pe

# ---------- Transformer blocks ----------
class MLP(nn.Module):
    def __init__(self, d_model: int, expansion: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(d_model * expansion)
        self.fc1 = nn.Linear(d_model, inner)
        self.fc2 = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, nH, dH)
        q = q.transpose(1, 2)        # (B, nH, T, dH)
        k = k.transpose(1, 2)        # (B, nH, T, dH)
        v = v.transpose(1, 2)        # (B, nH, T, dH)

        # scaled dot-product attention with causal masking
        # torch >=2.0: F.scaled_dot_product_attention supports is_causal=True
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(attn)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_expansion: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_expansion, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# ---------- Backbone ----------
@dataclass
class ARConfig:
    d_model: int = 512
    n_layers: int = 18
    n_heads: int = 8
    dropout: float = 0.0
    mlp_expansion: float = 4.0
    vocab_size: int = VOCAB  # 257 (0..255 + SOS)
    max_len: int = PIXELS + 1  # 3073 with SOS

class PixelARBackbone(nn.Module):
    def __init__(self, config: ARConfig):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.pos2d = Image2DPositional(config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.mlp_expansion, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)

    @property
    def d_model(self):
        return self.config.d_model

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        """
        toks: (B, T) Long (values in [0..256]), where 256 is SOS.
        returns: hidden states (B, T, d_model)
        """
        B, T = toks.shape
        if T > self.config.max_len:
            raise ValueError(f"Sequence too long: {T} > {self.config.max_len}")

        tok_emb = self.token(toks)                  # (B,T,d)
        pos_emb = self.pos2d(T, device=toks.device) # (T,d)
        x = tok_emb + pos_emb.unsqueeze(0)

        for blk in self.layers:
            x = blk(x)
        x = self.ln_f(x)
        return x

# ---------- Heads ----------
class PixelCategoricalHead(nn.Module):
    """256-way categorical over pixel values."""
    def __init__(self, d_model: int, out_classes: int = PIXEL_VOCAB):
        super().__init__()
        self.proj = nn.Linear(d_model, out_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # (B, T, 256)

def pixel_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, 256)
    targets: (B, T) [0..255], Long
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B*T, V),
        targets.reshape(B*T),
        reduction="mean",
    )
    return loss

# (Optional) DMOL head placeholder to swap in later
class PixelDMOLHead(nn.Module):
    """
    Placeholder for a discretized mixture-of-logistics head.
    Implement later if desired. Keep interface identical to PixelCategoricalHead.
    """
    def __init__(self, d_model: int, K: int = 10):
        super().__init__()
        self.K = K
        # For independent channels (simpler than PixelCNN++ coupling):
        # For each channel: mixture weights pi, means mu, log_scales s
        # Params per channel per mixture: 3
        # Total per position: 3 * K * C
        self.proj = nn.Linear(d_model, IMAGE_C * K * 3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # interpret in custom loss

# ---------- EMA ----------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)

# ---------- Sampling ----------
@torch.no_grad()
def sample_images(
    model: PixelARBackbone,
    head: nn.Module,
    num_images: int = 16,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Returns (N, 3, 32, 32) uint8
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    T = PIXELS + 1
    sos = torch.full((num_images, 1), SOS_ID, dtype=torch.long, device=device)
    seq = sos  # (N, 1)

    for step in range(PIXELS):
        h = model(seq)                             # (N, t, d)
        logits = head(h)[:, -1, :] / max(temperature, 1e-6)  # (N, 256)
        probs = F.softmax(logits, dim=-1)

        if top_p < 1.0:
            # nucleus per row
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf <= top_p
            mask[:, 0] = True
            keep = torch.zeros_like(probs, dtype=torch.bool)
            keep.scatter_(1, sorted_idx, mask)
            probs = torch.where(keep, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        nxt = torch.multinomial(probs, num_samples=1)  # (N,1) in [0..255]
        seq = torch.cat([seq, nxt], dim=1)

    seq_pixels = seq[:, 1:].to(torch.uint8)           # drop SOS
    imgs = seq_to_imgs_uint8(seq_pixels)              # (N,3,32,32)
    return imgs

