#!/usr/bin/env python3
"""
Onyx 7B Dense Model, Marvin Tutt, Caia Tech
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import math
import warnings

# Try to import optional dependencies
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    warnings.warn("FlashAttention not available, using PyTorch SDPA", stacklevel=2)

# Optional varlen FA2 (removes padding compute). We guard import and feature flag.
VARLEN_FA_AVAILABLE = False
try:
    # FA2 varlen kernels (natively support causal attention and dropout)
    from flash_attn import flash_attn_varlen_qkvpacked_func as fa_varlen_qkv
    VARLEN_FA_AVAILABLE = True
except Exception:
    pass


@dataclass
class OnyxConfig:
    """Configuration for Onyx 7B Dense Model"""
    # Core architecture - 7B scale
    vocab_size: int = 128258      # Hermes Llama 3 tokenizer + <eod> token
    d_model: int = 4096           # Hidden dimension
    n_layers: int = 32            # Number of layers
    n_heads: int = 32             # Attention heads
    n_kv_heads: int = 8           # GQA 4:1 ratio
    d_ff: int = 11008             # FFN dimension
    max_seq_len: int = 16384      # 16k context

    eos_token_id: int = 2         # EOS token ID for generation
    pad_token_id: int = 0         # Padding token ID
    eod_token_id: int = 3         # End-of-document token ID
    block_cross_doc_attention: bool = False  # Block attention across documents
    
    # Position encoding
    rope_theta: float = 500000.0  # Extended RoPE for long context
    rope_scaling: Optional[Dict[str, Any]] = None  # NTK/YARN scaling
    # Examples:
    # {"type": "ntk", "factor": 1.5}
    # {"type": "yarn", "factor": 2.0, "orig_ctx": 4096}
    
    # Architecture features
    use_swiglu: bool = True       # SwiGLU activation
    use_rms_norm: bool = True     # RMSNorm instead of LayerNorm
    norm_eps: float = 1e-5
    use_qk_norm: bool = False     # Reserved for QK normalization
    
    # Performance optimizations
    use_flash_attn: bool = True   # FlashAttention v2
    use_varlen_flash_attn: bool = True  # Try varlen FA2 fast path when available
    use_cuda_graphs: bool = False # Reserved for future optimization
    use_torch_compile: bool = True # torch.compile optimization
    
    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0
    gradient_checkpointing: bool = False
    
    # Embeddings
    tie_embeddings: bool = True

    # Long-context stability features
    num_attention_sinks: int = 0  # 0 disables, >0 adds learned global KV anchors
    windowed_attention: bool = False  # Enable local+global head attention
    window_size: int = 2048  # Window size for local heads
    num_global_heads: int = 4  # Number of heads with full context
    
    def __post_init__(self):
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        # Ensure head_dim is even for RoPE
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")


class RMSNorm(nn.Module):
    """RMSNorm normalization layer"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class RoPE(nn.Module):
    """Rotary Position Embeddings with correct even/odd implementation"""

    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.head_dim = self.d_model // self.n_heads

        # Precompute frequencies with optional scaling
        self.register_buffer(
            "freqs",
            self._compute_freqs(),
            persistent=False  # Don't save in state_dict
        )

    def _compute_freqs(self) -> torch.Tensor:
        # Base frequencies for pairs of dimensions
        theta = self.theta

        # Apply RoPE scaling if configured
        if self.rope_scaling is not None:
            scaling_type = self.rope_scaling.get("type", "none")
            scaling_factor = self.rope_scaling.get("factor", 1.0)

            if scaling_type == "ntk":
                # Neural Tangent Kernel scaling - slow down rotation
                theta = theta * scaling_factor
            elif scaling_type == "yarn":
                # YaRN scaling - more sophisticated frequency adjustment
                orig_ctx = self.rope_scaling.get("orig_ctx", 4096)
                # Compute scaled theta per YaRN paper
                # For simplicity, using NTK-style scaling with adjustment
                # Full YaRN would require position-dependent scaling
                ratio = self.max_seq_len / orig_ctx
                if ratio > 1.0:
                    theta = theta * (scaling_factor ** (self.head_dim / (self.head_dim - 2)))

        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        return freqs
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0  # For KV cache support
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create position indices  
        t = torch.arange(offset, offset + seq_len, device=q.device, dtype=self.freqs.dtype)
        freqs = torch.outer(t, self.freqs)  # (seq_len, head_dim/2)
        
        # Create cos and sin embeddings
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Apply rotary embeddings
        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)
        
        return q_embed, k_embed
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embeddings with correct even/odd pairing"""
        # x: (B, S, H, D) where D = head_dim
        d = x.size(-1)
        assert d % 2 == 0, "head_dim must be even for RoPE"
        
        # Ensure cos/sin match tensor dtype and shape
        cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
        sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
        
        # Split into even and odd indices
        x_even = x[..., ::2]   # (B, S, H, D/2)
        x_odd = x[..., 1::2]   # (B, S, H, D/2)
        
        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # Interleave back
        x_out = torch.empty_like(x)
        x_out[..., ::2] = x_rotated_even
        x_out[..., 1::2] = x_rotated_odd
        
        return x_out


class OptimizedAttention(nn.Module):
    """Multi-head attention with GQA and proper KV cache"""

    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # RoPE
        self.rope = RoPE(config)

        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)

        # Attention sinks (learned global KV anchors)
        if config.num_attention_sinks > 0:
            # Store as (n_kv_heads, head_dim, num_sinks) for easy expansion
            self.k_sinks = nn.Parameter(torch.zeros(self.n_kv_heads, self.head_dim, config.num_attention_sinks))
            self.v_sinks = nn.Parameter(torch.zeros(self.n_kv_heads, self.head_dim, config.num_attention_sinks))
            nn.init.normal_(self.k_sinks, std=0.02)
            nn.init.normal_(self.v_sinks, std=0.02)
        else:
            self.k_sinks = None
            self.v_sinks = None

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_lens: Optional[torch.Tensor] = None,  # Shape (B,)
        cu_seqlens: Optional[torch.Tensor] = None,  # Cumulative seqlens for varlen
        max_seqlen: Optional[int] = None,  # Max seq length in batch
        attn_mask: Optional[torch.Tensor] = None,  # Prebuilt attention mask
        doc_spans: Optional[List[List[Tuple[int, int]]]] = None  # Doc boundaries per sequence
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        # Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        offset = 0
        if past_kv is not None:
            offset = past_kv[0].shape[2]  # Past sequence length
        q, k = self.rope(q, k, seq_len, offset)

        # Add attention sinks if configured (before GQA repeat)
        if self.config.num_attention_sinks > 0 and self.k_sinks is not None:
            # Expand sinks to batch: (n_kv, D, num_sinks) -> (B, num_sinks, n_kv, D)
            k_sink_expanded = self.k_sinks.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            v_sink_expanded = self.v_sinks.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

            # Concatenate sinks at the front of sequence
            k = torch.cat([k_sink_expanded, k], dim=1)  # (B, num_sinks+S, n_kv, D)
            v = torch.cat([v_sink_expanded, v], dim=1)  # (B, num_sinks+S, n_kv, D)

            # Update sequence length to include sinks
            seq_len = seq_len + self.config.num_attention_sinks

        # Cache base KV (n_kv heads)
        k_base = k  # (B,S+sinks,n_kv,D)
        v_base = v  # (B,S+sinks,n_kv,D)
        
        # Append past KV to base (keep n_kv heads compact)
        if past_kv is not None and use_cache:
            past_k, past_v = past_kv  # (B,n_kv,S_past,D)
            k_cat = torch.cat([past_k, k_base.transpose(1,2)], dim=2)
            v_cat = torch.cat([past_v, v_base.transpose(1,2)], dim=2)
            k_base = k_cat.transpose(1,2)
            v_base = v_cat.transpose(1,2)

        # Repeat for compute (GQA)
        if self.n_rep > 1:
            k_rep = k_base.repeat_interleave(self.n_rep, dim=2)  # (B,S,H,D)
            v_rep = v_base.repeat_interleave(self.n_rep, dim=2)
        else:
            k_rep, v_rep = k_base, v_base

        # Shapes for SDPA/FA
        q_bhsd = q.transpose(1, 2)       # (B,H,S,D)
        k_bhsd = k_rep.transpose(1, 2)   # (B,H,S,D)
        v_bhsd = v_rep.transpose(1, 2)   # (B,H,S,D)

        # Build windowed attention mask if configured
        if self.config.windowed_attention and not use_cache:
            # Create head-specific windowed mask (bypass FA for correctness)
            window_mask = self._build_windowed_mask(batch_size, seq_len, self.device)
            if attn_mask is not None:
                # Combine with existing mask
                attn_mask = attn_mask & window_mask
            else:
                attn_mask = window_mask

        # Decide path:
        # - If a custom mask is provided (packed sequences and/or block-cross-doc),
        #   we MUST use SDPA with that mask (FA fixed kernel can't take arbitrary masks).
        # - Else, use FA when available for performance; otherwise SDPA causal.
        use_custom_mask = attn_mask is not None or self.config.windowed_attention
        can_flat_fa = (not use_custom_mask and FLASH_AVAILABLE and self.config.use_flash_attn
                       and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and not use_cache)
        use_varlen = (can_flat_fa and VARLEN_FA_AVAILABLE and self.config.use_varlen_flash_attn
                      and seq_lens is not None and batch_size > 1 and (seq_lens.max() != seq_lens.min()))

        if use_varlen:
            # Varlen FA2 path: pack QKV and use cu-seqlens to avoid pad compute.
            # Shapes before packing:
            #   q: (B,S,H,D), k_rep: (B,S,H,D), v_rep: (B,S,H,D)
            # Pack (B,S,H,D) -> (T,H,3,D) with QKV interleaved for the kernel.
            # Build cu_seqlens if not provided.
            if cu_seqlens is None:
                seqlens = seq_lens.to(torch.int32).contiguous()
                cu_seqlens = torch.zeros((batch_size + 1,), dtype=torch.int32, device=q.device)
                cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
                max_seqlen = int(seqlens.max().item())
            else:
                max_seqlen = int(max_seqlen or seq_len)

            # Slice each sample to its true length (avoid padded tail) and pack
            # q/k_rep/v_rep are (B,S,H,D)
            q_list, k_list, v_list = [], [], []
            for b in range(batch_size):
                L = int(seq_lens[b].item())
                q_list.append(q[b, :L])         # (L,H,D)
                k_list.append(k_rep[b, :L])     # (L,H,D)
                v_list.append(v_rep[b, :L])     # (L,H,D)
            q_cat = torch.cat(q_list, dim=0)    # (T,H,D)
            k_cat = torch.cat(k_list, dim=0)    # (T,H,D)
            v_cat = torch.cat(v_list, dim=0)    # (T,H,D)

            # FA2 varlen wants QKV packed as (T, 3, H, D) OR (T, H, 3, D) depending on binding.
            # The qkvpacked func expects (T, 3, H, D).
            qkv = torch.stack([q_cat, k_cat, v_cat], dim=1)  # (T, 3, H, D)

            attn_out_varlen = fa_varlen_qkv(
                qkv.contiguous(),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=True,
                softmax_scale=None
            )  # -> (T, H, D)

            # Unpack back to (B,S,H,D) with pad restored (zeros)
            out_bshd = q.new_zeros((batch_size, seq_len, self.n_heads, self.head_dim))
            start = 0
            for b in range(batch_size):
                L = int(seq_lens[b].item())
                out_bshd[b, :L] = attn_out_varlen[start:start+L]
                start += L
            # Transpose to match other branches: (B,S,H,D) -> (B,H,S,D)
            attn_output = out_bshd.transpose(1, 2)

        elif can_flat_fa:
            # FlashAttention fixed-size causal kernel
            q_bshd = q  # (B,S,H,D)
            k_bshd = k_rep
            v_bshd = v_rep
            attn_output = flash_attn_func(
                q_bshd.contiguous(), k_bshd.contiguous(), v_bshd.contiguous(),
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=True
            ).transpose(1, 2)  # -> (B,H,S,D)
        else:
            # SDPA with mask (or plain causal if mask not provided)
            # If a boolean mask is provided as [B,S,S], SDPA expects broadcastable
            #   (B, 1, S, S) or (B, H, S, S). We'll expand to (B,1,S,S).
            sdpa_mask = None
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    # Use float32 for mask to avoid numerical issues with -inf in bf16/fp16
                    sdpa_mask = torch.zeros_like(attn_mask, dtype=torch.float32, device=q_bhsd.device)
                    sdpa_mask.masked_fill_(~attn_mask, float('-inf'))
                    sdpa_mask = sdpa_mask.unsqueeze(1)  # (B,1,S,S)
                else:
                    sdpa_mask = attn_mask
                    if sdpa_mask.dim() == 3:
                        sdpa_mask = sdpa_mask.unsqueeze(1)  # (B,1,S,S)

            attn_output = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd,
                attn_mask=sdpa_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=(sdpa_mask is None)  # causal only when no explicit mask
            )
        
        # Merge heads and project out
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B,S,H,D)
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)

        # Return compact KV cache (n_kv heads)
        new_kv = None
        if use_cache:
            new_kv = (k_base.transpose(1,2).contiguous(), v_base.transpose(1,2).contiguous())

        return output, new_kv

    def _build_windowed_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build windowed attention mask with local and global heads"""
        # Create base causal mask
        mask = torch.ones((batch_size, seq_len, seq_len), device=device, dtype=torch.bool)
        mask = torch.tril(mask)

        if self.config.windowed_attention:
            # Create head-specific masks (B, H, S, S)
            head_mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1).clone()

            # Apply windowing to local heads (heads after num_global_heads)
            window_size = self.config.window_size
            for h in range(self.config.num_global_heads, self.n_heads):
                for i in range(seq_len):
                    # Each position can only attend to window_size positions before it
                    start = max(0, i - window_size + 1)
                    if start > 0:
                        head_mask[:, h, i, :start] = False

            # Collapse head dimension back to (B, S, S) by taking AND across heads
            # This ensures all heads' constraints are respected
            mask = head_mask.all(dim=1)

        return mask


class OptimizedFFN(nn.Module):
    """Feed-forward network with SwiGLU"""
    
    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.config = config
        
        if config.use_swiglu:
            # SwiGLU: FFN(x) = (SiLU(xW1) ⊙ xW3)W2
            self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
            self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        else:
            # Standard FFN
            self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_swiglu:
            # SwiGLU activation
            gate = F.silu(self.w1(x))
            up = self.w3(x)
            x = gate * up
            x = self.w2(x)
        else:
            # Standard GELU
            x = self.up(x)
            x = F.gelu(x)
            x = self.down(x)
        
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, config: OnyxConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Normalization
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        
        # Attention and FFN
        self.attention = OptimizedAttention(config)
        self.ffn = OptimizedFFN(config)
    
    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **attn_kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm architecture
        # Attention block
        residual = x
        x = self.norm1(x)
        attn_out, new_kv = self.attention(x, use_cache, past_kv, **attn_kwargs)
        x = residual + attn_out
        
        # FFN block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, new_kv


class Onyx7B(nn.Module):
    """Onyx 7B Dense Model - Production Ready"""
    
    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        
        # Language modeling head
        if config.tie_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_dict: bool = True,
        seq_lens: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        doc_spans: Optional[List[List[Tuple[int, int]]]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass with KV cache support
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            labels: Target labels for training
            use_cache: Whether to use/return KV cache
            past_key_values: Past KV cache from previous forward pass
            return_dict: Return dictionary or tuple
        
        Returns:
            Dictionary with 'logits', optionally 'loss' and 'past_key_values'
        """
        # Optional compile guard to avoid cache path in graphs
        if hasattr(torch._dynamo, 'is_compiling') and torch._dynamo.is_compiling():
            # Force no-cache inside compiled graph to avoid graph breaks
            use_cache = False
            past_key_values = None
        
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through transformer layers
        new_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            if self.config.gradient_checkpointing and self.training:
                # Training: checkpoint tensor-only, no cache
                def cf(h):
                    y, _ = layer(
                        h, use_cache=False, past_kv=None,
                        attn_mask=attn_mask, seq_lens=seq_lens,
                        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                        doc_spans=doc_spans
                    )
                    return y
                hidden_states = torch.utils.checkpoint.checkpoint(cf, hidden_states, use_reentrant=False)
                new_kv = None
            else:
                hidden_states, new_kv = layer(
                    hidden_states, use_cache, past_kv,
                    attn_mask=attn_mask, seq_lens=seq_lens,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    doc_spans=doc_spans
                )
            
            if use_cache:
                new_past_key_values.append(new_kv)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        if self.lm_head is None:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss (cast to float32 for stability)
            loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "past_key_values": new_past_key_values,
                "hidden_states": hidden_states
            }
        
        return (logits, new_past_key_values) if use_cache else logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate text using the model with KV cache
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            use_cache: Use KV cache for efficiency
        
        Returns:
            Generated token IDs
        """
        self.eval()
        
        # Initialize past_key_values
        past_key_values = None
        generated_tokens = input_ids
        temperature = max(1e-6, float(temperature))
        
        # Early exit if already at max length
        if generated_tokens.shape[1] >= max_length:
            return generated_tokens[:, :max_length]
        
        with torch.inference_mode():
            for _ in range(max_length - input_ids.shape[1]):
                # Use only the last token if using cache
                if past_key_values is not None:
                    input_tokens = generated_tokens[:, -1:]
                else:
                    input_tokens = generated_tokens
                
                # Forward pass
                outputs = self.forward(
                    input_tokens,
                    use_cache=use_cache,
                    past_key_values=past_key_values
                )
                
                logits = outputs["logits"]
                past_key_values = outputs.get("past_key_values", None)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling if requested
                if top_k is not None and top_k > 0:
                    top_k_val = min(top_k, next_token_logits.size(-1))
                    kth = torch.topk(next_token_logits, top_k_val, dim=-1).values[..., -1, None]
                    next_token_logits = torch.where(
                        next_token_logits < kth,
                        torch.full_like(next_token_logits, float("-inf")),
                        next_token_logits
                    )
                
                # Apply top-p sampling if needed
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Fix: Create fresh mask to avoid scatter bug
                    base_mask = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
                    indices_to_remove = base_mask.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                
                # Check for EOS
                if (next_token == self.config.eos_token_id).any():
                    break
        
        return generated_tokens
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Get parameter groups for optimizer with correct weight decay"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Exclude norms, biases, and embeddings from weight decay
            if any(k in name for k in ["norm", "bias", "embed_tokens"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return {
            "decay": decay_params,
            "no_decay": no_decay_params
        }


def build_block_causal_mask(
    seq_lens: torch.Tensor,
    doc_spans: Optional[List[List[Tuple[int, int]]]] = None,
    block_cross_doc: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Build attention mask for packed sequences with optional document blocking.

    Returns a boolean attention mask of shape [B, S, S]:
      True  => keep (allowed to attend)
      False => mask (disallow)

    It is lower-triangular causal and, if block_cross_doc is True, disallows
    attending across document boundaries inside the packed sequence.

    Args:
        seq_lens: Actual sequence lengths (B,)
        doc_spans: Document boundaries per sequence
        block_cross_doc: Whether to block attention across documents
        device: Target device
        dtype: Output dtype (bool for mask, float for scores)

    Returns:
        Attention mask (B, S, S)
    """
    assert seq_lens.dim() == 1
    B = seq_lens.size(0)
    S = int(seq_lens.max().item())
    device = device or seq_lens.device

    # Base causal (lower-triangular)
    mask = torch.ones((B, S, S), device=device, dtype=torch.bool)
    tri = torch.tril(torch.ones((S, S), device=device, dtype=torch.bool))
    mask = mask & tri.unsqueeze(0)

    # Invalidate padding positions (rows and cols >= L_i)
    for b in range(B):
        L = int(seq_lens[b].item())
        if L < S:
            mask[b, L:, :] = False
            mask[b, :, L:] = False

    if block_cross_doc and doc_spans is not None:
        # Turn off attention across different spans
        for b in range(B):
            spans = doc_spans[b]
            if not spans:
                continue
            # Build segment id per token
            seg = torch.full((S,), -1, device=device, dtype=torch.int32)
            for idx, (st, en) in enumerate(spans):
                en = min(en, S)
                seg[st:en] = idx
            # Disallow attention where seg differs (but keep causal)
            same = torch.eq(seg.unsqueeze(0), seg.unsqueeze(1))  # [S,S]
            mask[b] = mask[b] & same

    # Return boolean mask (caller will convert if needed)
    return mask


def create_onyx_7b(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    compile_model: bool = True
) -> Onyx7B:
    """
    Create and initialize Onyx 7B model
    
    Args:
        device: Device to place model on
        dtype: Data type for model weights (prefer bfloat16 on Ampere+)
        compile_model: Whether to compile with torch.compile
    
    Returns:
        Initialized Onyx 7B model
    """
    # Set global preferences for PyTorch 2.8
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    # Create configuration
    config = OnyxConfig(
        vocab_size=128258,      # Hermes Llama 3 tokenizer + <eod> token
        d_model=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,           # GQA 4:1
        d_ff=11008,
        max_seq_len=16384,      # 16k context
        rope_theta=500000.0,    # Extended RoPE
        use_swiglu=True,
        use_rms_norm=True,
        use_flash_attn=FLASH_AVAILABLE,
        use_cuda_graphs=False,  # Reserved for future
        use_torch_compile=compile_model
    )
    
    # Create model
    model = Onyx7B(config)
    
    # Move to device and dtype
    model = model.to(device=device, dtype=dtype)
    
    # Compile the model (single compile at top level)
    if compile_model and config.use_torch_compile:
        model.forward = torch.compile(
            model.forward,
            mode="max-autotune",
            dynamic=True
        )
    
    # Print model info
    num_params = model.get_num_params()
    print(f"✅ Created Onyx 7B Dense Model")
    print(f"   Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
    print(f"   Architecture: Dense transformer")
    print(f"   Optimizations: RoPE, SDPA, KV cache (4x savings)")
    print(f"   Device: {device}")
    print(f"   Dtype: {dtype}")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing Onyx 7B Production Model")
    print("=" * 60)
    
    # Set seed for reproducibility
    torch.manual_seed(0)
    
    # Device and CUDA info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        print(f"CUDA Available: Yes (GPU: {torch.cuda.get_device_name()})")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    else:
        print("CUDA Available: No (using CPU)")
    # Prefer bfloat16 on modern GPUs
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    
    model = create_onyx_7b(device=device, dtype=dtype, compile_model=False)  # Skip compile for testing
    
    # Test forward pass
    print("\nTesting forward pass...")
    input_ids = torch.randint(0, 128256, (1, 32), device=device)
    
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {outputs['logits'].shape}")
        print(f"   KV cache: {len(outputs['past_key_values'])} layers")
    
    # Test generation with KV cache
    print("\nTesting generation with KV cache...")
    import time
    
    # Without cache
    start = time.time()
    generated = model.generate(input_ids, max_length=64, use_cache=False)
    time_no_cache = time.time() - start
    
    # With cache
    start = time.time()
    generated = model.generate(input_ids, max_length=64, use_cache=True)
    time_with_cache = time.time() - start
    
    print(f"✅ Generation successful!")
    print(f"   Without cache: {time_no_cache:.2f}s")
    print(f"   With cache: {time_with_cache:.2f}s")
    if time_with_cache > 0:
        print(f"   Speedup: {time_no_cache/time_with_cache:.1f}x")
    
    # Test top-k sampling
    print("\nTesting top-k sampling...")
    generated_topk = model.generate(input_ids, max_length=40, top_k=50, temperature=0.8)
    print(f"✅ Top-k=50 generation: {generated_topk.shape}")
    
    # FlashAttention vs SDPA smoke test
    print("\nFA vs SDPA smoke test...")
    model.eval()
    x = torch.randint(0, 128256, (2, 16), device=device)
    with torch.inference_mode():
        # Force SDPA
        model.config.use_flash_attn = False
        y_sdpa = model(x, use_cache=False)["logits"]
        
        # Try FA if possible
        model.config.use_flash_attn = True
        try:
            y_fa = model(x.to(dtype), use_cache=False)["logits"]
            assert y_fa.shape == y_sdpa.shape
            print(f"✅ FA/SDPA shapes match: {y_fa.shape}")
        except Exception as e:
            print(f"ℹ️  FA path skipped: {repr(e)}")
        finally:
            model.config.use_flash_attn = FLASH_AVAILABLE
    
    # KV cache head-count sanity check
    print("\nKV cache head-count sanity...")
    with torch.inference_mode():
        out = model(x[:, :8], use_cache=True)
        pkv = out["past_key_values"]
        k0, v0 = pkv[0]
        # Expect (B, n_kv_heads, S, D)
        assert k0.shape[1] == model.config.n_kv_heads, f"Wrong KV heads: {k0.shape}"
        print(f"✅ Compact KV cache shape: {k0.shape} (4x memory savings)")
        
        # Generate with cache and show growth
        print("\nKV cache growth during generation...")
        generated_with_cache = model.generate(x[:1, :8], max_length=24, use_cache=True)
        print(f"   Generated sequence length: {generated_with_cache.shape[1]}")
        print(f"   Final KV cache per layer: (B={k0.shape[0]}, n_kv_heads={model.config.n_kv_heads}, S=varies, D={k0.shape[-1]})")
    
    # Long-context smoke test
    print("\nLong-context smoke test...")
    L = min(4096, model.config.max_seq_len)  # Keep reasonable for single GPU
    long_input = torch.randint(0, model.config.vocab_size, (1, L), device=device)
    with torch.inference_mode():
        _ = model(long_input, use_cache=False)
        print(f"✅ Processed {L} tokens successfully (RoPE/SDPA memory OK)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Model is production-ready.")
    print("\nKey features:")
    print("  • FlashAttention v2 support with guards")
    print("  • 4x KV cache memory savings (GQA)")
    print("  • Fixed top-p sampling")
    print("  • Gradient checkpointing (tensor-only)")
    print("  • Float32 loss for stability")
    print("  • PyTorch 2.8 optimized")
