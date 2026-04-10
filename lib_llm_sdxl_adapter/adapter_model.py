"""Implement the shared RouWei adapter network for ReForge."""

from __future__ import annotations

import torch
from torch import nn


def pad_to_length(
    tensor: torch.Tensor, target_length: int, dim: int = 1, value: int | float = 0
) -> torch.Tensor:
    """Pad or truncate one tensor along the requested dimension."""

    current_length = tensor.size(dim)
    if current_length >= target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_size = list(tensor.shape)
    pad_size[dim] = target_length - current_length
    padding = torch.full(pad_size, value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim)


class TransformerBlock(nn.Module):
    """Apply one self-attention block over a batched token sequence."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run self-attention followed by a feed-forward residual block."""

        key_padding_mask = None if mask is None else ~mask.bool()
        normalized = self.norm1(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return x


class LLMToSDXLAdapter(nn.Module):
    """Convert one batch of LLM hidden states into SDXL-style conditioning tensors."""

    def __init__(
        self,
        *,
        llm_dim: int = 1152,
        sdxl_seq_dim: int = 2048,
        sdxl_pooled_dim: int = 1280,
        max_input_len: int = 512,
        target_seq_len: int = 308,
        n_wide_blocks: int = 2,
        n_narrow_blocks: int = 3,
        num_heads: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.seq_projection: nn.Linear | None = None
        if llm_dim != sdxl_seq_dim:
            self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim)

        self.input_position_embeddings = nn.Parameter(
            torch.randn(1, max_input_len, sdxl_seq_dim)
        )
        self.output_position_embeddings = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        self.wide_attention_blocks = nn.ModuleList(
            [
                TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(n_wide_blocks)
            ]
        )
        self.compression_queries = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        self.compression_attention = nn.MultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        self.compression_gate = nn.Sequential(
            nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim), nn.Sigmoid()
        )
        self.narrow_attention_blocks = nn.ModuleList(
            [
                TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(n_narrow_blocks)
            ]
        )
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.pooling_token = nn.Parameter(torch.randn(1, 1, sdxl_seq_dim))
        self.pooled_projection = nn.Sequential(
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_pooled_dim),
        )

    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project one hidden-state batch into sequence and pooled SDXL conditioning tensors."""

        batch_size, seq_len, _ = llm_hidden_states.shape
        hidden_states = (
            self.seq_projection(llm_hidden_states)
            if self.seq_projection is not None
            else llm_hidden_states
        )

        if seq_len > self.max_input_len:
            hidden_states = hidden_states[:, : self.max_input_len, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_input_len]
        elif seq_len < self.max_input_len:
            hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1)
            if attention_mask is None:
                attention_mask = torch.ones(
                    batch_size,
                    self.max_input_len,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                attention_mask[:, seq_len:] = 0
            else:
                attention_mask = pad_to_length(
                    attention_mask, self.max_input_len, dim=1, value=0
                )

        hidden_states = hidden_states + self.input_position_embeddings
        for block in self.wide_attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        queries = self.compression_queries.expand(batch_size, -1, -1)
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        compressed_sequence, _ = self.compression_attention(
            queries,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        gate_input = torch.cat([queries, compressed_sequence], dim=-1)
        gate_weights = self.compression_gate(gate_input)
        compressed_sequence = (
            gate_weights * compressed_sequence + (1 - gate_weights) * queries
        )
        compressed_sequence = self.compression_norm(compressed_sequence)
        compressed_sequence = compressed_sequence + self.output_position_embeddings

        for block in self.narrow_attention_blocks:
            compressed_sequence = block(compressed_sequence)

        pooling_token = self.pooling_token.expand(batch_size, -1, -1)
        pooled_output, _ = self.pooling_attention(
            pooling_token, compressed_sequence, compressed_sequence, need_weights=False
        )
        pooled_output = self.pooled_projection(pooled_output.squeeze(1))
        return compressed_sequence, pooled_output
