"""Encoder-decoder Transformer for univariate or multivariate time-series forecasting.

This module implements the architecture from the classic Transformer diagram:
- Encoder stack with self-attention + FFN + residual/normalization
- Decoder stack with masked self-attention, cross-attention, FFN
- Final linear head for horizon forecasts

The model supports:
- Past target/context window (encoder input)
- Decoder input with shifted-right targets during training
- Autoregressive generation during inference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    input_size: int = 1
    output_size: int = 1
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 512


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: [1, max_seq_len, d_model] for batch-first tensors.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Sequence-to-sequence Transformer for forecasting.

    Inputs are expected batch-first:
      - src: [batch, context_length, input_size]
      - tgt: [batch, target_length, output_size] (shifted right during training)

    Forward output:
      - predictions: [batch, target_length, output_size]
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model

        self.src_projection = nn.Linear(cfg.input_size, cfg.d_model)
        self.tgt_projection = nn.Linear(cfg.output_size, cfg.d_model)

        self.src_positional_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.tgt_positional_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.output_head = nn.Linear(cfg.d_model, cfg.output_size)

    @staticmethod
    def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Return upper-triangular mask with -inf above diagonal."""
        mask = torch.full((size, size), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def encode(self, src: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_emb = self.src_projection(src) * math.sqrt(self.d_model)
        src_emb = self.src_positional_encoding(src_emb)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_projection(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_positional_encoding(tgt_emb)

        tgt_len = tgt_emb.size(1)
        tgt_mask = self._generate_causal_mask(tgt_len, tgt_emb.device)

        dec_out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output_head(dec_out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src=src, src_padding_mask=src_padding_mask)
        return self.decode(
            memory=memory,
            tgt=tgt,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask,
        )

    @torch.no_grad()
    def predict(self, src: torch.Tensor, horizon: int, start_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Autoregressive inference.

        Args:
            src: historical context [batch, context_len, input_size]
            horizon: number of future time steps to generate
            start_token: optional initial decoder token [batch, 1, output_size].
                         If None, zeros are used.

        Returns:
            Tensor of shape [batch, horizon, output_size]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        memory = self.encode(src)

        if start_token is None:
            ys = torch.zeros(batch_size, 1, self.cfg.output_size, device=device)
        else:
            ys = start_token

        for _ in range(horizon):
            out = self.decode(memory=memory, tgt=ys)
            next_step = out[:, -1:, :]
            ys = torch.cat([ys, next_step], dim=1)

        return ys[:, 1:, :]


def make_decoder_input(y_true: torch.Tensor) -> torch.Tensor:
    """Shift-right helper used for teacher forcing.

    y_true shape: [batch, horizon, output_size]
    returns:      [batch, horizon, output_size]
    """
    bos = torch.zeros(y_true.size(0), 1, y_true.size(2), device=y_true.device, dtype=y_true.dtype)
    return torch.cat([bos, y_true[:, :-1, :]], dim=1)
