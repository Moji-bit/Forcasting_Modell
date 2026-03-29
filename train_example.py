"""Minimal training example for TimeSeriesTransformer.

Run:
    python train_example.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from timeseries_transformer import TransformerConfig, TimeSeriesTransformer, make_decoder_input


def synthetic_series(num_points: int = 3000) -> torch.Tensor:
    t = torch.linspace(0, 80 * math.pi, steps=num_points)
    y = torch.sin(t) + 0.4 * torch.sin(0.2 * t) + 0.05 * torch.randn_like(t)
    return y.unsqueeze(-1)  # [T, 1]


def make_windows(series: torch.Tensor, context: int, horizon: int):
    xs, ys = [], []
    total = series.size(0)
    for i in range(total - context - horizon):
        xs.append(series[i : i + context])
        ys.append(series[i + context : i + context + horizon])
    return torch.stack(xs), torch.stack(ys)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_length = 96
    horizon = 24

    series = synthetic_series()
    x, y = make_windows(series, context=context_length, horizon=horizon)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    cfg = TransformerConfig(
        input_size=1,
        output_size=1,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=max(context_length, horizon) + 8,
    )
    model = TimeSeriesTransformer(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, 6):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            decoder_in = make_decoder_input(yb)
            pred = model(src=xb, tgt=decoder_in)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(loader.dataset)
        print(f"Epoch {epoch:02d} | train_loss={epoch_loss:.6f}")

    # Quick autoregressive inference on one batch.
    model.eval()
    sample_src, sample_tgt = next(iter(loader))
    sample_src = sample_src[:1].to(device)
    sample_tgt = sample_tgt[:1].to(device)
    forecast = model.predict(src=sample_src, horizon=horizon)

    print("target[:5]  :", sample_tgt[0, :5, 0].detach().cpu())
    print("forecast[:5]:", forecast[0, :5, 0].detach().cpu())


if __name__ == "__main__":
    main()
