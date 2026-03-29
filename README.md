# Time-Series Transformer (Encoder-Decoder)

Du wolltest genau dieses Transformer-Modell bauen (Encoder + Decoder mit Masked Attention, Cross-Attention, FFN, Add&Norm).
Dieses Repository enthält dafür eine direkt nutzbare PyTorch-Implementierung.

## Dateien

- `timeseries_transformer.py`
  - `TransformerConfig`: Modell-Hyperparameter
  - `TimeSeriesTransformer`: vollständiges Seq2Seq-Transformer-Modell
  - `make_decoder_input`: Shift-right für Teacher Forcing
- `train_example.py`
  - End-to-end Trainingsbeispiel auf synthetischen Zeitreihen

## Architektur-Mapping zum Diagramm

- **Encoder Block (Nx)**
  - Input-Projektion + Positional Encoding
  - Multi-Head Self-Attention
  - Feed Forward
  - Residual + LayerNorm (in `nn.Transformer` enthalten)
- **Decoder Block (Nx)**
  - Output-Projektion + Positional Encoding
  - Masked Multi-Head Self-Attention
  - Encoder-Decoder (Cross) Attention
  - Feed Forward
  - Residual + LayerNorm
- **Output Head**
  - Linear-Schicht auf `output_size`

## Installation

```bash
pip install torch
```

## Training starten

```bash
python train_example.py
```

## Eigene Daten verwenden

1. Erzeuge Fenster:
   - `src`: `[batch, context_length, input_size]`
   - `y_true`: `[batch, horizon, output_size]`
2. Erzeuge Decoder-Input mit Shift-right:
   ```python
   decoder_in = make_decoder_input(y_true)
   ```
3. Trainiere:
   ```python
   pred = model(src, decoder_in)
   loss = mse(pred, y_true)
   ```
4. Inferenz:
   ```python
   forecast = model.predict(src, horizon=24)
   ```

## Hinweise

- Für multivariate Features setze `input_size > 1`.
- Für probabilistische Forecasts kannst du den Output-Head erweitern (z. B. Quantile).
