# Transformer for Time Series Forecasting

Transformers are deep learning models that use **self-attention** to learn temporal dependencies and patterns in sequential data. For forecasting, a transformer is trained on historical values (and optional covariates) to predict future values.

## Architecture Overview

A time series transformer typically has two main blocks:

1. **Encoder**: Processes historical context (past values).
2. **Decoder**: Produces forecasts over the prediction horizon (future values).

---

## 1) Encoder

The encoder converts past observations into contextual representations that summarize relevant temporal information.

### a) Input Embedding + Positional Encoding
- Numeric time series inputs are projected into a latent vector space of size `d_model`.
- Positional (or time) encoding is added so the model can preserve order and timing information.

### b) Self-Attention
- Each time step attends to other time steps in the input window.
- This lets the model learn which historical points are most relevant for forecasting.

### c) Feed-Forward Network (FFN)
A two-layer MLP is applied at each position:

\[
\mathrm{FFN}(x) = \mathrm{ReLU}(xW_1 + b_1)W_2 + b_2
\]

### d) Residual Connections + Layer Normalization
- Each sub-layer (attention, FFN) is wrapped with residual connections.
- Layer normalization stabilizes optimization and improves training of deeper stacks.

### e) Stacked Encoder Layers
- Multiple encoder layers are stacked.
- Deeper layers progressively refine representations and capture higher-level temporal structure.

---

## 2) Decoder

The decoder generates future values for a forecast horizon `H`.

### a) Decoder Inputs
- **Training**: shifted ground-truth targets (teacher forcing) + known future covariates.
- **Inference**: autoregressive decoding using previously predicted values + known future covariates.

After embedding and positional encoding, decoder input shape is typically:

\[
[\text{batch\_size}, H, d_{model}]
\]

### b) Masked Self-Attention
- Prevents a future position from attending to later future positions.
- Ensures causal forecasting behavior.

### c) Cross-Attention (Encoder-Decoder Attention)
- Decoder queries attend to encoder outputs.
- Connects each forecast step to relevant historical context.

### d) Feed-Forward Network + Residual/Norm
- Same FFN block and residual-normalization pattern as in encoder layers.

### e) Multi-Head Attention
- Attention is split across multiple heads.
- Each head can focus on different temporal patterns (seasonality, trends, local events, etc.).

---

## Forecasting Workflow

1. Split timeline into:
   - observed past interval: `[0, b]`
   - forecast interval: `[b, b + horizon]`
2. Provide model inputs:
   - historical target values
   - optional dynamic covariates (past/future known features)
   - optional static features
3. Encoder learns compressed historical context.
4. Decoder generates predictions for each step in the horizon.

---

## Why Transformers Work Well for Time Series

- Capture long-range dependencies better than many recurrent models.
- Parallelizable training over sequence positions.
- Flexible integration of covariates and static metadata.

## Practical Challenges

- Data-hungry compared to simpler baselines.
- Sensitive to window size, horizon, and feature engineering.
- Computationally expensive for very long contexts.

## Common Improvements

- Probabilistic heads (quantile or distribution outputs).
- Patch/chunk-based tokenization for long sequences.
- Efficient attention variants for long-context forecasting.
