# Main Predictive Results

This report reproduces the headline predictive model comparison used in the anonymous GitHub release.

Evaluation setup:
- All models use the same shared training objective for this comparison: the active directional core on non-USD targets only.
- Active set definition: `A = {(b, i) : |y_{b,i}| >= tau}`.
- Threshold definition: `tau = Q0.40(|y_train_norm|)`.
- Core loss on the active set: `mean softplus(- rhat_{b,i} * sign(y_{b,i}))`.
- Seed set: `21, 42, 456`. Reported numbers are mean ± standard deviation across these seeds.
- `Extreme Hit` is the sign hit ratio computed on the pooled train-based extreme subset, where `|y|` exceeds the training `Q0.80` threshold over non-USD targets.
- `Stress Mean Hit Avg.` is the simple average of sign hit ratios across the three stress scenarios `{VIX stress, FX stress, Yield stress}`.
- `Stress Extreme Hit Avg. @ Q80` is computed inside each stress scenario using that scenario's realized `Q0.80` absolute-return subset, and then averaged across the three scenarios.
- Stress scenarios are defined from training-sample percentile rules: `VIX stress = train-rank(|VIX change|) >= 0.85`, `FX stress = train-rank(cross-currency mean absolute FX return) >= 0.85`, and `Yield stress = train-rank(US10Y change) >= 0.85`.
- `Corr-LSTM-GAT` and `FXRP` are included as reference baselines.

## Headline Predictive Metrics

| Model | Mean Hit ↑ | Extreme Hit ↑ | Stress Mean Hit Avg. ↑ | Stress Extreme Hit Avg. @ Q80 ↑ | RMSE ×10³ ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ARC_FX** | **0.4997 ± 0.0041** | **0.4993 ± 0.0170** | **0.5085 ± 0.0154** | **0.5403 ± 0.0247** | **6.219 ± 0.009** |
| MLP | 0.5105 ± 0.0083 | 0.5080 ± 0.0045 | 0.4929 ± 0.0047 | 0.4611 ± 0.0297 | 6.203 ± 0.001 |
| Transformer | 0.5023 ± 0.0056 | 0.5100 ± 0.0099 | 0.5080 ± 0.0186 | 0.5229 ± 0.0372 | 6.230 ± 0.011 |
| GNN | 0.5017 ± 0.0056 | 0.5080 ± 0.0146 | 0.4918 ± 0.0036 | 0.4965 ± 0.0101 | 6.216 ± 0.022 |
| Corr-LSTM-GAT | 0.5006 ± 0.0126 | 0.5197 ± 0.0180 | 0.5113 ± 0.0193 | 0.5398 ± 0.0763 | 6.204 ± 0.007 |
| FXRP | 0.4987 ± 0.0138 | 0.5098 ± 0.0106 | 0.4928 ± 0.0066 | 0.4697 ± 0.0243 | 6.214 ± 0.015 |

## Non-Trivial Directional Hit And F1

The neutral-band thresholds below are basis-point bands (`±1bp`, `±3bp`, `±5bp`, `±10bp`).

- `Non-trivial hit`: excludes only the observations where both the prediction and the realized return fall inside the same neutral band, then measures sign hit on the remaining subset.
- `Macro-F1`: computed on the same subset using ternary labels `{down, flat, up}` induced by the threshold.

### Non-Trivial Directional Hit

| Model | ±1bp | ±3bp | ±5bp | ±10bp |
| --- | ---: | ---: | ---: | ---: |
| ARC_FX | 0.5005 ± 0.0040 | 0.5022 ± 0.0041 | 0.5032 ± 0.0060 | 0.5051 ± 0.0059 |
| MLP | 0.5114 ± 0.0078 | 0.5131 ± 0.0080 | 0.5152 ± 0.0076 | 0.5164 ± 0.0088 |
| Transformer | 0.5030 ± 0.0060 | 0.5042 ± 0.0060 | 0.5049 ± 0.0062 | 0.5055 ± 0.0079 |
| GNN | 0.5027 ± 0.0062 | 0.5045 ± 0.0066 | 0.5071 ± 0.0071 | 0.5094 ± 0.0070 |
| Corr-LSTM-GAT | 0.5015 ± 0.0129 | 0.5030 ± 0.0138 | 0.5038 ± 0.0134 | 0.5069 ± 0.0138 |
| FXRP | 0.4997 ± 0.0135 | 0.5008 ± 0.0135 | 0.5006 ± 0.0138 | 0.5041 ± 0.0129 |

### Macro-F1

| Model | ±1bp | ±3bp | ±5bp | ±10bp |
| --- | ---: | ---: | ---: | ---: |
| ARC_FX | 0.2993 ± 0.0097 | 0.2329 ± 0.0158 | 0.1672 ± 0.0239 | 0.0482 ± 0.0264 |
| MLP | 0.2749 ± 0.0067 | 0.1883 ± 0.0216 | 0.1170 ± 0.0380 | 0.0220 ± 0.0136 |
| Transformer | 0.3057 ± 0.0100 | 0.2414 ± 0.0202 | 0.1811 ± 0.0295 | 0.0728 ± 0.0456 |
| GNN | 0.2880 ± 0.0098 | 0.1963 ± 0.0486 | 0.1188 ± 0.0728 | 0.0382 ± 0.0533 |
| Corr-LSTM-GAT | 0.2417 ± 0.0555 | 0.1354 ± 0.1155 | 0.0732 ± 0.0804 | 0.0013 ± 0.0023 |
| FXRP | 0.2453 ± 0.0303 | 0.1507 ± 0.0811 | 0.0859 ± 0.0863 | 0.0215 ± 0.0366 |
