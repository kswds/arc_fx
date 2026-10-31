# ARC_FX tuning summary (validation only)

- tuning data ends 2021-10-05; test period (from 2021-10-06) never loaded
- candidates: 30, seeds per candidate: 3
- default config: val hit 0.4909 (seed std 0.0014)
- best candidate `t021`: val hit 0.5012 (seed std 0.0018)
- val hit across candidates: median 0.4932, max 0.5012

Top 5:

trial     lr  weight_decay  dropout  edge_dropout  small_return_quantile  lambda_component  graph_reg_scale  hit_alpha  batch_size  val_hit_mean  val_hit_std  val_ic_mean  val_rmse_mean  best_epoch_mean  minutes
 t021 0.0010        0.0100      0.4          0.20                    0.6             0.000              0.1       0.05          64      0.501240     0.001797     0.003418       0.005207         3.666667     2.67
 t002 0.0010        0.0010      0.4          0.10                    0.4             0.001             10.0       0.01         128      0.497996     0.002703     0.002464       0.005208         4.666667     2.40
 t028 0.0003        0.0100      0.4          0.00                    0.4             0.001              0.1       0.00          64      0.497615     0.004486    -0.007551       0.005222         7.666667     3.43
 t020 0.0010        0.0010      0.4          0.20                    0.6             0.000              0.1       0.00         256      0.497583     0.004007    -0.004244       0.005246         5.666667     2.28
 t012 0.0010        0.0001      0.4          0.05                    0.0             0.001             10.0       0.00          64      0.496788     0.002314     0.006025       0.005209         3.333333     2.72