# Controlled selector comparison

The user requested both fixed backbones: the recent compact MLP/BCE/QK model and the original channel-separated GRU/static+QK+macro/legacy-loss model. Within each backbone, only the graph-selection operation varies: signed Top-k=3, dense signed, or signed soft-threshold. Original files/results are preserved.

All candidates receive the same supplied PIT feature arrays, normalization fitted on the original 60% training prefix, lookback 10 ending at decision row t, all ten non-USD targets, seeds 0–4, batch256, AdamW lr0.0003/weight decay0.0001, max20 epochs, patience4, and gradient norm clip1. This is a bounded controlled experiment, not an exact reproduction of the original 80-epoch/scheduler/checkpoint recipe. The original backbone's architecture and loss penalties remain intact. Current-noon FX availability remains a market-feed assumption; no extra lag14 policy is introduced here.

The train and test row indices exactly match the historical PIT 60/20/20 sequence split. Original validation is 1,048 dates immediately after training. The overlapping condition moves this same-length validation window forward by 524 panel rows while holding train/test fixed. Thus 524/1,048=50% of validation and 524/1,049≈49.95% of test overlap. The intervening 524 rows between training and shifted validation are not used as fitting labels. Test comprises 1,049 common decision dates, 2021-10-06 through 2025-12-17. Its first 524 and last 525 dates are reported separately under both protocols.

Top-k and dense use softmax(abs(score)) times tanh(score), with diagonal zero. Signed soft-threshold applies sign(score) relu(abs(score)−tau), diagonal zero, then divides each row by max(1, absolute row sum). Fixed tau grid is [0,0.025,0.05,0.1,0.2]. Tau is common across nodes/dates and is selected separately per backbone/protocol by mean validation sign-hit across all five seeds, breaking ties with lower validation training objective and then smaller tau. Checkpoints use the corresponding per-seed validation sign-hit maximum and objective tie-breaker.

The original static reference adjacency and its existing Top-k normalization remain fixed for all three choices; only the dynamic adjacency selection is replaced. All original regularization coefficients and edge-dropout0.05 are retained. Inverted edge dropout can enlarge row mass during training; the soft-threshold row-mass bound holds before dropout and at evaluation. Original target normalization, masked directional objective and inverse transformation remain the original implementation. Compact outputs are direction logits and are never converted into return predictions.

There are 70 optimizer trajectories: two backbones × seven selector/threshold settings × five seeds. Each trajectory tracks two independent validation/checkpoint/early-stop states. Constant learning rate and absence of validation-dependent training operations allow the same training trajectory to be reused for both protocols: neither validation changes gradients, optimizer updates or RNG. When a protocol reaches its patience, its selection state freezes even if the other protocol continues. There are consequently 140 logical validation conditions. After threshold selection is frozen, 60 selected models are evaluated on test.

The requested maximum is the highest test Mean Hit over the five seeds of each validation-selected setting. It is explicitly a post-hoc test-seed maximum and is accompanied by the five-seed mean/std. Test does not select thresholds or epochs. No period or currency subset is selected by test performance in this experiment. Overlapping validation/test results are intentionally contaminated by checkpoint/hyperparameter selection and cannot be called independent test performance. Even the original non-overlap test was already inspected in earlier work.

Paired circular moving-block bootstrap averages same-date currency correctness differences and then the common five fitted seeds. Blocks5/20/60, 2,000 repetitions, percentage-point effects. Intervals describe conditional time uncertainty; they do not correct threshold selection, maximum-seed reporting or validation/test overlap.

Reproduce:

```powershell
python src/check_arcfx_selector_compare.py --smoke-only
python src/arcfx_selector_compare.py
python src/check_arcfx_selector_compare.py
```

The smoke check verifies exact model/selector equivalence, common initialization, finite gradients, all-zero messages, weak-edge non-amplification, diagonal handling and the overlap geometry. Full verification additionally checks selection provenance, all seeds/targets, actual labels, maxima and checkpoint predictions. The runner persists resumable epoch states and retains failed attempts in a ledger.
