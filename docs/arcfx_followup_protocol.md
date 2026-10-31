# Controlled ARC_FX follow-up

Run `python src/arcfx_followup.py`; regenerate analysis using `--analysis-only`.
Outputs are isolated in `results/arcfx_followup/`. Original search files are read only.

## Questions and fixed comparisons

The model is the previous temporal MLP, dense signed QK graph, direct plus exact zero-message graph residual, and unweighted direction BCE. No encoder/loss/sparsification search is added. The primary comparisons are A4−A3, A3−A1, and A4−A2. A1: old input/no graph; A2: old input/graph; A3: latest input/no graph; A4: latest input/graph. All target rows, features within a policy, seeds 0–4, initialization of common modules, shuffle order, optimization settings, early stopping, and checkpoint criteria match. Graph and no-graph instantiate exactly the same parameter tensors; the no-graph variant zeros its messages during training and inference. USD gauge subtraction and fixed common market/regime features remain. There is no foreign-node pooling in the direct encoder or gate.

## Availability policies

The existing PIT non-FX builder assigns observation availability and checks against 11:30 New York. The H.10 rates themselves are noon observations, not necessarily publications available at noon. The [Federal Reserve](https://www.federalreserve.gov/releases/h10/default.htm) publishes the previous week's daily rates on Mondays at 16:15 ET, delayed for federal holidays. Its weekly release was interrupted in May 2006–January 2009. The panel uses subsequently downloaded historical data rather than complete per-release FX vintages.

Two policies separate this uncertainty. `market_feed_assumed` exactly reproduces the prior input and scalers; it requires an unverified contemporaneous spot feed. `conservative_lag14` lags FX-derived features by at least 14 calendar days and adds one panel-date lag to the supplied non-FX PIT features. Causal regimes and feature scalers are recomputed after these lags. This is a schedule buffer rather than certified reconstruction of every historical availability timestamp or revision. Training scalers only see lagged training features. Labels are always the same unshifted appreciation returns t→Target_Date.

There are 20 2×2 conditions per policy, 40 in total. Eight exactly matching old latest-input conditions can be reused after data/model hashes, settings, and recalculated prediction equality checks. Existing MLP and Transformer baselines add 10 conditions per policy under exactly the same feature/split/BCE/optimizer/seed/checkpoint rules. Planned total: 60 model conditions, 52 fresh neural fits, 8 reused fits. Constant estimates, 60 validation-only calibrators (including every matched neural baseline), inference interventions and metric calculations are separately counted. The small smoke test performs no fit.

## Probability, timing, and graph diagnostics

Raw labels are 1[r>0]. Historical sign Hit retains three-way zero handling; binary accuracy uses logit>0. Constant probability baselines obey their specified p>0.5 direction rule (down at a tie), while a neural logit exactly zero retains its legacy zero sign. Direction logits are never assigned return RMSE.

Train-only currency probabilities and fixed train/validation mean logits quantify static direction frequency. Positive common calibration slope and ten currency-node offsets are fitted only on validation BCE. USD offset is zero. No independent cross-pair correction is added. Reliability, Brier, per-currency confusion matrices/AUC, residual-label correlations and common circular shifts provide complementary diagnostics. Pooled AUC may benefit from different currency frequencies; within-currency AUC is needed for temporal discrimination. Calibration does not change within-currency ranking.

Inference interventions zero the actual aggregated message or replace adjacency by its training-sample mean, retaining states, gates and weights. Record original-correct→intervention-wrong and original-wrong→intervention-correct separately. Adjacent-date adjacency changes, message RMS, gates, and per-date/currency logit/Hit changes are saved. These are dependence interventions, not independently trained static-graph models. Circular shifts exclude ±60 dates and are descriptive under nonstationarity.

Paired circular block bootstrap resamples dates with all currencies and models together, averaging matched fitted seeds before time resampling. Blocks 5/20/60, 2,000 draws. Hit and binary differences are percentage points; BCE reduction is positive for improvement. Seed standard deviation and time confidence intervals are separate. Previously viewed report data and search/selection effects are not corrected.

## User-authorized scope exploration

The latest user instruction explicitly permits selecting favorable evaluation periods and forecast targets and takes precedence over the pasted text's older all-currency restriction. Every selected comparison uses the same dates and targets for all models, with the same full-universe training procedure. Report all results and search counts; retain all five seeds.

Search uses full reporting dates and contiguous calendar-half blocks with at least 126 target dates. Evaluate all 1,023 nonempty subsets of ten USD targets, plus 55 individually evaluated strength-difference currency pairs. Pair targets are synchronized raw strength-return differences. The graph stays at 11 nodes, and training losses still cover the original ten USD targets. Thus selection changes the evaluation estimand, not the trained universe.

Two fixed candidates are the original A4 and its validation-only calibrated version. Calibrated A4 is compared with calibrated A3, MLP and Transformer fitted using the same calibration procedure. Rank by absolute Hit, minimum advantage over all designated matched comparators (A3, MLP, Transformer, train-only direction frequency), and graph advantage over A3. All combinations are saved, including negative results. Full-sample diagnostics are also saved. A separate validation-only currency selection check is evaluated on all reporting dates. Report-selected maxima are descriptive; even a positive paired bootstrap interval on such a maximum would not establish selection-adjusted significance or future superiority.
