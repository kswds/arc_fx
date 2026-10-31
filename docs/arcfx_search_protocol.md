# ARC_FX bounded exploration protocol

This study keeps the released code and results unchanged. Run `python src/run_arcfx_search.py` from the repository root. Results are written to `results/arcfx_search/`; rerunning skips finished runs and restores unfinished epoch checkpoints. `python src/check_arcfx_search.py` checks behavioral invariants without fitting experiment candidates.

The configuration is saved in `configs/arcfx_search.json`. Initial search uses seed 0, two chronological development intervals (50–60% and 60–70% of panel rows), 20 maximum epochs, patience 4, batch size 256, AdamW learning rate 0.0003 and weight decay 0.0001. These are short screening runs, not the historical 80-epoch recipe. All candidates within each stage use the same budget and stopping rule. Checkpoint selection maximizes validation Mean Hit; ties minimize raw-label BCE. Candidate selection averages the two development intervals, then uses the worse interval as tie-breaker. The top two candidates and original model are checked at seeds 0,1,2. The winner receives seeds 3,4. The final reporting fits use the earlier 70% for training and the following interval to the historical report boundary for checkpoint selection.

Development scores are retrospective model-selection statistics, not an unbiased historical trading strategy: later development intervals participate in architecture selection. Final report rows exactly match the archived PIT results. This period was already inspected and must never be called an untouched test. Final architecture selection is saved before the reporting fits begin. Reporting results do not drive further architecture search.

The planned maximum is 54 neural experiment fits, plus one logistic baseline pipeline (10 currency-specific estimators) and a train-only majority estimate. The 250-step, 16-date smoke overfit is separate. Counts of attempts, failures, completed neural runs and baseline estimators must be disclosed separately. Checkpoints, prediction tables and epoch logs remain for every completed candidate; failed attempts remain in the ledger. No historical outputs are replaced.

## Data and time

Inputs are `fx_panel_pit.csv` and `nonfx_features_pit.csv`. At decision date t, the supplied forward target is the appreciation return from the noon fixing at t to the next panel fixing. Existing sequence code ends inputs at t−1 while taking the forward target on row t; new candidates end inputs at t. The original reference retains its older extra lag. Stored old `target_date` is the decision date; new outputs explicitly retain `decision_date` and use `Target_Date` as the actual return end date. Comparisons join by decision date and currency and assert equal realized targets.

This corrected alignment conditions on the noon FX observation. It is not evidence that the historical H.10 publication was available in real time or that execution at exactly the observed fixing was possible. A tradable deployment requires a timestamped live feed and execution/latency assumptions. Non-FX availability is governed by the supplied builder's 11:30 New York cutoff. Its source-clock assumptions and vintage dates are audited at code/artifact level, not independently certified against every historical release.

Each fold recomputes feature scalers, regime thresholds and target statistics using past rows only. One boundary target is purged so its return ends strictly before the next validation decision date. No target magnitude enters model inputs. Missing values retain the existing past-fitted imputation conventions. Known data breaks remain documented in `data/processed/method_breaks.csv`.

## Loss and outputs

Direction: u is a per-currency latent strength and z_i = u_i − u_USD, temperature 1. The unweighted loss is mean BCEWithLogits(z_i, 1[r_i>0]) across all training dates and ten non-USD currencies. No movement threshold or pair-specific intercept is used. All cross rates remain strength differences.

Dual head: a shared backbone also outputs v. Return prediction is sigma_train × (v_i − v_USD), with one common scalar sigma_train equal to pooled training-return standard deviation. Loss is BCE + lambda × mean Huber(v_i−v_USD, r_i/sigma_train; delta=1), lambda in {0.03,0.1}. This is scale-only normalization, without subtracting currency means. Direction Hit and return-head sign Hit/RMSE remain distinct. Pure BCE has no return RMSE because logits are not returns.

Historical reference: masked softplus(−z sign(y_standardized)) above training Q40, plus original graph and component penalties; legacy predictions are decoded with the historical per-currency mean and standard deviation. The aligned-active candidate retains the masked legacy loss and decoding but fixes input alignment. It is a diagnostic reference, not a corrected direction-label design.

Mean Hit keeps the historical three-way sign comparison, equally weighted across all dates and ten currencies. A zero target only matches an exactly zero prediction. BCE instead assigns zero targets to its non-positive class; the zero-target fraction is reported. No date/currency/confidence filtering improves the headline denominator.

Compact encoders share parameters across currencies and include currency embeddings and common market inputs. They use either GRU or flattened temporal MLP. Graph scores depend on input state through QK. The direct-plus-graph strength uses g(h)[q(h,m)−q(h,0)]; its correction is exactly zero without messages. The original-residual candidate applies the same cancellation idea to the original backbone. Signed top-k, dense signed attention, fixed soft threshold 0.05, static-plus-state scores, and a four-dimensional message are staged candidates. No extensive hyperparameter grid is used.

## Secondary metrics and uncertainty

Extreme uses training-only pooled absolute-return Q80. Stress scenarios use training-only Q85 for absolute VIX change, cross-currency realized forward-move magnitude, and upward US10Y change. Realized FX stress is an ex-post evaluation subset, not a predictive feature or an available trading signal. Stress extreme also uses the training-only Q80; this deliberately differs from the historical report's within-evaluation-scenario Q80. Secondary metrics must not be mixed as if the definitions were identical.

Graph intervention sets adjacency to zero at inference, holding parameters fixed. It measures dependence, not proof of benefit. A retrained no-message diagnostic is also evaluated for the winner. For the original backbone, zero adjacency may leave own-state processing in the relational branch; it is not identical to deleting that branch.

Paired circular moving-block bootstrap resamples dates with all ten currencies together. Default block length 20, sensitivity 5 and 60, 2,000 draws. The main contrast averages correctness across fitted seeds before time resampling. Thus intervals describe conditional time-sample uncertainty, not the seed distribution; seed standard deviations are reported separately. Development selection and repeated historical inspection are not corrected by these intervals. Archived baseline comparisons are contextual: their training endpoint and input alignment differ.

Favorable seed/year combinations are explicitly post hoc. The complete sorted table, including unfavorable rows, must accompany highlights; neither model selection nor headline results may replace the full report with these subsets. Partial years and denominator sizes must remain visible. No claim of significance follows merely from a favorable maximum.
