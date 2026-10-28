# Can a cheap probability correction replace a coefficient refit?

Declared 28 October 2026 before fitting or scoring any study47 model. This is an independent research example, not a deployment claim or new calibration algorithm.

## Decision and source boundary

When a score trained under one event prevalence is used in a rare-event review queue, can a probability adjustment preserve useful scoring and decisions without refitting every feature coefficient? Compare unchanged scores, known-prior odds correction, intercept recalibration, slope/intercept recalibration, a fixed-penalty coefficient refit and a constant known-prior baseline. Positive-rate changes are controlled scenarios, not observed production drift. A complete mixed or negative comparison meets the research objective.

Use exact active OpenML numerical benchmark versions MAGIC44125 and MiniBooNE44128 from suite337. Preserve metadata, exact download queries, retrieval times and source hashes. MAGIC is Monte Carlo simulation, not measured telescope deployment. MiniBooNE is a public particle-identification benchmark derived from an experiment; this analysis acquires no new physical measurements. OpenML metadata says Public and CC0 respectively; preserve the original UCI CC BY4.0 attribution boundary for both rather than infer that an uploader waived another creator's rights. Credit R. Bock, MAGIC Gamma Telescope, DOI10.24432/C52C8B; Byron Roe, MiniBooNE particle identification, DOI10.24432/C5QC87. Source papers and unpublished manuscript drafts remain local, outside every Git repository.

The original MAGIC description explicitly warns that background events are underrepresented and that accuracy alone is unsuitable. This motivates a prior-sensitivity experiment; it does not supply a measured deployment prevalence. The MiniBooNE original file is ordered by class, so row-order/time splits would be misleading. Use whole identical-feature groups and do not infer missing acquisition/run identities. Encoded positive labels are literal `g` for MAGIC and `True` for MiniBooNE; negatives are `h` and `False`. Validate both exact label vocabularies. ParticleID_n columns are supplied measurement descriptors, not inferred unique event IDs.

Parse numeric features, excluding only target and metadata-declared row-ID/ignore attributes. Require finite features and exact binary labels; do not silently impute or reinterpret extreme numbers. Deduplicate exact full-row values while acknowledging altered frequencies and not asserting that repeated events were erroneous. Canonical little-endian float64 predictor vectors normalize signed zero and missing values for group signatures. Select complete feature groups by signature order until the next group would exceed8,000 rows. Retain selected original row IDs, exact values, exclusion counts and labels. These are bounded benchmark derivatives, not representative deployment samples.

For seeds101,202,303,404,505, use SHA256(group_signature+colon+seed), first8hex modulo20: buckets0--9 fit,10--12 validation,13--15 calibration,16--19 test. Require at least20 observations of each class in every partition. Identical predictors stay together even when their labels disagree. No target-dependent sampling. Unknown run dependence and overlapping seed splits remain limitations; do not pool them as independent experiments.

## Base model, objective and optimization evidence

Use an original ridge-logistic implementation with SciPy L-BFGS-B, analytic gradient and single-thread linear algebra. With standardized features and intercept, minimize weighted mean binary logistic loss plus alpha/2 times the sum of squared feature coefficients; the intercept is unpenalized. The population feature means and SDs are fitted on fitting rows only, constant-channel scale=1. No test/calibration values enter base preprocessing or hyperparameter selection.

Finite alpha grid:1e-4,1e-3,1e-2,1e-1,1,10. Choose minimum unweighted validation log loss, ties by smaller alpha. Export every configuration, coefficient vector, scaler, prediction, objective, gradient evidence, iterations, status, runtime and both numerical starts. Refit the selected base model and scaler on fit+validation only. This is supervised fitting from scratch, not pretrained-model fine-tuning. Boundary winners and finite-grid limits must remain visible.

For each unconstrained ridge-logistic fit, solve from zero and a deterministic second start (feature coefficients0.01, intercept=logit empirical fitting prevalence). Use ftol1e-14,gtol1e-9,maxiter5,000,maxls50. Retain both attempts. Require successful numerical status, independently recomputed infinity-norm gradient<=1e-6, objective agreement between starts within1e-8 absolute/1e-8 relative and a positive-semidefinite Hessian within1e-10 numerical tolerance. Do not select a nonconverged fit or hide a failed attempt. If a source/method defect needs correction, preserve its failed receipt and distinguish any later reused test.

The logistic objective is convex because nonnegative observation weights multiply positive-semidefinite outer products, and alpha is nonnegative. Numerical stationarity/KKT residuals are optimization certificates at declared tolerance, not exact-arithmetic proof or statistical generalization guarantees. Independent checks must reconstruct objective/gradient/Hessian and directional finite differences without importing generation metric helpers.

## Controlled prior scenarios and adaptation

Target positive prevalences q={0.01,0.05,0.20,0.50} are declared experimental inputs. Within any fitting or evaluation partition, positive observations receive weight q/n_positive and negatives (1-q)/n_negative, so weights sum to1 and the weighted class proportion is exactly q. This changes the empirical class mixture while holding its within-class values fixed. It assumes label shift and a known target prevalence; it does not test estimating an unknown prevalence or covariate/concept shift.

The six methods are:

1. Unchanged base logit.
2. Prior-odds logit: base_logit+logit(q)-logit(base fit+validation prevalence). The Bayes correction presumes calibrated source probabilities and stable class-conditionals; it is not guaranteed merely by running the formula.
3. Intercept recalibration: add b to the frozen base logit, minimizing weighted calibration log loss over b in[-20,20]. Use a bracketed derivative root when interior; otherwise the appropriate boundary. Check scalar convex KKT conditions.
4. Slope/intercept recalibration: s*base_logit+b with s in[0,5],b in[-20,20], minimizing the same weighted calibration loss without a penalty. Run L-BFGS-B from(s,b)=(1,0),(1,odds_offset),(0.5,0). Retain all starts, objective agreement and independent box-KKT/Hessian checks. Bounds are declared engineering limits, not tuned findings. Report active boundaries; do not extend them against test results.
5. Coefficient refit: refit all logistic coefficients on fit+validation+calibration with weights for q, the frozen selected alpha and the frozen base fit+validation scaler. This uses the same available labels as the base-plus-calibrator workflow but does not retune regularization under the target mixture. Call it a fixed-penalty refit, not the best possible fully retuned system.
6. Constant probability q. This known-prior reference establishes whether feature scoring earns improvement beyond the supplied prior itself.

Adaptation never sees test labels or test feature statistics. Freeze base selection and all fitted calibrators/refits before scoring test. Test labels are used only to calculate the declared controlled-mixture weights and outcomes. Do not change q, penalties, calibrator bounds or decision thresholds after inspection.

## Metrics, decisions and uncertainty

Primary score: q-weighted test log loss from finite logits, using stable log-sum-exp. Also report weighted Brier score (probability MSE), weighted mean predicted probability and its deviation from q, plus original unweighted class counts. Retain each observation's true label, logit, probability, source row/group, method, scenario and weight. Do not use raw classification accuracy as the headline.

For a declared hypothetical false-positive cost10 and false-negative cost1, flag when probability>=10/11. Report weighted TP,FP,TN,FN, precision (null if no alerts), recall, FPR, review fraction and expected loss10*FP+FN per scenario observation. These are analytical loss units, not money, real detector costs or realized impact. Proper-score improvement need not improve this decision; retain that distinction.

For the single prespecified rare-event comparison q=0.01, bootstrap the paired log-loss difference prior-odds minus fixed-penalty refit with1,000 whole-feature-group draws per dataset/seed. Sample test groups with replacement, retaining all rows of each group; then recompute positive and negative class means and mix them with q. Seed=20261028+dataset_id+split_seed. Retain ordered group IDs, RNG/version, draw-index digest, class counts and every difference. Independently reconstruct every draw and percentile95% interval. Intervals are conditional on these benchmark groups and known-prior scenarios, not independent intervals across overlapping split seeds or evidence of real production drift.

## Release gates

Reconstruct all source panels/labels and group splits independently, using SQL for group/partition/label-count accounting. Independently verify every alpha validation loss, selected model, standardization statistic, exported prediction, adaptation objective, numerical stationarity/KKT residual, test metric, confusion matrix, bootstrap draw and reported summary. Contracts must cover held-target isolation, exact prior-weight sums, odds identity at unchanged prior, stable extreme logits, optimizer gradient/constraint logic and whole-group resampling.

Run a complete fresh temporary-directory refit/replay with pinned dependencies, comparing required selectors and numerical outputs at1e-7absolute/1e-9relative tolerance. Timings are actual measured outputs but not equality targets. Inspect decision-useful figures with visible controlled-scenario/simulation labels and a private LaTeX/PDF. Review intended public files for privacy, source rights and absent manuscripts/credentials. Require clean exact-head Linux replay before completion. No paid compute or fabricated production claims.
