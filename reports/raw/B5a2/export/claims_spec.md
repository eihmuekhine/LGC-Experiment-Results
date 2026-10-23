# B5a2 claims specification (B3a) — verbatim quotes

Source files and sha256:
- `prereg_build38.txt` sha256 `0f2e9039e3e2e843092f3f121a18c37f7c37d15424b428a08bb0d4354651ed27`
- `prereg_build38_amendment2.txt` sha256 `178809f98c1be220f1fefe0e938cc126b042b0a9209c70c3e9b575caa9fd2c58`

## prereg_build38.txt, section 6 (verbatim)

```
6. CLAIMS (confirm_ttba.py; method fixed; a failed claim is reported and dropped, never fixed):
   Holm family (T1 x 3 victims, P1 x 3 victims x 2 modes):
     T1 targeted : L2 AUC ratio EGC / TtBA_GH < 1 and upper 95% bound < 1.
     P1 both     : LPIPS AUC delta (EGC - TtBA_GH) < 0 and upper bound < 0.
   CI-only (non-inferiority):
     T2 targeted     : L2@10k ratio upper bound <= 1.03.
     N1 non-targeted : L2 AUC ratio and L2@10k ratio, upper bound <= 1.15.
   AUC = per-image mean of log L2 over budgets 500/1000/2500/5000/10000 (same as confirm_37; the 2000 checkpoint
   is recorded only for the descriptive comparison with the TtBA paper's Table 1).
   Ratios = paired geometric means; CI = image bootstrap (B = 10000); test = one-sided Wilcoxon signed-rank.
   Table-1 numbers of the TtBA paper are cited, not paired (different list and environment).
```

## prereg_build38_amendment2.txt, section 7 (verbatim)

```
7. Claims (paired, n <= 200 per cell; paired geometric-mean ratio or mean difference; image bootstrap B = 10000;
   one-sided Wilcoxon; Holm within each family).
   - prereg section 6 claims (T1, T2, N1, P1 vs official TtBA): unchanged.
   - New family P2 (Holm over baselines x victims x modes):
     LPIPS-vgg AUC delta (EGC - baseline) < 0, upper 95% bound < 0, for baseline in
     {TtBA, CGBA(NT), CGBA-H(T), ARS_OPT}.
   - Supporting (CI only, not in a Holm family): the same as P2 for LPIPS-alex and DISTS; perceptual ASR AUC.
   - Failure policy: a failed claim is reported and dropped. The method is never changed.
```

## prereg_build38_amendment2.txt, section 2 (verbatim; dev overlap)

```
2. Data. Unchanged protocol (prereg sections 3-4).
   - Evaluation set per victim and mode: rows 1-200 of splits/imagenet/ttba_github/{arch}_{mode}.csv. Targeted: the
     first 200 source-target pairs.
   - sha256 of the files: prereg lines 56-62. Pool-list hash: 0d480e34b1a62985c0d51fe78d51731b886fd91e83b3e7bc4e8ea9834e209f71.
   - Dev / selection / confirmation images (seeds 992, 2027, 3141, 2718) are KEPT (prereg section 7).
     - Rows affected in rows 1-200: NT 6 / 4 / 2, T 7 / 6 / 4 (VGG19 / ResNet-50 / ViT-B/32).
     - Every claim is recomputed without these rows (descriptive sensitivity analysis).
   - The permutation is per victim (official main.py). There are no shared images across all three victims.
```

## Contract constants used (contract/paper_contract.yaml, v1.9-rev7)

```
constants.auc_definition = "per-image mean of log(pix_l2) over auc_budgets (L2); mean of the metric over auc_budgets (LPIPS, DISTS, SSIM)"
constants.auc_budgets = [500, 1000, 2500, 5000, 10000]
constants.stats = {bootstrap_B: 10000, ci: 0.95, test: wilcoxon_signed_rank, claims_sided: one, descriptive_sided: two, multiplicity: holm}
constants.dev_overlap_rows_1_200 = {vgg19: {non_targeted: 6, targeted: 7}, resnet50: {non_targeted: 4, targeted: 6}, vit_b_32: {non_targeted: 2, targeted: 4}}
```

## Effect-statistic convention

The frozen registered implementation `confirm_ttba.py` (prereg section 6, "method fixed") computes:
- L2 ratio: `ratio = exp(mean(log o - log b))`, CI = exp(bootstrap percentile CI of the mean log-ratio)
  (`l2_summary` in analyze_34b.py: `ratio = math.exp(d.mean()); ci = boot_ci(d, np.mean)`).
- LPIPS AUC delta: median paired difference, CI = bootstrap percentile CI of the median
  (`confirm_ttba.py`: `boot_ci(dl, np.median)`; `delta = np.median(dl)`).
One-sided Wilcoxon signed-rank, alternative "less" (d = log(o/b) for L2, d = o - b for LPIPS/DISTS).

These conventions are followed by `tools/eval/claims_registered.py` (written in this session, see its docstring).
