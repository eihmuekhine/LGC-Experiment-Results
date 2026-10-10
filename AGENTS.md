# Project rules for coding agents (OpenCode) — LGC-EGC paper runs (kit v5)

Before touching experiments, baselines, splits, metrics, tables, figures or result files, load these skills:
- `lgc-reviewer-eval`: protocol, splits, metrics, tables.
- `lgc-baselines`: official baseline code, adapters, fairness gates.

The skill `attack-eval-protocol` and the evalkit `AGENTS.md` are superseded (archived in `docs/archive/`).

Short form:
1. **Hard-label only.** No logits, gradients or surrogates; no white-box initialisation; decision-only tracker.
2. **Same rows for every method.** Rows 1–200 of `splits/imagenet/ttba_github/{arch}_{mode}.csv`.
   - Our runner always gets `--split_csv … --num_samples 200`.
   - Dev-seed images are kept (prereg §7).
3. **Count every query.** Budget 10000. Checkpoints are read from `contract/paper_contract.yaml`: 100 200 500 1000
   2000 2500 5000 10000 (v1.1). Never hard-code another list.
4. **Frozen method.** Engine sha `835f1933d9e5`, run from `/root/autodl-tmp/LGC_frozen`. Never edit the engine.
5. **Official baseline code, default hyper-parameters.** Only the allowed adapter changes.
6. **Store adversarial images** (`--save_adv` / advstore) and compute metrics offline. Never re-run an attack for a
   metric.
7. **Never write or edit** `prereg_build38*.txt`. Never open n = 200 results before `build38-amend2` is tagged.
8. **After any change in `tools/eval/`**, run `python tools/eval/selftest_evalkit.py`; it must print ALL PASS.
9. **Never invent** citations, URLs, API names, hyper-parameters or numbers. Read the code or write UNVERIFIED.
10. **Ask before** installing packages, starting GPU runs longer than a smoke test, or deleting files.
