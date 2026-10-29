# E9 export (prereg_build38_amendment3.txt), 2026-10-29

HEAD at finish: 2a0799525a03f65be485370a7c2dec9809cf80be
Attack command: run_e9_parallel.sh run_attack() (B5 LGC_RC_EGC command line; only --methods and --out_dir differ).
LGC_RC_EGC rows: paper_exports/b5a2 (tag b5a2-done-2026-10-23), not re-run.

Tool sha256:
    5b28622a007d5cee0209ad44fd7c4fc1b616467159878b5239ccc4d55bf2bb51  prereg_build38_amendment3.txt
    278fa8706186c43b69e0e9a521144aeee83058c3bb27d8a1eb2802ab28948411  run_e9_parallel.sh
    b30addf1662ae9c1cef489e4d071981839bc67832dda36d5672326c880f5477f  e9_finish.sh
    19e0ab4f9403fe86ff26625549b76136da599c0b3d4de519077cb88ba29deb98  tools/eval/e9_gate.py
    ff75de4479d4b566b1003f6b64efea88512c5e64881814f2aa1d5bc3d6ec19e3  tools/eval/claims_e9.py
    e5e2ab10bb93448f527a79ca0df362af03071c539bbcb4a9ab664d18d4998cf0  tools/eval/claims_registered.py
    4f83267714d2ea07e6f31c1fc3a8415ec163382614d81aec82933ad420c2bbe3  tools/eval/eval_metrics_offline.py
    835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a  lgc_bench/engine.py
    9c62820e5fafbac7e8c3f2cf3a688e3e6e7c53f3c5fc7f709ef5f34d17861873  run_benchmark_33b.py

Commands (per cell): python tools/eval/eval_metrics_offline.py --roots Results/<cell> --out paper_exports/e9/metrics/<cell>/metrics.csv --device cuda:0 --batch 32
Claims: python tools/eval/claims_e9.py ; python tools/eval/claims_e9.py --no-devoverlap
