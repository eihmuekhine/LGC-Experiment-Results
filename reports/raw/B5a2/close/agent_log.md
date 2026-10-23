# agent_log — B5a2-close

| time | task | command | exit | output |
|---|---|---|---|---|
| 08:36:31 | C0 | pgrep -af "run_b5_parallel\|parallel -j\|run_benchmark\|run_ttba\|run_cgba\|run_arsopt" | 0 | only self-match (no attack process) |
| 08:37:36 | C1 | python /tmp/b5finish_probe/build_c1.py | 0 | 52-row C1 table (all cells reached n; error_rows=0; nonfinite=0) |
| 08:38:07 | C2 | python /tmp/b5finish_probe/make_alias.py | 0 | wrote reports/raw/B5a2/close/g5/ttba_table1_alias.json |
| 08:38:16 | C2 | python /tmp/b5finish_probe/verify_alias.py | 0 | only CGBA_H added; CGBA_H==CGBA-H; sha256 shipped=2cb224f6... alias=5532b173... |
| 08:38:34 | C2 | python tools/eval/g5_check.py --arch resnet50 --mode non_targeted --ttba_root Results/b5_main_TtBA_resnet50_NT --cgba_root Results/b5_main_CGBA_resnet50_NT | 0 | VERDICT: PASS |
| 08:38:48 | C2 | g5_check.py --arch vgg19 --mode non_targeted (TtBA vgg19 NT + CGBA vgg19 NT) | 0 | VERDICT: PASS |
| 08:38:48 | C2 | g5_check.py --arch vit_b_32 --mode non_targeted (TtBA vit_b_32 NT + CGBA vit_b_32 NT) | 0 | VERDICT: PASS |
| 08:39:02 | C2 | g5_check.py --arch resnet50 --mode targeted --table1 .../ttba_table1_alias.json (TtBA resnet50 T + CGBA_H resnet50 T) | 0 | VERDICT: PASS |
| 08:39:02 | C2 | g5_check.py --arch vgg19 --mode targeted --table1 alias (TtBA vgg19 T + CGBA_H vgg19 T) | 0 | VERDICT: PASS |
| 08:39:03 | C2 | g5_check.py --arch vit_b_32 --mode targeted --table1 alias (TtBA vit_b_32 T + CGBA_H vit_b_32 T) | 0 | VERDICT: PASS |
| 08:39:59 | C4 | python tools/eval/human_preflight.py --roots <52 dirs> --out reports/human_preflight_b5a2 --require_host | 0 | wrote reports/human_preflight_b5a2.md; FLAG lines = 53 |
| 08:46:27 | C5 | python tools/eval/check_contract.py --contract contract/paper_contract.yaml --stage partial --roots <48 dirs> --column_map contract/column_map.yaml --split_dir splits/imagenet/ttba_github splits/places365 splits/robust --report reports/raw/B5a2/close/contract_partial.md --status_out reports/raw/B5a2/close/contract_partial_status.md | 0 | all I01-I12 PASS (count=0), exit 0 |
| 08:46:51 | C6 | python /tmp/b5finish_probe/make_e17.py | 0 | wrote reports/E17_sources.md (1212 files) |
