# B5 attempt 1 — ABANDONED (2026-10-18)

- Instance: autodl-container-30d64fa2f6-699e3b58 (5 x NVIDIA GeForce RTX 5090, driver 580.105.08).
- Software: torch 2.8.0+cu128 / torchvision 0.23.0+cu128 / cuDNN 91002 (fresh container). env_build38.txt
  pre-registers torch 2.7.1+cu128 / torchvision 0.22.1+cu128.
- Launch: 2026-10-18 11:47 (bash run_b5_parallel.sh) at HEAD d2fcbd0; tag b5-go-2026-10-18 -> 999567a;
  reports/raw/B5/prompt.txt sha256 1fb343b4d0c9803655e35762867882a8dcd61020eee2e7aa9ae41a1bc381ea94;
  run_b5_parallel.sh sha256 c20cc34990e9ed2cf5fb1c8a18a79517d67617ba98884c13425461e3c21d17d2.
- Ended at 18:00 when H shut the instance down.
  Cells completed: 0 (reports/raw/B5/parallel.log held only its header).
- Looked at during the run: process list, GPU utilisation, directory sizes, row counts and the status / error
  columns of the two resnet50 TtBA runs files (24 rows each, status budget, error empty).
  Outcome metrics opened: none.
- Reason: launcher gate defects found by code review on 2026-10-18 (g5_check.py called with arguments it does not
  accept, so every TtBA / CGBA cell would have been marked failed and the D19 domain check skipped for every CGBA
  cell; LGC_RC_EGC under an X = 0 stop rule; 5 GPUs hard-coded; E17 output path differs from the contract), and the
  wall-clock on 5 GPUs. The torch 2.8.0 deviation, found afterwards, is a further reason not to use these data.
- Authorship: the 16 commits made on that instance (fdda03a..d2fcbd0) carry the placeholder git identity
  eihmuekhine@example.com. Labels "H-authorized" / "H (authorized)" in those commits and in the deviation table of
  reports/B5_host_probe.md were written by an assistant. Approved by H personally: none.
  Not approved by H (superseded by contract v1.9-rev7 and launcher v2): all 16 commits fdda03a..d2fcbd0 (placeholder
  git identity eihmuekhine@example.com); local contract rev7 with only the D16 note (D13/D14/D19 unchanged); the
  empty manifest (-maxdepth 2); the launcher edits (-j $B5_JOBS with the % 5 GPU formula; G5 draft with silent pass).
- Data: left on the attempt-1 instance only (Results/b5_*, reports/raw/B5/*.log); not copied; never used for any
  result or decision. Local commits made on that instance after d2fcbd0 were never pushed and are not part of the
  record.
- Attempt 2: pre-registered stack (torch 2.7.1+cu128 / torchvision 0.22.1+cu128 / cuDNN 90701; environment of
  instance autodl-container-fd9942a535-0cca2e31), launcher v2 after the B5-fix review, contract v1.9-rev7.