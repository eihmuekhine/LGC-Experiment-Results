#!/usr/bin/env bash
# Isolated reviewer-calibration workspace (no path to the post-B4d state). Run on AutoDL as the human (H).
set -euo pipefail
FROZEN=/root/autodl-tmp/LGC_frozen
MAIN=/root/autodl-tmp/Latent-Geometric-Chords
WS=/root/autodl-tmp/calib_ws
KIT="$(cd "$(dirname "$0")" && pwd)"
[ -d /root/autodl-tmp/calib_b4d ] || git -C "$FROZEN" worktree add /root/autodl-tmp/calib_b4d 88a7443
git -C /root/autodl-tmp/calib_b4d rev-parse --short HEAD
mkdir -p "$WS/.opencode/skills" "$WS/out"
cp -r "$MAIN/.opencode/skills/lgc-preflight" "$WS/.opencode/skills/"
cp "$MAIN/B4d.md" "$WS/B4d_transcript.md"
cp "$KIT/spec_B4d.txt" "$WS/spec_B4d.txt"
python3 - "$WS/opencode.json" "$MAIN/opencode.json" <<'PY'
import json, os, sys
perm = {
  "edit": {"*": "deny", "out/*": "allow"},
  "webfetch": "deny", "websearch": "deny",
  "external_directory": {"*": "deny",
                         "/root/autodl-tmp/calib_b4d/**": "allow",
                         "/root/autodl-tmp/baselines/**": "allow",
                         "/root/autodl-tmp/TtBA/**": "allow"},
  "bash": {"*": "ask",
           "git -C /root/autodl-tmp/calib_b4d *": "allow", "ls*": "allow", "cat *": "allow", "head *": "allow",
           "tail *": "allow", "sed -n *": "allow", "grep *": "allow", "rg *": "allow", "wc *": "allow",
           "sha256sum *": "allow", "find *": "allow",
           "*LGC_frozen*": "deny", "*Latent-Geometric-Chords*": "deny",
           "rm *": "deny", "git checkout*": "deny", "git reset*": "deny", "git commit*": "deny", "git push*": "deny"},
}
models = {"cal_glm": "openrouter/z-ai/glm-5.3",
          "cal_qwen": "openrouter/qwen/qwen3.8-max-0902",
          "cal_kimi": "openrouter/moonshotai/kimi-k3"}
cfg = {"$schema": "https://opencode.ai/config.json",
       "agent": {k: {"description": f"Calibration reviewer ({m})", "mode": "primary", "model": m, "permission": perm}
                 for k, m in models.items()}}
if os.path.exists(sys.argv[2]):
    main = json.load(open(sys.argv[2]))
    if "provider" in main:
        cfg["provider"] = main["provider"]   # provider pinning only; keys live in /connect storage, not here
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1], "| provider block copied:", "provider" in cfg)
PY
echo "Workspace ready: cd $WS && opencode ; then /new and Tab to cal_qwen / cal_kimi / cal_glm."
echo "Keys come from OpenCode /connect storage; nothing secret is written here."
