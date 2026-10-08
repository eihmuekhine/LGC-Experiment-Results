"""Shared, torch-free helpers for running the OFFICIAL TtBA repository (github.com/BUPTAIOC/TtBA) on a fixed image
list and recording per-image checkpoints in our per_checkpoint format (build 38).

Nothing here changes the TtBA algorithm.  The checkpoint recorder only observes the (queries, L2) records the
official ``Attacker.attack`` appends to ``self.L2_line`` after every iteration, together with the boundary point of
that iteration, so that the value at budget c is the state of the last completed iteration with queries <= c --
the same convention as the repo's ``OutResult.Summary`` (L2_LINE_sum[:, c]) and as our engine's checkpoints."""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

# official dataset name -> torchvision arch name (our --archs) -> README weight file
DATASETS = {"imagenet-vgg": "vgg19", "imagenet-resnet50": "resnet50", "imagenet-vit": "vit_b_32",
            "imagenet-inceptionv3": "inception_v3", "imagenet-efficient": "efficientnet_b0",
            "imagenet-densenet": "densenet161"}
WEIGHTS = {"vgg19": "vgg19-dcbb9e9d.pth", "resnet50": "resnet50-11ad3fa6.pth", "vit_b_32": "vit_b_32-d86f8d99.pth",
           "inception_v3": "inception_v3_google-0cc3c7bd.pth",
           "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth", "densenet161": "densenet161-8d451a50.pth"}
SPLIT_FIELDS = ["source", "source_label", "target", "target_label", "order", "protocol", "victim_weights",
                "loader_pos", "source_relpath", "target_relpath"]
CKPT_FIELDS = ["dataset", "arch", "mode", "image", "source_label", "target_label", "method", "space", "budget",
               "found", "state_query", "pix_l2", "pix_linf", "ssim", "lpips", "pix_l2_reported", "lpips_net",
               "seed_mode"]
RUN_FIELDS = ["dataset", "arch", "mode", "image", "method", "order", "status", "total_queries", "iterations",
              "wall_time_s", "success_eps", "acc_query", "final_l2_reported", "x0_pred_ok", "seed_mode"]
METHOD = "TtBA_GH"  # the official repository's attack, run by us


class CheckpointRecorder:
    """State at budget c = the last record (q, value, payload) with q <= c.

    ``on_record`` is called for every record in query order; a pending checkpoint c is closed with the PREVIOUS
    record as soon as a record with q > c arrives.  ``finish`` closes the rest with the last record."""

    def __init__(self, checkpoints: Sequence[int], init_value: float):
        self.pending: List[int] = sorted({int(c) for c in checkpoints})
        self.prev: Tuple[int, float, object] = (0, float(init_value), None)
        self.snap: Dict[int, Tuple[int, float, object]] = {}

    def on_record(self, q: int, value: float, payload) -> None:
        while self.pending and int(q) > self.pending[0]:
            self.snap[self.pending.pop(0)] = self.prev
        self.prev = (int(q), float(value), payload)

    def finish(self) -> Dict[int, Tuple[int, float, object]]:
        for c in self.pending:
            self.snap[c] = self.prev
        self.pending = []
        return self.snap


class HookList(list):
    """Drop-in for the Attacker's ``L2_line`` list: every append also feeds the owner's CheckpointRecorder with the
    boundary point of the iteration that produced it (``owner._last_bx``)."""

    def __init__(self, init, owner):
        super().__init__(init)
        self.owner = owner

    def append(self, item) -> None:  # noqa: D401
        super().append(item)
        self.owner.recorder.on_record(int(item[0]), float(item[1]), getattr(self.owner, "_last_bx", None))


def read_split_rows(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: str, fields: Sequence[str], rows: Sequence[dict]) -> None:
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def done_images(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["image"] for r in csv.DictReader(f) if r.get("status") == "ok"}
