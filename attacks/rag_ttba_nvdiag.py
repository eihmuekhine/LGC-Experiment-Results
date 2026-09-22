
import copy
import math
import random
from itertools import combinations

import numpy as np
import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBANVNormalDiagnosticAttack(RAGTtBAPGRAttack):
    """
    Offline normal-quality diagnostic for the validated RAG-TtBA baseline.

    Purpose
    -------
    Test whether the stochastic hard-label normal estimate is stable enough
    to define a meaningful local (v, n_perp) XY plane.

    At selected BASE trajectory iterations, this diagnostic freezes the SAME
    boundary point z_b and estimates independent normals using several probe
    counts, e.g.:

        N = 30, 60, 120

    with multiple independent repeats for each N.

    It reports:

        1) same-N repeat cosine consistency;
        2) cross-N cosine consistency;
        3) angle phi = angle(v, n);
        4) alpha = v^T n = cos(phi);
        5) ||n - (v^T n)v||, the orthogonal residual used to build e_y.

    IMPORTANT
    ---------
    Diagnostic probes are OFFLINE and are NOT counted inside the 2000-query
    attack budget. A shallow copy of the attack object is used for each
    diagnostic normal estimate, and global RNG state is restored afterwards.

    Therefore the main RAG_TTBA_BASE trajectory should reproduce the validated
    reference result exactly/near-exactly.

    This is NOT a final attack method and should not be reported as a fair
    query-budget comparison.
    """

    def __init__(
        self,
        *args,
        normal_sizes=(30, 60, 120),
        normal_repeats=2,
        snapshot_iterations=(0, 3, 6, 12),
        diagnostic_seed=1992,
        **kwargs,
    ):
        # Force the validated baseline method in the parent.
        kwargs.pop(
            "attack_method",
            None,
        )

        super().__init__(
            *args,
            attack_method="RAG_TTBA_BASE",
            **kwargs,
        )

        self.normal_sizes = tuple(
            int(x)
            for x in normal_sizes
        )

        if any(
            x <= 0
            for x in self.normal_sizes
        ):
            raise ValueError(
                "All normal_sizes must be positive."
            )

        self.normal_repeats = int(
            normal_repeats
        )

        if self.normal_repeats < 2:
            raise ValueError(
                "normal_repeats must be >= 2 so consistency can be measured."
            )

        self.snapshot_iterations = tuple(
            sorted(
                {
                    int(x)
                    for x in snapshot_iterations
                    if int(x) >= 0
                }
            )
        )

        self.diagnostic_seed = int(
            diagnostic_seed
        )

        self.normal_diagnostics = []
        self.diagnostic_query_total = 0

    # ==================================================================
    # RNG isolation
    # ==================================================================
    @staticmethod
    def _capture_rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": None,
        }

        if torch.cuda.is_available():
            state[
                "torch_cuda"
            ] = torch.cuda.get_rng_state_all()

        return state

    @staticmethod
    def _restore_rng_state(
        state,
    ):
        random.setstate(
            state[
                "python"
            ]
        )

        np.random.set_state(
            state[
                "numpy"
            ]
        )

        torch.random.set_rng_state(
            state[
                "torch_cpu"
            ]
        )

        if (
            torch.cuda.is_available()
            and state[
                "torch_cuda"
            ] is not None
        ):
            torch.cuda.set_rng_state_all(
                state[
                    "torch_cuda"
                ]
            )

    @staticmethod
    def _seed_all(
        seed,
    ):
        random.seed(
            seed
        )

        np.random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

    # ==================================================================
    # Probe isolation
    # ==================================================================
    def _make_probe_copy(
        self,
        seed,
        q_budget,
    ):
        """
        Shallow-copy the attack so model/decoder tensors are shared read-only,
        while accounting containers are isolated and reset.

        estimate_normal() is the only method called on this probe.
        """
        probe = copy.copy(
            self
        )

        # Isolate commonly mutable containers.
        if hasattr(
            self,
            "q_breakdown",
        ):
            probe.q_breakdown = type(
                self.q_breakdown
            )(
                (
                    key,
                    0,
                )
                for key in self.q_breakdown.keys()
            )

        if hasattr(
            self,
            "history",
        ):
            probe.history = []

        if hasattr(
            self,
            "diagnostics",
        ):
            probe.diagnostics = []

        if hasattr(
            self,
            "normal_diagnostics",
        ):
            probe.normal_diagnostics = []

        # Reset common query counters if they are concrete numeric attributes.
        numeric_counter_names = (
            "all_queries",
            "q_num",
            "query_count",
            "queries",
            "num_queries",
            "initial_query_offset",
        )

        for name in numeric_counter_names:
            if (
                name in probe.__dict__
                and isinstance(
                    probe.__dict__[
                        name
                    ],
                    (
                        int,
                        float,
                    ),
                )
            ):
                probe.__dict__[
                    name
                ] = 0

        # Keep enough budget for exactly one diagnostic normal estimate.
        if "max_queries" in probe.__dict__:
            probe.max_queries = int(
                q_budget
                + 64
            )

        if "seed" in probe.__dict__:
            probe.seed = int(
                seed
            )

        return probe

    def _estimate_isolated_normal(
        self,
        z_boundary,
        q_count,
        seed,
    ):
        rng_state = self._capture_rng_state()

        try:
            self._seed_all(
                seed
            )

            probe = self._make_probe_copy(
                seed=seed,
                q_budget=q_count,
            )

            eta, q_used = probe.estimate_normal(
                z_boundary.detach(),
                int(
                    q_count
                ),
            )

        finally:
            self._restore_rng_state(
                rng_state
            )

        if eta is None:
            return None, int(
                q_used
                if "q_used" in locals()
                else 0
            )

        norm = torch.norm(
            eta
        )

        if (
            not torch.isfinite(
                norm
            )
            or norm.item()
            < 1e-12
        ):
            return None, int(
                q_used
            )

        n = (
            eta / norm
        ).detach()

        return n, int(
            q_used
        )

    # ==================================================================
    # Geometry helpers
    # ==================================================================
    @staticmethod
    def _safe_cos(
        a,
        b,
    ):
        denom = (
            torch.norm(
                a
            )
            * torch.norm(
                b
            )
        )

        if denom.item() < 1e-12:
            return float(
                "nan"
            )

        value = torch.sum(
            a * b
        ) / denom

        return float(
            torch.clamp(
                value,
                -1.0,
                1.0,
            ).item()
        )

    @staticmethod
    def _angle_deg_from_cos(
        cosine,
    ):
        if not math.isfinite(
            cosine
        ):
            return float(
                "nan"
            )

        cosine = max(
            -1.0,
            min(
                1.0,
                float(
                    cosine
                ),
            ),
        )

        return float(
            math.degrees(
                math.acos(
                    cosine
                )
            )
        )

    @staticmethod
    def _mean_unit(
        vectors,
    ):
        if not vectors:
            return None

        mean = torch.stack(
            vectors,
            dim=0,
        ).mean(
            dim=0
        )

        norm = torch.norm(
            mean
        )

        if (
            not torch.isfinite(
                norm
            )
            or norm.item()
            < 1e-12
        ):
            return None

        return (
            mean / norm
        ).detach()

    # ==================================================================
    # Snapshot diagnostic
    # ==================================================================
    def _run_normal_snapshot(
        self,
        z_boundary,
        iteration_index,
    ):
        v_raw = (
            z_boundary.detach()
            - self.z_source.detach()
        )

        v_norm = torch.norm(
            v_raw
        )

        if (
            not torch.isfinite(
                v_norm
            )
            or v_norm.item()
            < 1e-12
        ):
            snapshot = {
                "iteration": int(
                    iteration_index
                ),
                "query_at_snapshot": int(
                    self.all_queries
                ),
                "valid": False,
                "reason": "ZERO_V_VECTOR",
                "levels": [],
                "cross_size": [],
                "diagnostic_queries": 0,
            }

            self.normal_diagnostics.append(
                snapshot
            )

            return snapshot

        v = (
            v_raw / v_norm
        ).detach()

        snapshot = {
            "iteration": int(
                iteration_index
            ),
            "query_at_snapshot": int(
                self.all_queries
            ),
            "valid": True,
            "reason": "OK",
            "radius_latent": float(
                v_norm.item()
            ),
            "levels": [],
            "cross_size": [],
            "diagnostic_queries": 0,
        }

        mean_normals = {}

        for q_count in self.normal_sizes:
            repeat_rows = []
            normal_vectors = []

            for repeat_index in range(
                self.normal_repeats
            ):
                seed = int(
                    self.diagnostic_seed
                    + 100000
                    * int(
                        iteration_index
                    )
                    + 1000
                    * int(
                        q_count
                    )
                    + int(
                        repeat_index
                    )
                )

                n, q_used = self._estimate_isolated_normal(
                    z_boundary=z_boundary,
                    q_count=q_count,
                    seed=seed,
                )

                self.diagnostic_query_total += int(
                    q_used
                )

                snapshot[
                    "diagnostic_queries"
                ] += int(
                    q_used
                )

                if n is None:
                    repeat_rows.append(
                        {
                            "repeat": int(
                                repeat_index
                            ),
                            "seed": int(
                                seed
                            ),
                            "q_requested": int(
                                q_count
                            ),
                            "q_used": int(
                                q_used
                            ),
                            "valid": False,
                            "alpha_v_dot_n": None,
                            "phi_deg": None,
                            "n_perp_norm": None,
                        }
                    )

                    continue

                alpha = self._safe_cos(
                    v,
                    n,
                )

                phi_deg = self._angle_deg_from_cos(
                    alpha
                )

                n_perp = (
                    n
                    - alpha * v
                )

                n_perp_norm = float(
                    torch.norm(
                        n_perp
                    ).item()
                )

                repeat_rows.append(
                    {
                        "repeat": int(
                            repeat_index
                        ),
                        "seed": int(
                            seed
                        ),
                        "q_requested": int(
                            q_count
                        ),
                        "q_used": int(
                            q_used
                        ),
                        "valid": True,
                        "alpha_v_dot_n": float(
                            alpha
                        ),
                        "phi_deg": float(
                            phi_deg
                        ),
                        "n_perp_norm": float(
                            n_perp_norm
                        ),
                    }
                )

                normal_vectors.append(
                    n
                )

            pairwise = []

            for ia, ib in combinations(
                range(
                    len(
                        normal_vectors
                    )
                ),
                2,
            ):
                c = self._safe_cos(
                    normal_vectors[
                        ia
                    ],
                    normal_vectors[
                        ib
                    ],
                )

                pairwise.append(
                    {
                        "a": int(
                            ia
                        ),
                        "b": int(
                            ib
                        ),
                        "cos_signed": float(
                            c
                        ),
                        "cos_abs": float(
                            abs(
                                c
                            )
                        ),
                        "angle_deg": float(
                            self._angle_deg_from_cos(
                                c
                            )
                        ),
                    }
                )

            mean_n = self._mean_unit(
                normal_vectors
            )

            if mean_n is not None:
                mean_normals[
                    int(
                        q_count
                    )
                ] = mean_n

                mean_alpha = self._safe_cos(
                    v,
                    mean_n,
                )

                mean_phi = self._angle_deg_from_cos(
                    mean_alpha
                )

                mean_perp = (
                    mean_n
                    - mean_alpha * v
                )

                mean_perp_norm = float(
                    torch.norm(
                        mean_perp
                    ).item()
                )

            else:
                mean_alpha = float(
                    "nan"
                )
                mean_phi = float(
                    "nan"
                )
                mean_perp_norm = float(
                    "nan"
                )

            valid_phis = [
                row[
                    "phi_deg"
                ]
                for row in repeat_rows
                if row[
                    "valid"
                ]
            ]

            valid_alphas = [
                row[
                    "alpha_v_dot_n"
                ]
                for row in repeat_rows
                if row[
                    "valid"
                ]
            ]

            same_n_cos = [
                row[
                    "cos_signed"
                ]
                for row in pairwise
            ]

            level = {
                "N": int(
                    q_count
                ),
                "repeats": repeat_rows,
                "pairwise_repeat_cosines": pairwise,
                "valid_repeat_count": int(
                    len(
                        normal_vectors
                    )
                ),
                "same_N_cos_mean": (
                    float(
                        np.mean(
                            same_n_cos
                        )
                    )
                    if same_n_cos
                    else None
                ),
                "same_N_cos_min": (
                    float(
                        np.min(
                            same_n_cos
                        )
                    )
                    if same_n_cos
                    else None
                ),
                "phi_mean_deg": (
                    float(
                        np.mean(
                            valid_phis
                        )
                    )
                    if valid_phis
                    else None
                ),
                "phi_std_deg": (
                    float(
                        np.std(
                            valid_phis
                        )
                    )
                    if valid_phis
                    else None
                ),
                "alpha_mean": (
                    float(
                        np.mean(
                            valid_alphas
                        )
                    )
                    if valid_alphas
                    else None
                ),
                "mean_normal_alpha": (
                    None
                    if not math.isfinite(
                        mean_alpha
                    )
                    else float(
                        mean_alpha
                    )
                ),
                "mean_normal_phi_deg": (
                    None
                    if not math.isfinite(
                        mean_phi
                    )
                    else float(
                        mean_phi
                    )
                ),
                "mean_normal_perp_norm": (
                    None
                    if not math.isfinite(
                        mean_perp_norm
                    )
                    else float(
                        mean_perp_norm
                    )
                ),
            }

            snapshot[
                "levels"
            ].append(
                level
            )

        # Compare averaged directions between different query counts.
        available_sizes = sorted(
            mean_normals.keys()
        )

        for qa, qb in combinations(
            available_sizes,
            2,
        ):
            c = self._safe_cos(
                mean_normals[
                    qa
                ],
                mean_normals[
                    qb
                ],
            )

            snapshot[
                "cross_size"
            ].append(
                {
                    "N_a": int(
                        qa
                    ),
                    "N_b": int(
                        qb
                    ),
                    "cos_signed": float(
                        c
                    ),
                    "cos_abs": float(
                        abs(
                            c
                        )
                    ),
                    "angle_deg": float(
                        self._angle_deg_from_cos(
                            c
                        )
                    ),
                }
            )

        self.normal_diagnostics.append(
            snapshot
        )

        if self.verbose == "Yes":
            print(
                "\n"
                f"      [NV-DIAG pre-Iter{iteration_index}] "
                f"attackQ={self.all_queries} "
                f"latentRadius={v_norm.item():.6f}"
            )

            for level in snapshot[
                "levels"
            ]:
                same_cos = level[
                    "same_N_cos_mean"
                ]

                print(
                    "         "
                    f"N={level['N']:<4} "
                    f"sameNcos="
                    f"{'NA' if same_cos is None else f'{same_cos:.4f}'} "
                    f"phiMean="
                    f"{'NA' if level['phi_mean_deg'] is None else f'{level['phi_mean_deg']:.2f}'}deg "
                    f"phiStd="
                    f"{'NA' if level['phi_std_deg'] is None else f'{level['phi_std_deg']:.2f}'} "
                    f"alphaMean="
                    f"{'NA' if level['alpha_mean'] is None else f'{level['alpha_mean']:.4f}'}"
                )

            for row in snapshot[
                "cross_size"
            ]:
                print(
                    "         "
                    f"cross N={row['N_a']}->{row['N_b']} "
                    f"cos={row['cos_signed']:.4f} "
                    f"angle={row['angle_deg']:.2f}deg"
                )

            print(
                "         "
                f"offlineDiagnosticQueries="
                f"{snapshot['diagnostic_queries']}"
            )

        return snapshot

    # ==================================================================
    # BASE trajectory + diagnostics
    # ==================================================================
    def attack(
        self,
    ):
        z_b = self._initialize_boundary()

        if z_b is None:
            return {
                "success": False,
                "final_z": self.z_source.detach(),
                "final": None,
                "checkpoints": self.checkpoint_summary(),
                "thresholds": self.threshold_summary(),
                "query_breakdown": dict(
                    self.q_breakdown
                ),
                "diagnostics": self.diagnostics,
                "normal_diagnostics": self.normal_diagnostics,
                "diagnostic_query_total": int(
                    self.diagnostic_query_total
                ),
            }

        self._record(
            z_b,
            "Initial",
        )

        for i in range(
            self.iteration
        ):
            if i in self.snapshot_iterations:
                self._run_normal_snapshot(
                    z_boundary=z_b,
                    iteration_index=i,
                )

            if (
                self.remaining_queries()
                <= self.update_query_reserve
            ):
                break

            q_full = int(
                self.N0
                * math.sqrt(
                    i + 1.0
                )
            )

            q_full = max(
                1,
                q_full,
            )

            q_full = min(
                q_full,
                max(
                    0,
                    self.remaining_queries()
                    - self.update_query_reserve,
                ),
            )

            if q_full <= 0:
                break

            eta, q_used = self.estimate_normal(
                z_b,
                q_full,
            )

            if eta is None:
                break

            before_l2 = float(
                self.rag_pixel_l2(
                    z_b
                )
            )

            z_next, bridge_info = self.step_rag_ttba(
                self.z_source,
                eta,
                z_b,
            )

            after_l2 = float(
                self.rag_pixel_l2(
                    z_next
                )
            )

            z_b = z_next.detach()

            self._record(
                z_b,
                f"Iter{i}",
            )

            self.diagnostics.append(
                {
                    "iteration": int(
                        i
                    ),
                    "query": int(
                        self.all_queries
                    ),
                    "normal_q_full": int(
                        q_full
                    ),
                    "normal_q_used": int(
                        q_used
                    ),
                    "before_pixel_l2": float(
                        before_l2
                    ),
                    "after_pixel_l2": float(
                        after_l2
                    ),
                    "bridge_L": (
                        None
                        if bridge_info is None
                        else float(
                            bridge_info[
                                "L"
                            ]
                        )
                    ),
                    "bridge_k_app": (
                        None
                        if bridge_info is None
                        else float(
                            bridge_info[
                                "k_app"
                            ]
                        )
                    ),
                }
            )

        best = min(
            self.history,
            key=lambda row: row[
                "l2_pix"
            ],
        )

        final = {
            "query": int(
                best[
                    "query"
                ]
            ),
            "budget_consumed": int(
                self.all_queries
            ),
            "l2_lat": float(
                best[
                    "l2_lat"
                ]
            ),
            "l2_pix": float(
                best[
                    "l2_pix"
                ]
            ),
            "ssim": float(
                best[
                    "ssim"
                ]
            ),
            "lpips": float(
                best[
                    "lpips"
                ]
            ),
            "asr": 1.0,
        }

        return {
            "success": True,
            "final_z": z_b.detach(),
            "final": final,
            "checkpoints": self.checkpoint_summary(),
            "thresholds": self.threshold_summary(),
            "query_breakdown": dict(
                self.q_breakdown
            ),
            "diagnostics": self.diagnostics,
            "normal_diagnostics": self.normal_diagnostics,
            "diagnostic_query_total": int(
                self.diagnostic_query_total
            ),
        }
