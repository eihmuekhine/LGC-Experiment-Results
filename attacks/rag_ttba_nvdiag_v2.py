
import copy
import math
import random
from itertools import combinations

import numpy as np
import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBANVNormalDiagnosticV2Attack(RAGTtBAPGRAttack):
    """
    Offline normal-estimator diagnostic v2 for the validated RAG-TtBA baseline.

    This class DOES NOT replace the attack normal estimator.  It keeps the
    validated BASE attack trajectory unchanged and, at selected boundary states,
    performs isolated OFFLINE diagnostics comparing:

      (A) ONE_SIDED_PARENT
          The repository's current production estimator, called through
          self.estimate_normal() on a probe copy.

      (B) PAIRED_ANTITHETIC
          A strict hard-label antithetic estimator using the same total victim
          query budget:

              u_j ~ N(0, I)

              s_plus  = phi(z_b + sigma * u_j)
              s_minus = phi(z_b - sigma * u_j)

              eta = sum_j (s_plus - s_minus) * u_j

          where phi is +1 for adversarial and -1 for clean.

    Fair query-budget convention
    ----------------------------
    A diagnostic budget B means B victim-model queries for BOTH estimators.

      ONE_SIDED_PARENT:
          B one-sided queries.

      PAIRED_ANTITHETIC:
          B/2 antithetic direction pairs = B queries.

    Therefore normal-query-budget comparisons are query-fair.

    For each estimator / sigma / budget / repeat the diagnostic reports:
      * signed and absolute repeat cosine consistency;
      * phi = angle(v, n);
      * alpha = v^T n;
      * ||n - alpha v||;
      * raw eta norm;
      * for paired estimator: informative-pair rate
        (fraction of +/- pairs with opposite hard labels).

    It also reports:
      * cross-budget consistency at fixed estimator/sigma;
      * cross-sigma consistency at fixed estimator/budget;
      * paired-vs-one-sided cosine at matched sigma/budget.

    IMPORTANT
    ---------
    All diagnostic victim queries are OFFLINE.  They are not included in the
    2000-query attack budget.  The global RNG state is restored after every
    diagnostic estimate, so the main BASE trajectory should remain unchanged.
    """

    ONE_SIDED = "ONE_SIDED_PARENT"
    PAIRED = "PAIRED_ANTITHETIC"

    def __init__(
        self,
        *args,
        diagnostic_sigmas=(2e-4, 5e-4, 1e-3, 2e-3),
        diagnostic_query_budgets=(60, 120, 240),
        normal_repeats=2,
        snapshot_iterations=(0,),
        diagnostic_seed=2992,
        **kwargs,
    ):
        kwargs.pop("attack_method", None)

        super().__init__(
            *args,
            attack_method="RAG_TTBA_BASE",
            **kwargs,
        )

        self.diagnostic_sigmas = tuple(
            float(x) for x in diagnostic_sigmas
        )
        if any(x <= 0.0 for x in self.diagnostic_sigmas):
            raise ValueError("All diagnostic_sigmas must be > 0.")

        self.diagnostic_query_budgets = tuple(
            int(x) for x in diagnostic_query_budgets
        )
        if any(x <= 1 for x in self.diagnostic_query_budgets):
            raise ValueError(
                "All diagnostic_query_budgets must be >= 2."
            )

        self.normal_repeats = int(normal_repeats)
        if self.normal_repeats < 2:
            raise ValueError(
                "normal_repeats must be >= 2."
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

        self.diagnostic_seed = int(diagnostic_seed)

        self.normal_diagnostics_v2 = []
        self.diagnostic_query_total = 0

    # ------------------------------------------------------------------
    # RNG isolation
    # ------------------------------------------------------------------
    @staticmethod
    def _capture_rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": None,
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch_cpu"])

        if (
            torch.cuda.is_available()
            and state["torch_cuda"] is not None
        ):
            torch.cuda.set_rng_state_all(
                state["torch_cuda"]
            )

    @staticmethod
    def _seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Probe isolation
    # ------------------------------------------------------------------
    def _make_probe_copy(
        self,
        seed,
        query_budget,
        sigma,
    ):
        probe = copy.copy(self)

        # Isolate common mutable accounting containers.
        if hasattr(self, "q_breakdown"):
            try:
                probe.q_breakdown = type(self.q_breakdown)(
                    (key, 0)
                    for key in self.q_breakdown.keys()
                )
            except Exception:
                probe.q_breakdown = {
                    key: 0
                    for key in self.q_breakdown.keys()
                }

        if hasattr(self, "history"):
            probe.history = []

        if hasattr(self, "diagnostics"):
            probe.diagnostics = []

        if hasattr(self, "normal_diagnostics_v2"):
            probe.normal_diagnostics_v2 = []

        # Reset concrete query counters.
        for name in (
            "all_queries",
            "q_num",
            "query_count",
            "queries",
            "num_queries",
            "initial_query_offset",
        ):
            if (
                name in probe.__dict__
                and isinstance(
                    probe.__dict__[name],
                    (int, float),
                )
            ):
                probe.__dict__[name] = 0

        if "max_queries" in probe.__dict__:
            probe.max_queries = int(
                query_budget + 64
            )

        if "seed" in probe.__dict__:
            probe.seed = int(seed)

        # Critical: diagnostic sigma is changed ONLY on the probe copy.
        probe.sigma = float(sigma)

        return probe

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_unit(x):
        norm = torch.norm(x)
        if (
            not torch.isfinite(norm)
            or norm.item() < 1e-12
        ):
            return None, float(norm.item())
        return (x / norm).detach(), float(norm.item())

    @staticmethod
    def _safe_cos(a, b):
        denom = torch.norm(a) * torch.norm(b)
        if denom.item() < 1e-12:
            return float("nan")

        value = torch.sum(a * b) / denom
        value = torch.clamp(value, -1.0, 1.0)

        return float(value.item())

    @staticmethod
    def _angle_deg_from_cos(cosine):
        if not math.isfinite(cosine):
            return float("nan")

        c = max(
            -1.0,
            min(1.0, float(cosine)),
        )
        return float(
            math.degrees(math.acos(c))
        )

    @staticmethod
    def _mean_unit(vectors):
        if not vectors:
            return None

        mean = torch.stack(
            vectors,
            dim=0,
        ).mean(dim=0)

        norm = torch.norm(mean)
        if (
            not torch.isfinite(norm)
            or norm.item() < 1e-12
        ):
            return None

        return (mean / norm).detach()

    def _geometry_row(
        self,
        n,
        v,
    ):
        alpha = self._safe_cos(v, n)
        phi_deg = self._angle_deg_from_cos(alpha)

        n_perp = n - alpha * v
        n_perp_norm = float(
            torch.norm(n_perp).item()
        )

        return {
            "alpha_v_dot_n": float(alpha),
            "phi_deg": float(phi_deg),
            "n_perp_norm": float(n_perp_norm),
        }

    # ------------------------------------------------------------------
    # Current production one-sided normal
    # ------------------------------------------------------------------
    def _estimate_one_sided_parent(
        self,
        z_boundary,
        sigma,
        query_budget,
        seed,
    ):
        rng_state = self._capture_rng_state()

        try:
            self._seed_all(seed)

            probe = self._make_probe_copy(
                seed=seed,
                query_budget=query_budget,
                sigma=sigma,
            )

            q_before = int(
                getattr(probe, "all_queries", 0)
            )

            eta, q_used_returned = probe.estimate_normal(
                z_boundary.detach(),
                int(query_budget),
            )

            q_after = int(
                getattr(probe, "all_queries", q_before)
            )

            q_used_counter = max(
                0,
                q_after - q_before,
            )

            # Prefer the concrete counter when available.
            q_used = (
                q_used_counter
                if q_used_counter > 0
                else int(q_used_returned)
            )

        finally:
            self._restore_rng_state(rng_state)

        if eta is None:
            return {
                "valid": False,
                "normal": None,
                "q_used": int(
                    q_used
                    if "q_used" in locals()
                    else 0
                ),
                "eta_norm": None,
                "informative_pairs": None,
                "pair_count": None,
                "informative_pair_rate": None,
                "plus_adv_count": None,
                "minus_adv_count": None,
            }

        n, eta_norm = self._safe_unit(eta)

        return {
            "valid": n is not None,
            "normal": n,
            "q_used": int(q_used),
            "eta_norm": float(eta_norm),
            "informative_pairs": None,
            "pair_count": None,
            "informative_pair_rate": None,
            "plus_adv_count": None,
            "minus_adv_count": None,
        }

    # ------------------------------------------------------------------
    # Strict hard-label antithetic paired normal
    # ------------------------------------------------------------------
    def _estimate_paired_antithetic(
        self,
        z_boundary,
        sigma,
        query_budget,
        seed,
    ):
        # Fair-budget convention: 2 queries per direction pair.
        pair_count_requested = int(query_budget) // 2

        rng_state = self._capture_rng_state()

        try:
            self._seed_all(seed)

            probe = self._make_probe_copy(
                seed=seed,
                query_budget=query_budget,
                sigma=sigma,
            )

            eta = torch.zeros_like(
                z_boundary
            )

            q_before = int(
                getattr(probe, "all_queries", 0)
            )

            pair_count_used = 0
            informative_pairs = 0
            plus_adv_count = 0
            minus_adv_count = 0

            for _ in range(pair_count_requested):
                # Need two victim queries to complete a fair pair.
                if probe.remaining_queries() < 2:
                    break

                u = torch.randn_like(
                    z_boundary
                )

                z_plus = (
                    z_boundary
                    + float(sigma) * u
                )

                z_minus = (
                    z_boundary
                    - float(sigma) * u
                )

                s_plus = probe.is_adversarial(
                    z_plus,
                    category="normal",
                )

                # If budget or another guard prevented the first query,
                # do not form an incomplete pair.
                if s_plus not in (-1, 1):
                    break

                s_minus = probe.is_adversarial(
                    z_minus,
                    category="normal",
                )

                if s_minus not in (-1, 1):
                    break

                pair_count_used += 1

                if s_plus == 1:
                    plus_adv_count += 1

                if s_minus == 1:
                    minus_adv_count += 1

                weight = float(
                    s_plus - s_minus
                )

                if weight != 0.0:
                    informative_pairs += 1
                    eta = eta + weight * u

            q_after = int(
                getattr(probe, "all_queries", q_before)
            )

            q_used = max(
                0,
                q_after - q_before,
            )

        finally:
            self._restore_rng_state(rng_state)

        n, eta_norm = self._safe_unit(eta)

        informative_pair_rate = (
            float(informative_pairs)
            / float(pair_count_used)
            if pair_count_used > 0
            else 0.0
        )

        return {
            "valid": n is not None,
            "normal": n,
            "q_used": int(q_used),
            "eta_norm": float(eta_norm),
            "informative_pairs": int(informative_pairs),
            "pair_count": int(pair_count_used),
            "informative_pair_rate": float(
                informative_pair_rate
            ),
            "plus_adv_count": int(plus_adv_count),
            "minus_adv_count": int(minus_adv_count),
        }

    # ------------------------------------------------------------------
    # One matched estimate
    # ------------------------------------------------------------------
    def _estimate_method(
        self,
        estimator,
        z_boundary,
        sigma,
        query_budget,
        seed,
    ):
        if estimator == self.ONE_SIDED:
            return self._estimate_one_sided_parent(
                z_boundary=z_boundary,
                sigma=sigma,
                query_budget=query_budget,
                seed=seed,
            )

        if estimator == self.PAIRED:
            return self._estimate_paired_antithetic(
                z_boundary=z_boundary,
                sigma=sigma,
                query_budget=query_budget,
                seed=seed,
            )

        raise ValueError(
            f"Unknown estimator={estimator}"
        )

    # ------------------------------------------------------------------
    # Snapshot diagnostic
    # ------------------------------------------------------------------
    def _run_normal_snapshot_v2(
        self,
        z_boundary,
        iteration_index,
    ):
        v_raw = (
            z_boundary.detach()
            - self.z_source.detach()
        )
        v, radius = self._safe_unit(v_raw)

        if v is None:
            snapshot = {
                "iteration": int(iteration_index),
                "query_at_snapshot": int(
                    self.all_queries
                ),
                "valid": False,
                "reason": "ZERO_V_VECTOR",
                "radius_latent": float(radius),
                "rows": [],
                "repeat_consistency": [],
                "cross_budget": [],
                "cross_sigma": [],
                "paired_vs_one_sided": [],
                "diagnostic_queries": 0,
            }
            self.normal_diagnostics_v2.append(
                snapshot
            )
            return snapshot

        snapshot = {
            "iteration": int(iteration_index),
            "query_at_snapshot": int(
                self.all_queries
            ),
            "valid": True,
            "reason": "OK",
            "radius_latent": float(radius),
            "rows": [],
            "repeat_consistency": [],
            "cross_budget": [],
            "cross_sigma": [],
            "paired_vs_one_sided": [],
            "diagnostic_queries": 0,
        }

        # Key -> list of valid normal vectors across repeats.
        normals = {}
        mean_normals = {}

        estimators = (
            self.ONE_SIDED,
            self.PAIRED,
        )

        for estimator_index, estimator in enumerate(
            estimators
        ):
            for sigma_index, sigma in enumerate(
                self.diagnostic_sigmas
            ):
                for budget_index, budget in enumerate(
                    self.diagnostic_query_budgets
                ):
                    key = (
                        estimator,
                        float(sigma),
                        int(budget),
                    )
                    normals[key] = []

                    for repeat_index in range(
                        self.normal_repeats
                    ):
                        # Deterministic non-overlapping seed blocks.
                        seed = int(
                            self.diagnostic_seed
                            + 10_000_000
                            * int(iteration_index)
                            + 1_000_000
                            * estimator_index
                            + 100_000
                            * sigma_index
                            + 1_000
                            * budget_index
                            + repeat_index
                        )

                        result = self._estimate_method(
                            estimator=estimator,
                            z_boundary=z_boundary,
                            sigma=sigma,
                            query_budget=budget,
                            seed=seed,
                        )

                        q_used = int(
                            result["q_used"]
                        )
                        self.diagnostic_query_total += q_used
                        snapshot[
                            "diagnostic_queries"
                        ] += q_used

                        row = {
                            "estimator": estimator,
                            "sigma": float(sigma),
                            "query_budget": int(budget),
                            "repeat": int(repeat_index),
                            "seed": int(seed),
                            "q_used": int(q_used),
                            "valid": bool(
                                result["valid"]
                            ),
                            "eta_norm": result[
                                "eta_norm"
                            ],
                            "alpha_v_dot_n": None,
                            "phi_deg": None,
                            "n_perp_norm": None,
                            "pair_count": result[
                                "pair_count"
                            ],
                            "informative_pairs": result[
                                "informative_pairs"
                            ],
                            "informative_pair_rate": result[
                                "informative_pair_rate"
                            ],
                            "plus_adv_count": result[
                                "plus_adv_count"
                            ],
                            "minus_adv_count": result[
                                "minus_adv_count"
                            ],
                        }

                        if (
                            result["valid"]
                            and result["normal"] is not None
                        ):
                            n = result["normal"]

                            geom = self._geometry_row(
                                n=n,
                                v=v,
                            )

                            row.update(geom)

                            normals[key].append(n)

                        snapshot["rows"].append(row)

                    mean_n = self._mean_unit(
                        normals[key]
                    )
                    if mean_n is not None:
                        mean_normals[key] = mean_n

                    # Repeat consistency at the exact same config.
                    pairwise = []
                    for ia, ib in combinations(
                        range(
                            len(normals[key])
                        ),
                        2,
                    ):
                        c = self._safe_cos(
                            normals[key][ia],
                            normals[key][ib],
                        )
                        pairwise.append(c)

                    snapshot[
                        "repeat_consistency"
                    ].append(
                        {
                            "estimator": estimator,
                            "sigma": float(sigma),
                            "query_budget": int(budget),
                            "valid_repeat_count": int(
                                len(normals[key])
                            ),
                            "cos_signed_mean": (
                                float(np.mean(pairwise))
                                if pairwise
                                else None
                            ),
                            "cos_signed_min": (
                                float(np.min(pairwise))
                                if pairwise
                                else None
                            ),
                            "cos_abs_mean": (
                                float(
                                    np.mean(
                                        np.abs(pairwise)
                                    )
                                )
                                if pairwise
                                else None
                            ),
                        }
                    )

        # --------------------------------------------------------------
        # Cross-budget consistency at fixed estimator and sigma.
        # --------------------------------------------------------------
        for estimator in estimators:
            for sigma in self.diagnostic_sigmas:
                for b1, b2 in combinations(
                    self.diagnostic_query_budgets,
                    2,
                ):
                    k1 = (
                        estimator,
                        float(sigma),
                        int(b1),
                    )
                    k2 = (
                        estimator,
                        float(sigma),
                        int(b2),
                    )

                    if (
                        k1 not in mean_normals
                        or k2 not in mean_normals
                    ):
                        continue

                    c = self._safe_cos(
                        mean_normals[k1],
                        mean_normals[k2],
                    )

                    snapshot["cross_budget"].append(
                        {
                            "estimator": estimator,
                            "sigma": float(sigma),
                            "budget_a": int(b1),
                            "budget_b": int(b2),
                            "cos_signed": float(c),
                            "cos_abs": float(abs(c)),
                            "angle_deg": float(
                                self._angle_deg_from_cos(c)
                            ),
                        }
                    )

        # --------------------------------------------------------------
        # Cross-sigma consistency at fixed estimator and budget.
        # --------------------------------------------------------------
        for estimator in estimators:
            for budget in self.diagnostic_query_budgets:
                for s1, s2 in combinations(
                    self.diagnostic_sigmas,
                    2,
                ):
                    k1 = (
                        estimator,
                        float(s1),
                        int(budget),
                    )
                    k2 = (
                        estimator,
                        float(s2),
                        int(budget),
                    )

                    if (
                        k1 not in mean_normals
                        or k2 not in mean_normals
                    ):
                        continue

                    c = self._safe_cos(
                        mean_normals[k1],
                        mean_normals[k2],
                    )

                    snapshot["cross_sigma"].append(
                        {
                            "estimator": estimator,
                            "query_budget": int(budget),
                            "sigma_a": float(s1),
                            "sigma_b": float(s2),
                            "cos_signed": float(c),
                            "cos_abs": float(abs(c)),
                            "angle_deg": float(
                                self._angle_deg_from_cos(c)
                            ),
                        }
                    )

        # --------------------------------------------------------------
        # Matched paired-vs-one-sided comparison.
        # --------------------------------------------------------------
        for sigma in self.diagnostic_sigmas:
            for budget in self.diagnostic_query_budgets:
                k_one = (
                    self.ONE_SIDED,
                    float(sigma),
                    int(budget),
                )
                k_pair = (
                    self.PAIRED,
                    float(sigma),
                    int(budget),
                )

                if (
                    k_one not in mean_normals
                    or k_pair not in mean_normals
                ):
                    continue

                c = self._safe_cos(
                    mean_normals[k_one],
                    mean_normals[k_pair],
                )

                snapshot[
                    "paired_vs_one_sided"
                ].append(
                    {
                        "sigma": float(sigma),
                        "query_budget": int(budget),
                        "cos_signed": float(c),
                        "cos_abs": float(abs(c)),
                        "angle_deg": float(
                            self._angle_deg_from_cos(c)
                        ),
                    }
                )

        self.normal_diagnostics_v2.append(
            snapshot
        )

        if self.verbose == "Yes":
            print()
            print(
                f"      [NV-DIAG-V2 pre-Iter{iteration_index}] "
                f"attackQ={self.all_queries} "
                f"latentRadius={radius:.6f}"
            )

            for sigma in self.diagnostic_sigmas:
                print(
                    f"         sigma={sigma:.1e}"
                )

                for budget in self.diagnostic_query_budgets:
                    for estimator in estimators:
                        rc = next(
                            (
                                row
                                for row in snapshot[
                                    "repeat_consistency"
                                ]
                                if (
                                    row["estimator"]
                                    == estimator
                                    and abs(
                                        row["sigma"]
                                        - float(sigma)
                                    )
                                    < 1e-15
                                    and row["query_budget"]
                                    == int(budget)
                                )
                            ),
                            None,
                        )

                        matched_rows = [
                            row
                            for row in snapshot["rows"]
                            if (
                                row["estimator"] == estimator
                                and abs(
                                    row["sigma"]
                                    - float(sigma)
                                )
                                < 1e-15
                                and row["query_budget"]
                                == int(budget)
                                and row["valid"]
                            )
                        ]

                        phi_mean = (
                            float(
                                np.mean(
                                    [
                                        row["phi_deg"]
                                        for row in matched_rows
                                    ]
                                )
                            )
                            if matched_rows
                            else float("nan")
                        )

                        info_rate = None
                        if estimator == self.PAIRED:
                            info_values = [
                                row[
                                    "informative_pair_rate"
                                ]
                                for row in matched_rows
                                if row[
                                    "informative_pair_rate"
                                ] is not None
                            ]
                            if info_values:
                                info_rate = float(
                                    np.mean(info_values)
                                )

                        cos_text = "NA"
                        if (
                            rc is not None
                            and rc[
                                "cos_signed_mean"
                            ] is not None
                        ):
                            cos_text = (
                                f"{rc['cos_signed_mean']:.4f}"
                            )

                        if estimator == self.PAIRED:
                            info_text = (
                                "NA"
                                if info_rate is None
                                else f"{info_rate:.3f}"
                            )
                            print(
                                "            "
                                f"{estimator:<20} "
                                f"B={budget:<4} "
                                f"sameCos={cos_text:<8} "
                                f"phi={phi_mean:6.2f}deg "
                                f"infoRate={info_text}"
                            )
                        else:
                            print(
                                "            "
                                f"{estimator:<20} "
                                f"B={budget:<4} "
                                f"sameCos={cos_text:<8} "
                                f"phi={phi_mean:6.2f}deg"
                            )

            print(
                "         offlineDiagnosticQueries="
                f"{snapshot['diagnostic_queries']}"
            )

        return snapshot

    # ------------------------------------------------------------------
    # Validated BASE attack trajectory with isolated diagnostic snapshots
    # ------------------------------------------------------------------
    def attack(self):
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
                "normal_diagnostics_v2":
                    self.normal_diagnostics_v2,
                "diagnostic_query_total": int(
                    self.diagnostic_query_total
                ),
            }

        self._record(
            z_b,
            "Initial",
        )

        for i in range(self.iteration):
            if i in self.snapshot_iterations:
                self._run_normal_snapshot_v2(
                    z_boundary=z_b,
                    iteration_index=i,
                )

            if (
                self.remaining_queries()
                <= self.update_query_reserve
            ):
                break

            # EXACT validated BASE schedule:
            # N_t = N0 * sqrt(t)
            q_full = int(
                self.N0
                * math.sqrt(i + 1.0)
            )
            q_full = max(1, q_full)

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
                self.rag_pixel_l2(z_b)
            )

            z_next, bridge_info = self.step_rag_ttba(
                self.z_source,
                eta,
                z_b,
            )

            after_l2 = float(
                self.rag_pixel_l2(z_next)
            )

            z_b = z_next.detach()

            self._record(
                z_b,
                f"Iter{i}",
            )

            self.diagnostics.append(
                {
                    "iteration": int(i),
                    "query": int(
                        self.all_queries
                    ),
                    "normal_q_full": int(q_full),
                    "normal_q_used": int(q_used),
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
                            bridge_info["L"]
                        )
                    ),
                    "bridge_k_app": (
                        None
                        if bridge_info is None
                        else float(
                            bridge_info["k_app"]
                        )
                    ),
                }
            )

        best = min(
            self.history,
            key=lambda row: row["l2_pix"],
        )

        final = {
            "query": int(best["query"]),
            "budget_consumed": int(
                self.all_queries
            ),
            "l2_lat": float(
                best["l2_lat"]
            ),
            "l2_pix": float(
                best["l2_pix"]
            ),
            "ssim": float(
                best["ssim"]
            ),
            "lpips": float(
                best["lpips"]
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
            "normal_diagnostics_v2":
                self.normal_diagnostics_v2,
            "diagnostic_query_total": int(
                self.diagnostic_query_total
            ),
        }
