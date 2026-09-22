
import math

import numpy as np
import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBABQCompareAttack(RAGTtBAPGRAttack):
    """
    Fair comparison of:

      1) RAG_TTBA_BASE
         Validated TtBA baseline.

      2) RAG_TTBA_K075
         Diagnostic control motivated by the earlier QS-FD screen:
             k_use = rho * k_app,  rho=0.75 by default.
         It uses the exact BASE TtBA bridge threshold search and adds
         ZERO selector queries.

      3) RAG_TTBA_BQ
         Boundary-Quality-Aware TtBA.

         The method reallocates the BASE bridge-query budget rather than
         adding a large K-ray screen:

             coarse TtBA threshold search
                    |
                    v
             k_app and rho*k_app
                    |
                    v
             sequential radial hard-label probes
                    |
                    v
             choose boundary-quality-supported ray
                    |
                    v
             full source-to-adversarial refinement of winner only

    Default BQ query allocation
    ---------------------------
    BASE bridge tolerance = 1e-3 -> normally 10 bridge queries.

    BQ uses:
        bq_bridge_tol = 1/64 = 0.015625
            -> at most 6 bridge queries

        radial lambdas = (0.85, 0.925)
            -> at most 2 candidates x 2 depths = 4 BQ queries

    Therefore:
        bridge + BQ screening <= 10 queries/iteration

    before the same final boundary refinement.  This makes the proposed
    selector approximately budget-neutral relative to the original TtBA
    bridge search.

    Strict threat model
    -------------------
    * victim top-1 hard labels only;
    * no logits/probabilities used by attack logic;
    * no victim gradients/backward;
    * all victim queries counted exactly once through the parent counter;
    * RAG pixel L2 is decoder-side FREE feedback.

    Important
    ---------
    RAG_TTBA_K075 is a diagnostic control, not the proposed innovation.
    RAG_TTBA_BQ is the proposed mechanism because it makes the choice
    online from queried boundary-feasibility evidence.
    """

    METHOD_BASE = "RAG_TTBA_BASE"
    METHOD_K075 = "RAG_TTBA_K075"
    METHOD_BQ = "RAG_TTBA_BQ"

    SUPPORTED_BQ = {
        METHOD_BASE,
        METHOD_K075,
        METHOD_BQ,
    }

    def __init__(
        self,
        *args,
        attack_method="RAG_TTBA_BASE",
        # Shared contracted candidate.
        bq_rho=0.75,
        # Budget-neutral BQ threshold search:
        # 6 bisection queries because 2^-6 = 0.015625.
        bq_bridge_tol=0.015625,
        # Deepest radial test first.  Two candidates x two depths = <=4 Q.
        bq_lambdas=(0.85, 0.925),
        # FREE monotonic safeguard after winner refinement.
        bq_accept_eps=1e-8,
        bq_monotone_safety=True,
        **kwargs,
    ):
        if attack_method not in self.SUPPORTED_BQ:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported={sorted(self.SUPPORTED_BQ)}"
            )

        requested = str(attack_method)

        # Keep parent on the validated BASE normal-estimator path.
        super().__init__(
            *args,
            attack_method=self.METHOD_BASE,
            **kwargs,
        )

        self.requested_attack_method = requested

        self.bq_rho = float(bq_rho)
        if not (0.0 < self.bq_rho <= 1.0):
            raise ValueError("bq_rho must be in (0, 1].")

        self.bq_bridge_tol = float(bq_bridge_tol)
        if not (0.0 < self.bq_bridge_tol < 1.0):
            raise ValueError("bq_bridge_tol must be in (0, 1).")

        self.bq_lambdas = tuple(
            sorted(
                {
                    float(x)
                    for x in bq_lambdas
                }
            )
        )
        if not self.bq_lambdas:
            raise ValueError("bq_lambdas cannot be empty.")
        if any(
            (x <= 0.0 or x >= 1.0)
            for x in self.bq_lambdas
        ):
            raise ValueError(
                "Every bq lambda must be strictly inside (0, 1)."
            )

        self.bq_accept_eps = float(bq_accept_eps)
        self.bq_monotone_safety = bool(bq_monotone_safety)

        # Extra exact accounting category.
        if "bq" not in self.q_breakdown:
            self.q_breakdown["bq"] = 0

        self.bq_diagnostics = []

    # ==================================================================
    # TtBA threshold search without final refinement
    # ==================================================================
    def _ttba_threshold_only(
        self,
        x_s,
        eta,
        x_b,
        bridge_tol,
    ):
        """
        Run only the TtBA bridge binary search.

        Invariant:
            L = last bridge coefficient observed adversarial at current radius
            R = clean-side upper bracket (under the TtBA monotonic bridge model)

        Returns L, k_app, radius and exact bridge-query count.
        """
        radius = torch.norm(
            x_b - x_s
        )

        if radius.item() < self.eps:
            return {
                "L": 0.0,
                "R": 0.0,
                "k_app": 0.0,
                "radius": 0.0,
                "queries": 0,
            }

        q_before = int(
            self.all_queries
        )

        L = 0.0
        R = 1.0

        while abs(R - L) > float(bridge_tol):
            if self.remaining_queries() <= 0:
                break

            mid = 0.5 * (
                L + R
            )

            d = self._ttba_attempt_unit(
                x_s,
                x_b,
                eta,
                mid,
            )

            cand = (
                x_s
                + radius * d
            )

            status = self.is_adversarial(
                cand,
                category="bridge",
            )

            if status == 0:
                break

            if status == 1:
                L = mid
            else:
                R = mid

        k_app = self._ttba_piecewise_k(
            L
        )

        # Numerical safety.
        k_app = float(
            max(
                0.0,
                min(
                    float(L),
                    float(k_app),
                ),
            )
        )

        return {
            "L": float(L),
            "R": float(R),
            "k_app": float(k_app),
            "radius": float(
                radius.item()
            ),
            "queries": int(
                self.all_queries
                - q_before
            ),
        }

    # ==================================================================
    # K075 diagnostic control
    # ==================================================================
    def _step_k075(
        self,
        x_s,
        eta,
        x_b,
    ):
        q_before = int(
            self.all_queries
        )

        threshold = self._ttba_threshold_only(
            x_s=x_s,
            eta=eta,
            x_b=x_b,
            bridge_tol=self.ttba_bridge_tol,
        )

        radius = float(
            threshold["radius"]
        )

        if radius <= self.eps:
            return x_b.detach(), {
                "mode": self.METHOD_K075,
                **threshold,
                "k_use": 0.0,
                "selector_queries": int(
                    self.all_queries
                    - q_before
                ),
            }

        k_app = float(
            threshold["k_app"]
        )

        k_use = float(
            max(
                0.0,
                min(
                    threshold["L"],
                    self.bq_rho * k_app,
                ),
            )
        )

        d = self._ttba_attempt_unit(
            x_s,
            x_b,
            eta,
            k_use,
        )

        # Same endpoint assumption used by BASE:
        # k_use <= k_app <= L, where L is the adversarial bridge side.
        adv_trial = (
            x_s
            + radius * d
        )

        z_new = self._binary_search_known_adv(
            x_s,
            adv_trial,
            category="boundary",
            max_steps=100,
        )

        return z_new.detach(), {
            "mode": self.METHOD_K075,
            **threshold,
            "rho": float(
                self.bq_rho
            ),
            "k_use": float(
                k_use
            ),
            "selector_queries": int(
                self.all_queries
                - q_before
            ),
        }

    # ==================================================================
    # Boundary-quality probe helpers
    # ==================================================================
    def _candidate_direction(
        self,
        x_s,
        eta,
        x_b,
        k,
    ):
        return self._ttba_attempt_unit(
            x_s,
            x_b,
            eta,
            float(k),
        )

    def _query_two_radial_points(
        self,
        point_contract,
        point_base,
    ):
        """
        Query the contracted and BASE ray at one radial depth in one batch.

        Returns:
            status_contract, status_base

        where:
            +1 = adversarial
            -1 = clean
             0 = unavailable because budget ended.
        """
        if self.remaining_queries() <= 0:
            return 0, 0

        points = torch.cat(
            [
                point_contract,
                point_base,
            ],
            dim=0,
        )

        labels = self._query_labels(
            points,
            category="bq",
        )

        if labels.numel() == 0:
            return 0, 0

        status = self._status_from_labels(
            labels
        )

        s_contract = int(
            status[0].item()
        )

        s_base = 0
        if status.numel() >= 2:
            s_base = int(
                status[1].item()
            )

        return s_contract, s_base

    # ==================================================================
    # BQ-TtBA proposed mechanism
    # ==================================================================
    def _step_bq(
        self,
        x_s,
        eta,
        x_b,
    ):
        total_before = int(
            self.all_queries
        )

        # Reallocate BASE's ~10 bridge queries:
        # 6 coarse bridge + <=4 radial BQ probes.
        threshold = self._ttba_threshold_only(
            x_s=x_s,
            eta=eta,
            x_b=x_b,
            bridge_tol=self.bq_bridge_tol,
        )

        radius = float(
            threshold["radius"]
        )

        L = float(
            threshold["L"]
        )

        k_base = float(
            threshold["k_app"]
        )

        k_contract = float(
            max(
                0.0,
                min(
                    L,
                    self.bq_rho * k_base,
                ),
            )
        )

        if radius <= self.eps:
            return x_b.detach(), {
                "mode": self.METHOD_BQ,
                **threshold,
                "rho": float(
                    self.bq_rho
                ),
                "k_base": float(
                    k_base
                ),
                "k_contract": float(
                    k_contract
                ),
                "selection_reason": "ZERO_RADIUS",
                "selected": "INCUMBENT",
                "bq_queries": 0,
                "selector_queries": int(
                    self.all_queries
                    - total_before
                ),
                "probes": [],
            }

        d_base = self._candidate_direction(
            x_s,
            eta,
            x_b,
            k_base,
        )

        d_contract = self._candidate_direction(
            x_s,
            eta,
            x_b,
            k_contract,
        )

        # Default fallback is exactly the coarse-TtBA BASE ray at full radius.
        selected_name = "BASE"
        selected_k = float(
            k_base
        )
        selected_point = (
            x_s
            + radius * d_base
        ).detach()
        selected_point_verified_adv = False
        selection_reason = "NO_DECISIVE_RADIAL_EVIDENCE"

        probe_rows = []

        bq_before = int(
            self.q_breakdown[
                "bq"
            ]
        )

        for lam in self.bq_lambdas:
            if self.remaining_queries() < 2:
                selection_reason = "BUDGET_FALLBACK_BASE"
                break

            lam = float(
                lam
            )

            p_contract = (
                x_s
                + (lam * radius) * d_contract
            )

            p_base = (
                x_s
                + (lam * radius) * d_base
            )

            s_contract, s_base = (
                self._query_two_radial_points(
                    p_contract,
                    p_base,
                )
            )

            row = {
                "lambda": float(
                    lam
                ),
                "status_contract": int(
                    s_contract
                ),
                "status_base": int(
                    s_base
                ),
                "pixel_l2_contract": None,
                "pixel_l2_base": None,
            }

            # If one query was unavailable, fall back safely.
            if (
                s_contract == 0
                or s_base == 0
            ):
                probe_rows.append(
                    row
                )
                selection_reason = "BUDGET_FALLBACK_BASE"
                break

            # One ray survives deeper toward source and the other does not:
            # this is the cleanest boundary-depth evidence.
            if (
                s_contract == 1
                and s_base == -1
            ):
                selected_name = "CONTRACT"
                selected_k = float(
                    k_contract
                )
                selected_point = (
                    p_contract.detach()
                )
                selected_point_verified_adv = True
                selection_reason = (
                    "CONTRACT_DEEPER_ADVERSARIAL"
                )

                row[
                    "pixel_l2_contract"
                ] = float(
                    self.rag_pixel_l2(
                        p_contract
                    )
                )

                probe_rows.append(
                    row
                )
                break

            if (
                s_contract == -1
                and s_base == 1
            ):
                selected_name = "BASE"
                selected_k = float(
                    k_base
                )
                selected_point = (
                    p_base.detach()
                )
                selected_point_verified_adv = True
                selection_reason = (
                    "BASE_DEEPER_ADVERSARIAL"
                )

                row[
                    "pixel_l2_base"
                ] = float(
                    self.rag_pixel_l2(
                        p_base
                    )
                )

                probe_rows.append(
                    row
                )
                break

            # Both rays are adversarial at the deepest currently tested radius.
            # They have equal radial depth evidence; use FREE RAG Pixel L2 only
            # as a tie-break among VERIFIED adversarial points.
            if (
                s_contract == 1
                and s_base == 1
            ):
                l2_contract = float(
                    self.rag_pixel_l2(
                        p_contract
                    )
                )

                l2_base = float(
                    self.rag_pixel_l2(
                        p_base
                    )
                )

                row[
                    "pixel_l2_contract"
                ] = l2_contract
                row[
                    "pixel_l2_base"
                ] = l2_base

                if (
                    l2_contract
                    < l2_base
                ):
                    selected_name = "CONTRACT"
                    selected_k = float(
                        k_contract
                    )
                    selected_point = (
                        p_contract.detach()
                    )
                else:
                    selected_name = "BASE"
                    selected_k = float(
                        k_base
                    )
                    selected_point = (
                        p_base.detach()
                    )

                selected_point_verified_adv = True
                selection_reason = (
                    "BOTH_ADV_L2_TIEBREAK"
                )

                probe_rows.append(
                    row
                )
                break

            # Both are clean at this deep radius. Move outward to the next
            # lambda and ask again.
            probe_rows.append(
                row
            )
            selection_reason = "BOTH_CLEAN_TRY_SHALLOWER"

        bq_queries = int(
            self.q_breakdown[
                "bq"
            ]
            - bq_before
        )

        # If radial probing selected a VERIFIED adversarial contracted point,
        # use it directly as the upper endpoint. Otherwise, use the full-radius
        # BASE TtBA endpoint exactly as the original method does.
        z_new = self._binary_search_known_adv(
            x_s,
            selected_point,
            category="boundary",
            max_steps=100,
        )

        return z_new.detach(), {
            "mode": self.METHOD_BQ,
            **threshold,
            "rho": float(
                self.bq_rho
            ),
            "k_base": float(
                k_base
            ),
            "k_contract": float(
                k_contract
            ),
            "selected": str(
                selected_name
            ),
            "selected_k": float(
                selected_k
            ),
            "selected_point_verified_adv":
                int(
                    selected_point_verified_adv
                ),
            "selection_reason": str(
                selection_reason
            ),
            "bq_queries": int(
                bq_queries
            ),
            "selector_queries": int(
                self.all_queries
                - total_before
            ),
            "probes": probe_rows,
        }

    # ==================================================================
    # Main comparison loop
    # ==================================================================
    def attack(
        self,
    ):
        # Exact validated BASE implementation.
        if (
            self.requested_attack_method
            == self.METHOD_BASE
        ):
            result = super().attack()

            result[
                "requested_attack_method"
            ] = self.METHOD_BASE

            result[
                "bq_diagnostics"
            ] = []

            return result

        z_b = self._initialize_boundary()

        if z_b is None:
            return {
                "success": False,
                "final_z": self.z_source.detach(),
                "final": None,
                "checkpoints":
                    self.checkpoint_summary(),
                "thresholds":
                    self.threshold_summary(),
                "query_breakdown":
                    dict(
                        self.q_breakdown
                    ),
                "diagnostics":
                    self.diagnostics,
                "bq_diagnostics":
                    self.bq_diagnostics,
                "requested_attack_method":
                    self.requested_attack_method,
            }

        self._record(
            z_b,
            "Initial",
        )

        for i in range(
            self.iteration
        ):
            if (
                self.remaining_queries()
                <= self.update_query_reserve
            ):
                break

            before_l2 = float(
                self.rag_pixel_l2(
                    z_b
                )
            )

            # Because parent attack_method is BASE, this is exactly the
            # validated fresh-normal schedule N_t=N0*sqrt(t).
            eta, normal_info = (
                self._choose_normal(
                    z_b,
                    i,
                )
            )

            if eta is None:
                break

            if self.remaining_queries() <= 0:
                break

            if (
                self.requested_attack_method
                == self.METHOD_K075
            ):
                z_candidate, step_info = (
                    self._step_k075(
                        self.z_source,
                        eta,
                        z_b,
                    )
                )

            elif (
                self.requested_attack_method
                == self.METHOD_BQ
            ):
                z_candidate, step_info = (
                    self._step_bq(
                        self.z_source,
                        eta,
                        z_b,
                    )
                )

            else:
                raise RuntimeError(
                    "Unknown requested method."
                )

            candidate_l2 = float(
                self.rag_pixel_l2(
                    z_candidate
                )
            )

            # Optional FREE monotone safety for the two experimental variants.
            if self.bq_monotone_safety:
                accepted = (
                    candidate_l2
                    < before_l2
                    - self.bq_accept_eps
                )
            else:
                accepted = True

            if accepted:
                z_next = z_candidate.detach()
                after_l2 = float(
                    candidate_l2
                )
            else:
                z_next = z_b.detach()
                after_l2 = float(
                    before_l2
                )

            progress = self._relative_progress(
                before_l2,
                after_l2,
            )

            z_b = z_next.detach()
            self.prev_progress = float(
                progress
            )

            self._record(
                z_b,
                f"Iter{i}",
            )

            compact_step = {
                key: value
                for key, value
                in step_info.items()
                if key != "probes"
            }

            diag = {
                "iteration": int(
                    i
                ),
                "query": int(
                    self.all_queries
                ),
                "method":
                    self.requested_attack_method,
                "pixel_l2_before":
                    float(
                        before_l2
                    ),
                "pixel_l2_candidate":
                    float(
                        candidate_l2
                    ),
                "pixel_l2_after":
                    float(
                        after_l2
                    ),
                "progress":
                    float(
                        progress
                    ),
                "accepted":
                    int(
                        accepted
                    ),
            }

            if normal_info is not None:
                diag.update(
                    {
                        f"normal_{k}": v
                        for k, v
                        in normal_info.items()
                    }
                )

            diag.update(
                {
                    f"step_{k}": v
                    for k, v
                    in compact_step.items()
                }
            )

            self.diagnostics.append(
                diag
            )

            self.bq_diagnostics.append(
                {
                    "iteration": int(
                        i
                    ),
                    "query": int(
                        self.all_queries
                    ),
                    "accepted": int(
                        accepted
                    ),
                    **step_info,
                }
            )

            if self.verbose == "Yes":
                if (
                    self.requested_attack_method
                    == self.METHOD_K075
                ):
                    print(
                        "         K075: "
                        f"qN={normal_info['q_used']}/"
                        f"{normal_info['q_full']} "
                        f"L={step_info['L']:.4f} "
                        f"k_app={step_info['k_app']:.4f} "
                        f"k_use={step_info['k_use']:.4f} "
                        f"progress={progress:.4f} "
                        f"accepted={int(accepted)}"
                    )

                else:
                    print(
                        "         BQ: "
                        f"qN={normal_info['q_used']}/"
                        f"{normal_info['q_full']} "
                        f"qBridge={step_info['queries']} "
                        f"qBQ={step_info['bq_queries']} "
                        f"L={step_info['L']:.4f} "
                        f"kBase={step_info['k_base']:.4f} "
                        f"kCtr={step_info['k_contract']:.4f} "
                        f"selected={step_info['selected']} "
                        f"reason={step_info['selection_reason']} "
                        f"progress={progress:.4f} "
                        f"accepted={int(accepted)}"
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

        q_sum = int(
            sum(
                int(v)
                for v in self.q_breakdown.values()
            )
        )

        return {
            "success": True,
            "final_z": z_b.detach(),
            "final": final,
            "checkpoints":
                self.checkpoint_summary(),
            "thresholds":
                self.threshold_summary(),
            "query_breakdown":
                dict(
                    self.q_breakdown
                ),
            "query_audit": {
                "all_queries": int(
                    self.all_queries
                ),
                "sum_breakdown": int(
                    q_sum
                ),
                "matches": bool(
                    q_sum
                    == int(
                        self.all_queries
                    )
                ),
            },
            "diagnostics":
                self.diagnostics,
            "bq_diagnostics":
                self.bq_diagnostics,
            "requested_attack_method":
                self.requested_attack_method,
        }
