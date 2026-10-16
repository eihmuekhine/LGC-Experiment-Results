
import math

import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBATraceAttack(RAGTtBAPGRAttack):
    """
    RAG-TtBA with query-reused bridge-trace selection.

    Supported methods
    -----------------
    RAG_TTBA_BASE
        The validated baseline. Fresh Gaussian normal estimate every iteration,
        followed by the original TtBA bridge search and piecewise k_app.

    RAG_TTBA_TRACE
        Uses the EXACT SAME bridge victim queries as BASE, but does not discard
        adversarial midpoint candidates already queried during bridge search.

        Let the bridge search query:
            k_i -> hard label

        For every k_i already VERIFIED adversarial, TRACE computes the
        query-free RAG pixel distortion:
            D_i = || Phi(z(k_i)) - x0 ||_2

        It also includes the standard BASE k_app candidate in the candidate set.

        The selected endpoint is:
            argmin D_i

        over:
            {BASE k_app} union {already-queried adversarial bridge points}

        Only the selected endpoint is source-to-candidate boundary refined.

    Important properties
    --------------------
    * ZERO extra victim queries are used to create/evaluate the trace candidate set.
    * Bridge query count is intended to match BASE.
    * Only one boundary refinement is performed, as in BASE.
    * Victim access remains strict top-1 hard label only.
    * Decoder/RAG pixel L2 is attack-side computation and costs zero victim queries.

    This file intentionally does NOT alter the normal estimator or query schedule.
    """

    SUPPORTED_TRACE = {
        "RAG_TTBA_BASE",
        "RAG_TTBA_TRACE",
    }

    def __init__(
        self,
        *args,
        attack_method="RAG_TTBA_BASE",
        trace_min_free_gain=0.0,
        **kwargs,
    ):
        if attack_method not in self.SUPPORTED_TRACE:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported={sorted(self.SUPPORTED_TRACE)}"
            )

        # Parent validates its own method set. Initialize as BASE, then switch
        # to the requested TRACE label.
        super().__init__(
            *args,
            attack_method="RAG_TTBA_BASE",
            **kwargs,
        )

        self.attack_method = attack_method
        self.trace_min_free_gain = float(
            trace_min_free_gain
        )

        # Diagnostics only. These do NOT affect victim-query accounting.
        self.trace_iterations = 0
        self.trace_nonbase_selected = 0
        self.trace_total_bridge_queries = 0
        self.trace_total_verified_adv = 0

    # ==================================================================
    # Preserve BASE normal-estimation behavior exactly for both methods.
    # ==================================================================
    def _choose_normal(
        self,
        z_boundary,
        iteration_index,
    ):
        q_full = int(
            self.N0
            * math.sqrt(
                iteration_index + 1.0
            )
        )
        q_full = max(
            q_full,
            1,
        )

        available_for_normal = max(
            0,
            self.remaining_queries()
            - self.update_query_reserve,
        )

        q_full = min(
            q_full,
            available_for_normal,
        )

        if q_full <= 0:
            return None, None

        eta, q_used = self.estimate_normal(
            z_boundary,
            q_full,
        )

        return eta, {
            "action": "FULL_REFRESH",
            "q_full": int(
                q_full
            ),
            "q_target": int(
                q_full
            ),
            "q_used": int(
                q_used
            ),
            "prev_progress": (
                None
                if self.prev_progress is None
                else float(
                    self.prev_progress
                )
            ),
            "stall_severity": 1.0,
            "history_alpha": 0.0,
            "reuse_streak": 0,
        }

    # ==================================================================
    # Dispatch
    # ==================================================================
    def step_rag_ttba(
        self,
        x_s,
        eta,
        x_b,
    ):
        if self.attack_method == "RAG_TTBA_BASE":
            return super().step_rag_ttba(
                x_s,
                eta,
                x_b,
            )

        return self._step_rag_ttba_trace(
            x_s,
            eta,
            x_b,
        )

    # ==================================================================
    # Query-reused bridge trace
    # ==================================================================
    def _step_rag_ttba_trace(
        self,
        x_s,
        eta,
        x_b,
    ):
        """
        Same TtBA bridge search as BASE, while retaining every already-queried
        bridge midpoint and its hard-label status.

        No extra victim query is made for candidate selection.
        """
        radius = torch.norm(
            x_b - x_s
        )

        if radius.item() < 1e-12:
            return x_b.detach(), {
                "L": 0.0,
                "k_app": 0.0,
                "trace_bridge_query_count": 0,
                "trace_verified_adv_count": 0,
                "trace_candidate_count": 1,
                "trace_selected_k": 0.0,
                "trace_selected_source": "BASE_KAPP",
                "trace_selected_nonbase": 0,
                "trace_base_free_l2": float(
                    self.rag_pixel_l2(
                        x_b
                    )
                ),
                "trace_selected_free_l2": float(
                    self.rag_pixel_l2(
                        x_b
                    )
                ),
                "trace_free_gain": 0.0,
                "trace_records": [],
            }

        L = 0.0
        R = 1.0

        trace_records = []
        verified_adv = []

        q_bridge_before = int(
            self.q_breakdown[
                "bridge"
            ]
        )

        # --------------------------------------------------------------
        # EXACT BASE bridge-query pattern.
        # --------------------------------------------------------------
        while abs(
            R - L
        ) > self.ttba_bridge_tol:
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

            q_before = int(
                self.all_queries
            )

            status = self.is_adversarial(
                cand,
                category="bridge",
            )

            q_after = int(
                self.all_queries
            )

            # No query was actually possible (budget exhausted).
            if q_after == q_before:
                break

            record = {
                "k": float(
                    mid
                ),
                "status": int(
                    status
                ),
                "free_l2": None,
            }

            trace_records.append(
                record
            )

            if status == -1:
                R = mid

            elif status == 1:
                L = mid

                # Store the already-queried adversarial endpoint.
                verified_adv.append(
                    {
                        "k": float(
                            mid
                        ),
                        "z": cand.detach(),
                        "record_index": int(
                            len(
                                trace_records
                            )
                            - 1
                        ),
                    }
                )

            else:
                # Defensive stop. Under normal hard-label operation status is
                # +1/-1. A zero means no usable decision was obtained.
                break

        q_bridge_after = int(
            self.q_breakdown[
                "bridge"
            ]
        )

        bridge_query_count = int(
            q_bridge_after
            - q_bridge_before
        )

        # --------------------------------------------------------------
        # Standard TtBA BASE endpoint.
        #
        # This is included in TRACE's candidate set so TRACE can always
        # select the standard BASE direction when the reused trace provides
        # no better free RAG evidence.
        # --------------------------------------------------------------
        k_app = float(
            self._ttba_piecewise_k(
                L
            )
        )

        d_base = self._ttba_attempt_unit(
            x_s,
            x_b,
            eta,
            k_app,
        )

        z_base = (
            x_s
            + radius * d_base
        ).detach()

        base_free_l2 = float(
            self.rag_pixel_l2(
                z_base
            )
        )

        candidates = [
            {
                "k": float(
                    k_app
                ),
                "z": z_base,
                "source": "BASE_KAPP",
                "free_l2": float(
                    base_free_l2
                ),
            }
        ]

        # --------------------------------------------------------------
        # Reuse ONLY points that were already queried and verified adversarial.
        # Their RAG scores cost zero victim queries.
        # --------------------------------------------------------------
        for item in verified_adv:
            score = float(
                self.rag_pixel_l2(
                    item[
                        "z"
                    ]
                )
            )

            trace_records[
                item[
                    "record_index"
                ]
            ][
                "free_l2"
            ] = float(
                score
            )

            # Avoid duplicate k if a bridge midpoint numerically matches k_app.
            if abs(
                item[
                    "k"
                ]
                - k_app
            ) <= 1e-12:
                continue

            candidates.append(
                {
                    "k": float(
                        item[
                            "k"
                        ]
                    ),
                    "z": item[
                        "z"
                    ].detach(),
                    "source": "REUSED_ADV_QUERY",
                    "free_l2": float(
                        score
                    ),
                }
            )

        # --------------------------------------------------------------
        # Query-free candidate selection.
        #
        # Tie-break explicitly favors BASE_KAPP, so TRACE does not switch
        # away from BASE without a strictly better free RAG score.
        # --------------------------------------------------------------
        raw_best = min(
            candidates,
            key=lambda row: (
                float(
                    row[
                        "free_l2"
                    ]
                ),
                0
                if row[
                    "source"
                ] == "BASE_KAPP"
                else 1,
            ),
        )

        free_gain = float(
            base_free_l2
            - float(
                raw_best[
                    "free_l2"
                ]
            )
        )

        if (
            raw_best[
                "source"
            ] != "BASE_KAPP"
            and free_gain
            >= self.trace_min_free_gain
        ):
            selected = raw_best
        else:
            selected = candidates[
                0
            ]
            free_gain = 0.0

        selected_nonbase = int(
            selected[
                "source"
            ]
            != "BASE_KAPP"
        )

        # --------------------------------------------------------------
        # Only ONE source-to-candidate refinement, same structural cost
        # as BASE.
        # --------------------------------------------------------------
        z_new = self._binary_search_known_adv(
            x_s,
            selected[
                "z"
            ],
            category="boundary",
            max_steps=100,
        )

        # Diagnostics.
        self.trace_iterations += 1
        self.trace_nonbase_selected += int(
            selected_nonbase
        )
        self.trace_total_bridge_queries += int(
            bridge_query_count
        )
        self.trace_total_verified_adv += int(
            len(
                verified_adv
            )
        )

        info = {
            # BASE-compatible keys.
            "L": float(
                L
            ),
            "k_app": float(
                k_app
            ),

            # TRACE diagnostics.
            "trace_bridge_query_count": int(
                bridge_query_count
            ),
            "trace_verified_adv_count": int(
                len(
                    verified_adv
                )
            ),
            "trace_candidate_count": int(
                len(
                    candidates
                )
            ),
            "trace_selected_k": float(
                selected[
                    "k"
                ]
            ),
            "trace_selected_source": str(
                selected[
                    "source"
                ]
            ),
            "trace_selected_nonbase": int(
                selected_nonbase
            ),
            "trace_base_free_l2": float(
                base_free_l2
            ),
            "trace_selected_free_l2": float(
                selected[
                    "free_l2"
                ]
            ),
            "trace_free_gain": float(
                free_gain
            ),
            "trace_records": trace_records,
        }

        if self.verbose == "Yes":
            print(
                "         TRACE: "
                f"qBridge={bridge_query_count} "
                f"advTrace={len(verified_adv)} "
                f"cands={len(candidates)} "
                f"baseK={k_app:.4f} "
                f"selectedK={selected['k']:.4f} "
                f"source={selected['source']} "
                f"freeGain={free_gain:.6f}"
            )

        return z_new.detach(), info

    # ==================================================================
    # Add aggregate TRACE diagnostics to the normal parent result.
    # ==================================================================
    def attack(
        self,
    ):
        result = super().attack()

        result[
            "trace_summary"
        ] = {
            "iterations": int(
                self.trace_iterations
            ),
            "nonbase_selected": int(
                self.trace_nonbase_selected
            ),
            "total_bridge_queries": int(
                self.trace_total_bridge_queries
            ),
            "total_verified_adv": int(
                self.trace_total_verified_adv
            ),
        }

        return result
