
import math

import numpy as np
import torch

from attacks.rag_ttba_sub16 import RAGTtBASubspaceAttack


class RAGTtBASub16TanAttack(RAGTtBASubspaceAttack):
    """
    Matched-budget test of an EXACTLY TANGENTIAL low-dimensional normal.

    Methods
    -------
    RAG_TTBA_BASE
        Validated full-space baseline.

    RAG_TTBA_SUB16
        Previous r=16 subspace-normal attack. Included only as an optional
        control.

    RAG_TTBA_SUB16_TAN
        Uses the SAME r=16 hard-label subspace estimator and SAME victim-query
        schedule as RAG_TTBA_SUB16, but removes the radial component along v:

            v = Normalize(z_b - z_s)

            n_sub = Normalize(estimated subspace normal)

            t = n_sub - (n_sub^T v) v

            n_tan = Normalize(t)

        Hence, up to floating-point error,

            n_tan^T v = 0.

        The standard TtBA bridge then uses

            d(k) = Normalize((1-k) v + k n_tan).

    Important
    ---------
    Orthogonalization is FREE computation. It uses no additional victim-model
    queries. The normal-query target remains exactly the BASE schedule:

        N_t = int(N0 * sqrt(t+1)).

    The bridge search and source-to-boundary refinement are unchanged.

    Threat model
    ------------
    Strict hard-label only. No logits, probabilities, gradients, or backward
    pass through the victim model.

    Why this experiment exists
    --------------------------
    The previous SUB16 smoke showed that the raw subspace normal was stable but
    strongly radial (large v^T n), and the attack became much worse than BASE.
    This experiment isolates whether that radial contamination caused the
    failure by forcing the estimated second TtBA direction to be exactly
    tangent to v.
    """

    METHOD_TAN = "RAG_TTBA_SUB16_TAN"

    SUPPORTED = {
        RAGTtBASubspaceAttack.METHOD_BASE,
        RAGTtBASubspaceAttack.METHOD_SUB16,
        METHOD_TAN,
    }

    def __init__(
        self,
        *args,
        attack_method="RAG_TTBA_SUB16_TAN",
        tangent_eps=1e-10,
        **kwargs,
    ):
        if attack_method not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported={sorted(self.SUPPORTED)}"
            )

        requested = str(
            attack_method
        )

        # The parent understands BASE and SUB16.  TAN uses the same subspace
        # estimator machinery as SUB16, then we orthogonalize its output in
        # this subclass.
        parent_method = (
            requested
            if requested
            in {
                RAGTtBASubspaceAttack.METHOD_BASE,
                RAGTtBASubspaceAttack.METHOD_SUB16,
            }
            else RAGTtBASubspaceAttack.METHOD_SUB16
        )

        super().__init__(
            *args,
            attack_method=parent_method,
            **kwargs,
        )

        # Override the method label after the parent has initialized all
        # subspace state.
        self.requested_attack_method = requested

        self.tangent_eps = float(
            tangent_eps
        )

        if self.tangent_eps <= 0.0:
            raise ValueError(
                "tangent_eps must be > 0."
            )

        self.tangent_projection_diagnostics = []

    # ==================================================================
    # Geometry
    # ==================================================================
    def _project_normal_to_exact_tangent(
        self,
        eta,
        z_boundary,
    ):
        if eta is None:
            return None, {
                "valid": False,
                "reason": "ETA_NONE",
            }

        eta_flat = eta.detach().reshape(
            -1
        )

        z_flat = z_boundary.detach().reshape(
            -1
        )

        source_flat = self.z_source.detach().reshape(
            -1
        )

        v_raw = (
            z_flat
            - source_flat
        )

        v_norm = torch.norm(
            v_raw
        )

        if (
            not torch.isfinite(
                v_norm
            )
            or v_norm.item() < self.tangent_eps
        ):
            return None, {
                "valid": False,
                "reason": "ZERO_V",
            }

        v = (
            v_raw / v_norm
        ).detach()

        eta_norm = torch.norm(
            eta_flat
        )

        if (
            not torch.isfinite(
                eta_norm
            )
            or eta_norm.item() < self.tangent_eps
        ):
            return None, {
                "valid": False,
                "reason": "ZERO_ETA",
            }

        n_raw = (
            eta_flat / eta_norm
        ).detach()

        raw_alpha = float(
            torch.dot(
                n_raw,
                v,
            ).item()
        )

        raw_alpha = max(
            -1.0,
            min(
                1.0,
                raw_alpha,
            ),
        )

        raw_phi_deg = float(
            math.degrees(
                math.acos(
                    raw_alpha
                )
            )
        )

        tangent_raw = (
            n_raw
            - raw_alpha * v
        )

        tangent_norm = torch.norm(
            tangent_raw
        )

        tangent_norm_value = float(
            tangent_norm.item()
        )

        if (
            not torch.isfinite(
                tangent_norm
            )
            or tangent_norm_value < self.tangent_eps
        ):
            return None, {
                "valid": False,
                "reason": "DEGENERATE_TANGENT",
                "raw_alpha": float(
                    raw_alpha
                ),
                "raw_phi_deg": float(
                    raw_phi_deg
                ),
                "raw_perp_norm": float(
                    tangent_norm_value
                ),
            }

        n_tan_flat = (
            tangent_raw / tangent_norm
        ).detach()

        tan_dot_v = float(
            torch.dot(
                n_tan_flat,
                v,
            ).item()
        )

        # Numerical sanity.  This should be extremely close to zero.
        abs_tan_dot_v = float(
            abs(
                tan_dot_v
            )
        )

        eta_tan = n_tan_flat.reshape_as(
            eta
        ).detach()

        return eta_tan, {
            "valid": True,
            "reason": "OK",
            "raw_alpha": float(
                raw_alpha
            ),
            "raw_phi_deg": float(
                raw_phi_deg
            ),
            "raw_perp_norm": float(
                tangent_norm_value
            ),
            "tan_dot_v": float(
                tan_dot_v
            ),
            "abs_tan_dot_v": float(
                abs_tan_dot_v
            ),
            "tan_phi_deg": 90.0,
            "latent_radius": float(
                v_norm.item()
            ),
        }

    # ==================================================================
    # Normal selection
    # ==================================================================
    def _choose_normal(
        self,
        z_boundary,
        iteration_index,
    ):
        # BASE and raw SUB16 are delegated unchanged to the already-tested
        # implementation.
        if (
            self.requested_attack_method
            != self.METHOD_TAN
        ):
            return super()._choose_normal(
                z_boundary,
                iteration_index,
            )

        # Exact same target query schedule as BASE/SUB16.
        q_full = int(
            self.N0
            * math.sqrt(
                iteration_index
                + 1.0
            )
        )

        q_full = max(
            1,
            q_full,
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

        # Use the exact same r=16 one-sided subspace estimator used by the
        # previous SUB16 test.
        eta_raw, q_used, raw_diag = (
            self._estimate_subspace_normal(
                z_boundary=z_boundary,
                q_target=q_full,
                iteration_index=iteration_index,
            )
        )

        eta_tan, tan_diag = (
            self._project_normal_to_exact_tangent(
                eta=eta_raw,
                z_boundary=z_boundary,
            )
        )

        combined_diag = dict(
            raw_diag
        )

        combined_diag.update(
            {
                "attack_method":
                    self.METHOD_TAN,
                "tangent_valid":
                    bool(
                        tan_diag.get(
                            "valid",
                            False,
                        )
                    ),
                "tangent_reason":
                    tan_diag.get(
                        "reason"
                    ),
                "raw_alpha_before_projection":
                    tan_diag.get(
                        "raw_alpha"
                    ),
                "raw_phi_deg_before_projection":
                    tan_diag.get(
                        "raw_phi_deg"
                    ),
                "raw_perp_norm_before_projection":
                    tan_diag.get(
                        "raw_perp_norm"
                    ),
                "tan_dot_v":
                    tan_diag.get(
                        "tan_dot_v"
                    ),
                "abs_tan_dot_v":
                    tan_diag.get(
                        "abs_tan_dot_v"
                    ),
                "tan_phi_deg":
                    tan_diag.get(
                        "tan_phi_deg"
                    ),
            }
        )

        # Preserve the parent's subspace diagnostics list so runner/result
        # handling stays compatible.
        self.subspace_normal_diagnostics.append(
            combined_diag
        )

        self.tangent_projection_diagnostics.append(
            {
                "iteration": int(
                    iteration_index
                ),
                "q_target": int(
                    q_full
                ),
                "q_used": int(
                    q_used
                ),
                **tan_diag,
            }
        )

        if self.verbose == "Yes":
            if (
                raw_diag.get(
                    "valid",
                    False,
                )
                and tan_diag.get(
                    "valid",
                    False,
                )
            ):
                split_text = (
                    "NA"
                    if raw_diag.get(
                        "split_half_cos"
                    ) is None
                    else (
                        f"{raw_diag['split_half_cos']:.4f}"
                    )
                )

                print(
                    "         SUB16-TAN: "
                    f"qN={q_used}/{q_full} "
                    f"rawSplitCos={split_text} "
                    f"rawPhi={tan_diag['raw_phi_deg']:.2f}deg "
                    f"rawAlpha={tan_diag['raw_alpha']:.4f} "
                    f"rawPerp={tan_diag['raw_perp_norm']:.4f} "
                    f"tanDotV={tan_diag['tan_dot_v']:.3e} "
                    f"advFrac={raw_diag['adv_fraction']:.3f}"
                )

            else:
                print(
                    "         SUB16-TAN: invalid "
                    f"rawReason={raw_diag.get('reason')} "
                    f"tanReason={tan_diag.get('reason')}"
                )

        # No extra-query fallback is used.  If the tangential component is
        # degenerate, returning None stops rather than silently changing the
        # threat/query budget.
        if eta_tan is None:
            return None, {
                "action": "SUB16_TAN_INVALID",
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
                "normal_method":
                    "SUBSPACE_EXACT_TANGENT",
                "subspace_dim": int(
                    self.subspace_dim
                ),
                "tangent_valid": False,
            }

        return eta_tan, {
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
            "normal_method":
                "SUBSPACE_EXACT_TANGENT",
            "subspace_dim": int(
                self.subspace_dim
            ),
            "raw_split_half_cos":
                raw_diag.get(
                    "split_half_cos"
                ),
            "raw_alpha":
                tan_diag.get(
                    "raw_alpha"
                ),
            "raw_perp_norm":
                tan_diag.get(
                    "raw_perp_norm"
                ),
            "tan_dot_v":
                tan_diag.get(
                    "tan_dot_v"
                ),
            "tangent_valid": True,
        }

    # ==================================================================
    # Result augmentation
    # ==================================================================
    def attack(
        self,
    ):
        result = super().attack()

        valid_tan = [
            row
            for row in self.tangent_projection_diagnostics
            if row.get(
                "valid",
                False,
            )
        ]

        raw_alpha_values = [
            float(
                row[
                    "raw_alpha"
                ]
            )
            for row in valid_tan
            if row.get(
                "raw_alpha"
            ) is not None
            and np.isfinite(
                row[
                    "raw_alpha"
                ]
            )
        ]

        raw_perp_values = [
            float(
                row[
                    "raw_perp_norm"
                ]
            )
            for row in valid_tan
            if row.get(
                "raw_perp_norm"
            ) is not None
            and np.isfinite(
                row[
                    "raw_perp_norm"
                ]
            )
        ]

        abs_dot_values = [
            float(
                row[
                    "abs_tan_dot_v"
                ]
            )
            for row in valid_tan
            if row.get(
                "abs_tan_dot_v"
            ) is not None
            and np.isfinite(
                row[
                    "abs_tan_dot_v"
                ]
            )
        ]

        result[
            "requested_attack_method"
        ] = self.requested_attack_method

        result[
            "tangent_projection_diagnostics"
        ] = self.tangent_projection_diagnostics

        result[
            "tangent_projection_summary"
        ] = {
            "method":
                self.requested_attack_method,
            "projection_updates": int(
                len(
                    valid_tan
                )
            ),
            "mean_raw_alpha": (
                None
                if not raw_alpha_values
                else float(
                    np.mean(
                        raw_alpha_values
                    )
                )
            ),
            "median_raw_alpha": (
                None
                if not raw_alpha_values
                else float(
                    np.median(
                        raw_alpha_values
                    )
                )
            ),
            "mean_raw_perp_norm": (
                None
                if not raw_perp_values
                else float(
                    np.mean(
                        raw_perp_values
                    )
                )
            ),
            "max_abs_tan_dot_v": (
                None
                if not abs_dot_values
                else float(
                    np.max(
                        abs_dot_values
                    )
                )
            ),
        }

        return result
