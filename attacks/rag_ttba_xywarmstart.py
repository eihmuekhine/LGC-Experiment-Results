
import math
from collections import OrderedDict

import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBAXYWarmStartAttack(RAGTtBAPGRAttack):
    """
    Controlled early-XY warm-start followed permanently by standard RAG-TtBA.

    Motivation
    ----------
    Prior controlled runs showed:

      * direct XY coordinate search can make strong early improvements;
      * using XY as the full optimizer is much worse than RAG-TtBA later;
      * therefore XY should be treated only as an EARLY boundary contraction
        mechanism, not as the full iterative optimizer.

    Local coordinate plane
    ----------------------
        e_x = v_t
            = (z_b^t - z_s) / ||z_b^t - z_s||

        e_y = normalized component of the fresh estimated normal n_t
              orthogonal to e_x

        z(x,y) = z_s + r_t (x e_x + y e_y)

        r_t = ||z_b^t - z_s||

    Hence the current boundary point is exactly:

        (x,y) = (1,0).

    Methods
    -------
    RAG_TTBA_BASE
        No XY warm-start. Standard validated RAG-TtBA from iteration 0.

    RAG_XY_WS1
        Use exactly 1 accepted early XY update, then permanently switch
        to standard RAG-TtBA.

    RAG_XY_WS2
        Use exactly 2 accepted early XY updates, then permanently switch
        to standard RAG-TtBA.

    RAG_XY_WS3
        Use exactly 3 accepted early XY updates, then permanently switch
        to standard RAG-TtBA.

    Important failure rule
    ----------------------
    During the warm-start phase, if an XY attempt fails to produce an
    improving adversarial point, the attack immediately:

        1) aborts XY warm-start permanently;
        2) runs standard RAG-TtBA in that SAME iteration.

    This prevents a failed warm-start from stalling the attack.

    Query accounting
    ----------------
    Every victim query is counted exactly once under:

        init
        normal
        bridge
        xy
        boundary

    Decoder/RAG pixel-L2 evaluation costs zero victim queries.

    Threat model
    ------------
    * hard-label top-1 victim access only;
    * no victim logits;
    * no victim gradients.
    """

    METHOD_TO_WARMSTART_UPDATES = {
        "RAG_TTBA_BASE": 0,
        "RAG_XY_WS1": 1,
        "RAG_XY_WS2": 2,
        "RAG_XY_WS3": 3,
    }

    SUPPORTED_WARMSTART = set(
        METHOD_TO_WARMSTART_UPDATES.keys()
    )

    def __init__(
        self,
        *args,
        attack_method="RAG_TTBA_BASE",
        xy_primary_points=None,
        xy_rescue_points=None,
        xy_accept_eps=1e-8,
        **kwargs,
    ):
        if attack_method not in self.SUPPORTED_WARMSTART:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported={sorted(self.SUPPORTED_WARMSTART)}"
            )

        # Parent validates its own method names, so initialize it as BASE.
        super().__init__(
            *args,
            attack_method="RAG_TTBA_BASE",
            **kwargs,
        )

        self.attack_method = attack_method
        self.xy_target_updates = int(
            self.METHOD_TO_WARMSTART_UPDATES[
                attack_method
            ]
        )

        # Same feasibility-first points that gave the strongest early XY
        # behavior in the controlled frontier experiment.
        self.xy_primary_points = (
            [
                (0.90, 0.10),
                (0.82, 0.20),
                (0.70, 0.30),
                (0.58, 0.48),
            ]
            if xy_primary_points is None
            else self._validate_xy_points(
                xy_primary_points
            )
        )

        self.xy_rescue_points = (
            [
                (0.96, 0.05),
                (0.92, 0.15),
                (0.86, 0.28),
                (0.75, 0.42),
            ]
            if xy_rescue_points is None
            else self._validate_xy_points(
                xy_rescue_points
            )
        )

        self.xy_accept_eps = float(
            xy_accept_eps
        )

        # Warm-start state.
        self.xy_accepted_updates = 0
        self.xy_attempted_updates = 0
        self.xy_aborted = False

        # Dedicated query bucket.
        old_q = self.q_breakdown
        self.q_breakdown = OrderedDict()

        for key, value in old_q.items():
            self.q_breakdown[key] = value

            if key == "bridge":
                self.q_breakdown["xy"] = 0

        if "xy" not in self.q_breakdown:
            self.q_breakdown["xy"] = 0

    # ==================================================================
    # Geometry helpers
    # ==================================================================
    @staticmethod
    def _validate_xy_points(
        points,
    ):
        out = []

        for item in points:
            if len(item) != 2:
                raise ValueError(
                    "Each XY point must contain exactly two values."
                )

            x = float(
                item[0]
            )
            y = float(
                item[1]
            )

            if x <= 0.0 or y < 0.0:
                raise ValueError(
                    "Warm-start searches Q1 only: require x>0 and y>=0."
                )

            rho = math.sqrt(
                x * x
                + y * y
            )

            if rho >= 1.0:
                raise ValueError(
                    f"XY point {(x, y)} must lie strictly inside "
                    "the unit-radius disk."
                )

            out.append(
                (x, y)
            )

        return out

    def _build_xy_basis(
        self,
        z_b,
        eta,
    ):
        v = (
            z_b - self.z_source
        ) / (
            torch.norm(
                z_b - self.z_source
            )
            + self.eps
        )

        n = eta / (
            torch.norm(
                eta
            )
            + self.eps
        )

        e_x = v

        n_x = torch.sum(
            n * e_x
        )

        n_res = (
            n - n_x * e_x
        )

        n_res_norm = torch.norm(
            n_res
        )

        if (
            not torch.isfinite(
                n_res_norm
            )
            or n_res_norm.item()
            < 1e-10
        ):
            return None

        e_y = (
            n_res
            / (
                n_res_norm
                + self.eps
            )
        )

        n_y = torch.sum(
            n * e_y
        )

        phi_deg = math.degrees(
            math.atan2(
                max(
                    0.0,
                    float(
                        n_y.item()
                    ),
                ),
                float(
                    n_x.item()
                ),
            )
        )

        phi_deg = max(
            0.0,
            min(
                180.0,
                phi_deg,
            ),
        )

        return {
            "e_x": e_x.detach(),
            "e_y": e_y.detach(),
            "phi_deg": float(
                phi_deg
            ),
        }

    @staticmethod
    def _polar_from_xy(
        x,
        y,
    ):
        rho = math.sqrt(
            float(x) * float(x)
            + float(y) * float(y)
        )

        theta_deg = math.degrees(
            math.atan2(
                float(y),
                float(x),
            )
        )

        return (
            float(rho),
            float(
                theta_deg
                % 360.0
            ),
        )

    def _make_xy_candidate(
        self,
        radius,
        basis,
        x,
        y,
    ):
        return (
            self.z_source
            + radius
            * (
                float(x)
                * basis["e_x"]
                + float(y)
                * basis["e_y"]
            )
        )

    # ==================================================================
    # Feasibility-first XY warm-start step
    # ==================================================================
    def _query_xy_set(
        self,
        points,
        radius,
        basis,
        stage,
    ):
        rows = []

        for x, y in points:
            if self.remaining_queries() <= 0:
                break

            rho, theta_deg = self._polar_from_xy(
                x,
                y,
            )

            if rho >= 1.0:
                continue

            z = self._make_xy_candidate(
                radius,
                basis,
                x,
                y,
            )

            status = self.is_adversarial(
                z,
                category="xy",
            )

            row = {
                "stage": str(
                    stage
                ),
                "x": float(
                    x
                ),
                "y": float(
                    y
                ),
                "rho": float(
                    rho
                ),
                "theta_deg": float(
                    theta_deg
                ),
                "status": int(
                    status
                ),
                "z": z.detach(),
                "pixel_l2": None,
            }

            # RAG score only after hard-label adversarial feasibility
            # is verified.
            if status == 1:
                row[
                    "pixel_l2"
                ] = float(
                    self.rag_pixel_l2(
                        z
                    )
                )

            rows.append(
                row
            )

        return rows

    @staticmethod
    def _verified_adv(
        rows,
    ):
        return [
            row
            for row in rows
            if row[
                "status"
            ] == 1
        ]

    def _choose_best_verified_adv(
        self,
        rows,
    ):
        adv = self._verified_adv(
            rows
        )

        if not adv:
            return None

        return min(
            adv,
            key=lambda row: (
                row[
                    "pixel_l2"
                ],
                row[
                    "rho"
                ],
            ),
        )

    def _step_xy_warmstart(
        self,
        z_b,
        eta,
    ):
        self.xy_attempted_updates += 1

        before_l2 = float(
            self.rag_pixel_l2(
                z_b
            )
        )

        radius = torch.norm(
            z_b
            - self.z_source
        )

        if radius.item() < 1e-12:
            return z_b.detach(), {
                "xy_found": 0,
                "xy_accepted": 0,
                "xy_reason": "ZERO_RADIUS",
                "xy_primary_queries": 0,
                "xy_rescue_queries": 0,
                "xy_total_queries": 0,
            }

        basis = self._build_xy_basis(
            z_b,
            eta,
        )

        if basis is None:
            return z_b.detach(), {
                "xy_found": 0,
                "xy_accepted": 0,
                "xy_reason": "DEGENERATE_XY_BASIS",
                "xy_primary_queries": 0,
                "xy_rescue_queries": 0,
                "xy_total_queries": 0,
            }

        q_before = int(
            self.q_breakdown[
                "xy"
            ]
        )

        primary_rows = self._query_xy_set(
            self.xy_primary_points,
            radius,
            basis,
            stage="PRIMARY",
        )

        best = self._choose_best_verified_adv(
            primary_rows
        )

        rescue_rows = []

        if (
            best is None
            and self.remaining_queries() > 0
        ):
            rescue_rows = self._query_xy_set(
                self.xy_rescue_points,
                radius,
                basis,
                stage="RESCUE",
            )

            best = self._choose_best_verified_adv(
                rescue_rows
            )

        q_after = int(
            self.q_breakdown[
                "xy"
            ]
        )

        all_rows = (
            primary_rows
            + rescue_rows
        )

        num_adv = len(
            self._verified_adv(
                all_rows
            )
        )

        q_xy = int(
            q_after
            - q_before
        )

        if best is None:
            return z_b.detach(), {
                "xy_found": 0,
                "xy_accepted": 0,
                "xy_reason": "NO_ADVERSARIAL_XY_POINT",
                "xy_phi_deg": float(
                    basis[
                        "phi_deg"
                    ]
                ),
                "xy_primary_queries": int(
                    len(
                        primary_rows
                    )
                ),
                "xy_rescue_queries": int(
                    len(
                        rescue_rows
                    )
                ),
                "xy_total_queries": int(
                    q_xy
                ),
                "xy_num_verified_adv": int(
                    num_adv
                ),
                "xy_selected_stage": None,
                "xy_selected_x": None,
                "xy_selected_y": None,
                "xy_selected_rho": None,
                "xy_selected_theta_deg": None,
                "xy_selected_pixel_l2": None,
                "xy_refined_pixel_l2": float(
                    before_l2
                ),
                "xy_before_pixel_l2": float(
                    before_l2
                ),
                "xy_in_full_ttba_arc": None,
            }

        # Selected endpoint was just verified adversarial.
        # The source is known clean.
        refined = self._binary_search_known_adv(
            self.z_source,
            best[
                "z"
            ],
            category="boundary",
            max_steps=100,
        )

        refined_l2 = float(
            self.rag_pixel_l2(
                refined
            )
        )

        accepted = bool(
            refined_l2
            < before_l2
            - self.xy_accept_eps
        )

        if accepted:
            z_next = refined.detach()
            self.xy_accepted_updates += 1
        else:
            z_next = z_b.detach()

        theta = float(
            best[
                "theta_deg"
            ]
        )

        return z_next, {
            "xy_found": 1,
            "xy_accepted": int(
                accepted
            ),
            "xy_reason": (
                "IMPROVED"
                if accepted
                else "REFINED_POINT_DID_NOT_IMPROVE"
            ),
            "xy_phi_deg": float(
                basis[
                    "phi_deg"
                ]
            ),
            "xy_primary_queries": int(
                len(
                    primary_rows
                )
            ),
            "xy_rescue_queries": int(
                len(
                    rescue_rows
                )
            ),
            "xy_total_queries": int(
                q_xy
            ),
            "xy_num_verified_adv": int(
                num_adv
            ),
            "xy_selected_stage": str(
                best[
                    "stage"
                ]
            ),
            "xy_selected_x": float(
                best[
                    "x"
                ]
            ),
            "xy_selected_y": float(
                best[
                    "y"
                ]
            ),
            "xy_selected_rho": float(
                best[
                    "rho"
                ]
            ),
            "xy_selected_theta_deg": float(
                best[
                    "theta_deg"
                ]
            ),
            "xy_selected_pixel_l2": float(
                best[
                    "pixel_l2"
                ]
            ),
            "xy_refined_pixel_l2": float(
                refined_l2
            ),
            "xy_before_pixel_l2": float(
                before_l2
            ),
            "xy_in_full_ttba_arc": bool(
                theta
                <= basis[
                    "phi_deg"
                ]
                + 1e-8
            ),
        }

    # ==================================================================
    # Phase logic
    # ==================================================================
    def _xy_phase_active(
        self,
    ):
        return bool(
            self.xy_target_updates > 0
            and not self.xy_aborted
            and self.xy_accepted_updates
            < self.xy_target_updates
        )

    # ==================================================================
    # Main attack
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

            xy_phase_before = self._xy_phase_active()

            if xy_phase_before:
                # Leave room for up to 8 XY probes plus one boundary refine.
                reserve = max(
                    self.update_query_reserve,
                    len(
                        self.xy_primary_points
                    )
                    + len(
                        self.xy_rescue_points
                    )
                    + 10,
                )
            else:
                reserve = self.update_query_reserve

            q_full = min(
                q_full,
                max(
                    0,
                    self.remaining_queries()
                    - reserve,
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

            bridge_info = None
            xy_info = {
                "xy_found": 0,
                "xy_accepted": 0,
                "xy_reason": "NOT_IN_XY_WARMSTART",
                "xy_primary_queries": 0,
                "xy_rescue_queries": 0,
                "xy_total_queries": 0,
                "xy_num_verified_adv": 0,
                "xy_selected_stage": None,
                "xy_selected_x": None,
                "xy_selected_y": None,
                "xy_selected_rho": None,
                "xy_selected_theta_deg": None,
                "xy_selected_pixel_l2": None,
                "xy_refined_pixel_l2": None,
                "xy_phi_deg": None,
                "xy_in_full_ttba_arc": None,
            }

            used_xy = 0
            used_ttba = 0
            xy_failed_then_ttba = 0

            if xy_phase_before:
                used_xy = 1

                z_xy, xy_info = self._step_xy_warmstart(
                    z_b,
                    eta,
                )

                if bool(
                    xy_info.get(
                        "xy_accepted",
                        0,
                    )
                ):
                    z_next = z_xy

                else:
                    # Critical safety rule:
                    # abort XY forever and run TtBA immediately in the SAME
                    # iteration so a failed warm-start cannot stall progress.
                    self.xy_aborted = True
                    xy_failed_then_ttba = 1
                    used_ttba = 1

                    if self.remaining_queries() > 0:
                        z_next, bridge_info = self.step_rag_ttba(
                            self.z_source,
                            eta,
                            z_b,
                        )
                    else:
                        z_next = z_b.detach()

            else:
                used_ttba = 1

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

            phase_after = (
                "XY_WARMSTART"
                if self._xy_phase_active()
                else "TTBA"
            )

            diag = {
                "iteration": int(
                    i
                ),
                "query": int(
                    self.all_queries
                ),
                "method": self.attack_method,
                "xy_target_updates": int(
                    self.xy_target_updates
                ),
                "xy_accepted_updates": int(
                    self.xy_accepted_updates
                ),
                "xy_attempted_updates": int(
                    self.xy_attempted_updates
                ),
                "xy_aborted": int(
                    self.xy_aborted
                ),
                "phase_before": (
                    "XY_WARMSTART"
                    if xy_phase_before
                    else "TTBA"
                ),
                "phase_after": str(
                    phase_after
                ),
                "used_xy": int(
                    used_xy
                ),
                "used_ttba": int(
                    used_ttba
                ),
                "xy_failed_then_ttba": int(
                    xy_failed_then_ttba
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

            diag.update(
                xy_info
            )

            self.diagnostics.append(
                diag
            )

            if self.verbose == "Yes":
                if used_xy:
                    print(
                        "         XY-WARMSTART: "
                        f"target={self.xy_target_updates} "
                        f"acceptedTotal={self.xy_accepted_updates} "
                        f"attemptedTotal={self.xy_attempted_updates} "
                        f"found={xy_info.get('xy_found', 0)} "
                        f"accepted={xy_info.get('xy_accepted', 0)} "
                        f"advPts={xy_info.get('xy_num_verified_adv', 0)} "
                        f"qXY={xy_info.get('xy_total_queries', 0)} "
                        f"x={xy_info.get('xy_selected_x')} "
                        f"y={xy_info.get('xy_selected_y')} "
                        f"rho={xy_info.get('xy_selected_rho')} "
                        f"theta={xy_info.get('xy_selected_theta_deg')} "
                        f"abort={int(self.xy_aborted)} "
                        f"sameIterTTBA={xy_failed_then_ttba}"
                    )

                else:
                    print(
                        "         XY-WARMSTART: TTBA_ONLY "
                        f"target={self.xy_target_updates} "
                        f"acceptedTotal={self.xy_accepted_updates} "
                        f"L={bridge_info['L']:.4f} "
                        f"k={bridge_info['k_app']:.4f}"
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
            "xy_target_updates": int(
                self.xy_target_updates
            ),
            "xy_accepted_updates": int(
                self.xy_accepted_updates
            ),
            "xy_attempted_updates": int(
                self.xy_attempted_updates
            ),
            "xy_aborted": int(
                self.xy_aborted
            ),
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
        }
