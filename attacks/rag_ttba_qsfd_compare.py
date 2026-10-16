
import math
from collections import OrderedDict

import numpy as np
import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBAQSFDCompareAttack(RAGTtBAPGRAttack):
    """
    Query-corrected comparison of three methods under the same RAG latent frame:

      1) RAG_TTBA_BASE
         Validated TtBA baseline.

      2) RAG_QSFD
         Standalone QS-FD core:
           - same hard-label Gaussian normal estimator as BASE;
           - K directions in a 90-degree sector of span{eta, v};
           - each direction is taken to an APPROXIMATE queried boundary;
           - choose the candidate with the smallest FREE RAG pixel L2;
           - fully refine only the winner;
           - keep the current boundary if no candidate improves it.

      3) RAG_TTBA_QSFD
         TtBA + QS-FD:
           - run the exact TtBA bridge threshold search to obtain L and k_app;
           - create K local TtBA rays around k_app, all clipped to [0, L];
           - query an approximate radial boundary on every candidate ray;
           - choose the best queried boundary candidate by FREE RAG pixel L2;
           - fully refine only the winner;
           - keep the current boundary if no candidate improves it.

    Why this code is "query corrected"
    ----------------------------------
    The supplied student QS code internally called is_adversarial() and
    bin_search(), but returned (best_adv, 0), while the outer q_num was updated
    only by the returned qs value. Therefore the reported 500/1000/2000
    checkpoints could undercount QS victim-model calls.

    Here there is only ONE authoritative counter: self.all_queries.
    Every classifier evaluation goes through _query_labels()/is_adversarial()
    and is counted exactly once.

    Another correction
    ------------------
    The supplied QS implementation initialized a radial binary search with
    hi=R without first proving that the endpoint at R was adversarial.
    Binary search requires a clean/adversarial bracket. This implementation
    first verifies an adversarial endpoint (and, if needed, expands outward),
    then performs the coarse radial binary search.

    Threat model
    ------------
    Strict decision-based black-box:
      - top-1 hard labels only;
      - no victim logits/probabilities;
      - no victim gradients/backward;
      - RAG decoder metrics are free attack-side computations.
    """

    METHOD_BASE = "RAG_TTBA_BASE"
    METHOD_QSFD = "RAG_QSFD"
    METHOD_COMBO = "RAG_TTBA_QSFD"

    SUPPORTED_COMPARE = {
        METHOD_BASE,
        METHOD_QSFD,
        METHOD_COMBO,
    }

    def __init__(
        self,
        *args,
        attack_method="RAG_TTBA_BASE",
        # QS-FD screening
        qsfd_k=3,
        qsfd_sector_deg=90.0,
        qsfd_coarse_steps=6,
        qsfd_expand_factors=(1.0, 1.25, 1.50),
        qsfd_accept_eps=1e-8,
        # TtBA + QS-FD local ray span around k_app.
        qsfd_combo_span=0.25,
        # Optional adaptive-K support. Keep "fixed" for the first fair smoke.
        qsfd_k_mode="fixed",
        qsfd_k_early=3,
        qsfd_k_late=5,
        qsfd_switch_ssim=0.995,
        # Safety: do not start a QS update without at least this many
        # post-normal queries still available.
        qsfd_min_post_normal_reserve=12,
        **kwargs,
    ):
        if attack_method not in self.SUPPORTED_COMPARE:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported={sorted(self.SUPPORTED_COMPARE)}"
            )

        requested_method = str(attack_method)

        # Parent must remain on the validated BASE normal-estimation path.
        super().__init__(
            *args,
            attack_method=self.METHOD_BASE,
            **kwargs,
        )

        self.requested_attack_method = requested_method

        self.qsfd_k = int(qsfd_k)
        if self.qsfd_k < 2:
            raise ValueError("qsfd_k must be >= 2.")

        self.qsfd_sector_deg = float(qsfd_sector_deg)
        if not (0.0 < self.qsfd_sector_deg <= 180.0):
            raise ValueError("qsfd_sector_deg must be in (0, 180].")

        self.qsfd_coarse_steps = int(qsfd_coarse_steps)
        if self.qsfd_coarse_steps < 1:
            raise ValueError("qsfd_coarse_steps must be >= 1.")

        self.qsfd_expand_factors = tuple(
            float(x) for x in qsfd_expand_factors
        )
        if not self.qsfd_expand_factors:
            raise ValueError("qsfd_expand_factors cannot be empty.")
        if any(x <= 0.0 for x in self.qsfd_expand_factors):
            raise ValueError("All qsfd_expand_factors must be > 0.")

        self.qsfd_accept_eps = float(qsfd_accept_eps)

        self.qsfd_combo_span = float(qsfd_combo_span)
        if self.qsfd_combo_span < 0.0:
            raise ValueError("qsfd_combo_span must be >= 0.")

        self.qsfd_k_mode = str(qsfd_k_mode).lower()
        if self.qsfd_k_mode not in {"fixed", "adaptive"}:
            raise ValueError(
                "qsfd_k_mode must be 'fixed' or 'adaptive'."
            )

        self.qsfd_k_early = max(2, int(qsfd_k_early))
        self.qsfd_k_late = max(2, int(qsfd_k_late))
        self.qsfd_switch_ssim = float(qsfd_switch_ssim)
        self.qsfd_min_post_normal_reserve = max(
            1, int(qsfd_min_post_normal_reserve)
        )

        # Extend the parent's exact query accounting.
        if "qsfd" not in self.q_breakdown:
            self.q_breakdown["qsfd"] = 0

        self.qsfd_diagnostics = []

    # ==================================================================
    # Basic geometry helpers
    # ==================================================================
    def _unit(self, x):
        n = torch.norm(x)
        if (
            not torch.isfinite(n)
            or n.item() < self.eps
        ):
            return None
        return x / (n + self.eps)

    def _free_ssim(self, z):
        with torch.no_grad():
            x = self.model.get_pixel_image(z)
            return float(
                self.ssim_fn(
                    x,
                    self.x_orig,
                    data_range=1.0,
                ).detach().item()
            )

    def _current_k(self, z_b):
        if self.qsfd_k_mode == "fixed":
            return int(self.qsfd_k)

        ssim_value = self._free_ssim(z_b)
        if ssim_value >= self.qsfd_switch_ssim:
            return int(self.qsfd_k_late)

        return int(self.qsfd_k_early)

    def _sector_directions(
        self,
        eta,
        v,
        k,
    ):
        """
        Corrected version of the supplied QS plane rotation.

        e1 = normalized eta.
        e2 = normalized component of v orthogonal to e1.

        Then directions are uniformly sampled in
            phi in [0, sector_deg].

        The current v-ray is handled separately as the incumbent boundary,
        so QS-FD can never be forced to accept a worse candidate.
        """
        e1 = self._unit(eta)
        v_u = self._unit(v)

        if e1 is None or v_u is None:
            return []

        proj = torch.sum(v_u * e1) * e1
        e2_raw = v_u - proj
        e2 = self._unit(e2_raw)

        if e2 is None:
            # eta and v are nearly collinear. There is no reliable 2-D sector.
            return [e1.detach()]

        # Orient e2 toward v.
        if torch.sum(e2 * v_u).item() < 0.0:
            e2 = -e2

        sector = math.radians(self.qsfd_sector_deg)

        if k == 1:
            phis = [0.0]
        else:
            phis = np.linspace(
                0.0,
                sector,
                num=int(k),
                endpoint=True,
            ).tolist()

        directions = []

        for phi in phis:
            d = (
                math.cos(float(phi)) * e1
                + math.sin(float(phi)) * e2
            )
            d = self._unit(d)

            if d is not None:
                directions.append(
                    {
                        "direction": d.detach(),
                        "phi_deg": float(
                            math.degrees(float(phi))
                        ),
                    }
                )

        return directions

    # ==================================================================
    # Query-correct radial boundary search
    # ==================================================================
    def _coarse_boundary_on_ray(
        self,
        direction,
        base_radius,
    ):
        """
        Find a valid adversarial endpoint on this ray, then run a SMALL
        clean/adv radial binary search.

        z(lambda) = z_source + lambda * direction

        z_source is a known clean source.
        The upper endpoint is queried before binary search.

        Returns a VERIFIED adversarial coarse-boundary point or None.
        """
        direction = self._unit(direction)
        if direction is None:
            return None, {
                "success": False,
                "reason": "ZERO_DIRECTION",
                "queries": 0,
            }

        q_before = int(self.all_queries)

        adv_point = None
        adv_radius = None
        endpoint_factor = None

        # First establish a valid adversarial upper bracket.
        for factor in self.qsfd_expand_factors:
            if self.remaining_queries() <= 0:
                break

            radius = float(base_radius) * float(factor)
            point = (
                self.z_source
                + radius * direction
            )

            status = self.is_adversarial(
                point,
                category="qsfd",
            )

            if status == 0:
                break

            if status == 1:
                adv_point = point.detach()
                adv_radius = float(radius)
                endpoint_factor = float(factor)
                break

        if adv_point is None:
            return None, {
                "success": False,
                "reason": "NO_ADV_ENDPOINT",
                "queries": int(
                    self.all_queries - q_before
                ),
            }

        clean_radius = 0.0
        clean_point = self.z_source.detach()
        adv = adv_point.detach()

        # Coarse binary search. The invariant is:
        #   clean_point = known clean
        #   adv         = known adversarial
        for _ in range(self.qsfd_coarse_steps):
            if self.remaining_queries() <= 0:
                break

            mid_radius = 0.5 * (
                clean_radius + adv_radius
            )

            mid = (
                self.z_source
                + mid_radius * direction
            )

            status = self.is_adversarial(
                mid,
                category="qsfd",
            )

            if status == 0:
                break

            if status == 1:
                adv = mid.detach()
                adv_radius = float(mid_radius)
            else:
                clean_point = mid.detach()
                clean_radius = float(mid_radius)

        return adv.detach(), {
            "success": True,
            "reason": "OK",
            "queries": int(
                self.all_queries - q_before
            ),
            "endpoint_factor": float(
                endpoint_factor
            ),
            "coarse_adv_radius": float(
                adv_radius
            ),
            "coarse_clean_radius": float(
                clean_radius
            ),
            "coarse_gap": float(
                adv_radius - clean_radius
            ),
        }

    # ==================================================================
    # Corrected standalone QS-FD
    # ==================================================================
    def step_qsfd(
        self,
        x_s,
        eta,
        x_b,
    ):
        q_before = int(self.all_queries)

        radius = float(
            torch.norm(x_b - x_s).item()
        )

        if radius < self.eps:
            return x_b.detach(), {
                "mode": "QSFD",
                "selector_queries": 0,
                "candidate_count": 0,
                "selected": "INCUMBENT",
            }

        v = self._unit(
            x_b - x_s
        )

        if v is None:
            return x_b.detach(), {
                "mode": "QSFD",
                "selector_queries": 0,
                "candidate_count": 0,
                "selected": "INCUMBENT",
            }

        k = self._current_k(x_b)
        rays = self._sector_directions(
            eta=eta,
            v=v,
            k=k,
        )

        incumbent_l2 = self.rag_pixel_l2(
            x_b
        )

        candidates = []

        for j, item in enumerate(rays):
            # Leave at least one query for a possible winner refinement.
            if self.remaining_queries() <= 1:
                break

            coarse, info = self._coarse_boundary_on_ray(
                direction=item["direction"],
                base_radius=radius,
            )

            record = {
                "index": int(j),
                "phi_deg": float(
                    item["phi_deg"]
                ),
                **info,
            }

            if coarse is not None:
                pixel_l2 = self.rag_pixel_l2(
                    coarse
                )

                record["coarse_pixel_l2"] = float(
                    pixel_l2
                )

                candidates.append(
                    {
                        "z": coarse.detach(),
                        "pixel_l2": float(
                            pixel_l2
                        ),
                        "record": record,
                    }
                )

            else:
                record["coarse_pixel_l2"] = None

        if not candidates:
            return x_b.detach(), {
                "mode": "QSFD",
                "selector_queries": int(
                    self.all_queries - q_before
                ),
                "candidate_count": 0,
                "selected": "INCUMBENT",
                "incumbent_pixel_l2": float(
                    incumbent_l2
                ),
                "candidates": [],
            }

        best = min(
            candidates,
            key=lambda item: item["pixel_l2"],
        )

        # Fully refine only the winner. Its endpoint is already verified adv.
        refined = self._binary_search_known_adv(
            x_s,
            best["z"],
            category="boundary",
            max_steps=100,
        )

        refined_l2 = self.rag_pixel_l2(
            refined
        )

        if (
            refined_l2
            < incumbent_l2 - self.qsfd_accept_eps
        ):
            selected = "QSFD"
            z_out = refined.detach()
        else:
            selected = "INCUMBENT"
            z_out = x_b.detach()

        return z_out, {
            "mode": "QSFD",
            "selector_queries": int(
                self.all_queries - q_before
            ),
            "candidate_count": int(
                len(candidates)
            ),
            "selected": selected,
            "incumbent_pixel_l2": float(
                incumbent_l2
            ),
            "winner_coarse_pixel_l2": float(
                best["pixel_l2"]
            ),
            "winner_refined_pixel_l2": float(
                refined_l2
            ),
            "winner_phi_deg": float(
                best["record"]["phi_deg"]
            ),
            "candidates": [
                item["record"]
                for item in candidates
            ],
        }

    # ==================================================================
    # TtBA threshold without final refinement
    # ==================================================================
    def _ttba_threshold_only(
        self,
        x_s,
        eta,
        x_b,
    ):
        radius = torch.norm(
            x_b - x_s
        )

        if radius.item() < self.eps:
            return {
                "L": 0.0,
                "k_app": 0.0,
                "radius": float(
                    radius.item()
                ),
                "queries": 0,
            }

        q_before = int(self.all_queries)

        L = 0.0
        R = 1.0

        while abs(R - L) > self.ttba_bridge_tol:
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

            if status == -1:
                R = mid
            else:
                L = mid

        k_app = self._ttba_piecewise_k(
            L
        )

        return {
            "L": float(L),
            "k_app": float(k_app),
            "radius": float(
                radius.item()
            ),
            "queries": int(
                self.all_queries - q_before
            ),
        }

    def _combo_candidate_ks(
        self,
        k_app,
        L,
        k,
    ):
        """
        K local rays around k_app.

        We keep every candidate inside [0, L], where the TtBA bridge search
        has evidence for the adversarial side at the current radius under the
        usual monotonic bridge assumption.

        k_app is always included.
        """
        k_app = float(
            max(0.0, min(1.0, k_app))
        )
        L = float(
            max(0.0, min(1.0, L))
        )

        if k <= 1:
            return [k_app]

        if k_app <= self.eps:
            raw = np.linspace(
                0.0,
                max(L, self.eps),
                num=k,
            )
        else:
            offsets = np.linspace(
                -self.qsfd_combo_span,
                self.qsfd_combo_span,
                num=k,
            )

            raw = k_app * (
                1.0 + offsets
            )

        values = [
            float(
                max(
                    0.0,
                    min(L, float(x)),
                )
            )
            for x in raw
        ]

        values.append(
            float(
                max(0.0, min(L, k_app))
            )
        )

        # Stable unique ordering.
        unique = []
        for value in sorted(values):
            if (
                not unique
                or abs(value - unique[-1]) > 1e-8
            ):
                unique.append(value)

        return unique

    # ==================================================================
    # TtBA + QS-FD
    # ==================================================================
    def step_ttba_qsfd(
        self,
        x_s,
        eta,
        x_b,
    ):
        q_before = int(self.all_queries)

        threshold = self._ttba_threshold_only(
            x_s=x_s,
            eta=eta,
            x_b=x_b,
        )

        L = float(
            threshold["L"]
        )
        k_app = float(
            threshold["k_app"]
        )
        radius = float(
            threshold["radius"]
        )

        incumbent_l2 = self.rag_pixel_l2(
            x_b
        )

        if radius < self.eps:
            return x_b.detach(), {
                "mode": "TTBA_QSFD",
                "L": L,
                "k_app": k_app,
                "selector_queries": int(
                    self.all_queries - q_before
                ),
                "candidate_count": 0,
                "selected": "INCUMBENT",
            }

        k_count = self._current_k(
            x_b
        )

        candidate_ks = self._combo_candidate_ks(
            k_app=k_app,
            L=L,
            k=k_count,
        )

        candidates = []

        for j, kval in enumerate(
            candidate_ks
        ):
            if self.remaining_queries() <= 1:
                break

            d = self._ttba_attempt_unit(
                x_s,
                x_b,
                eta,
                float(kval),
            )

            coarse, info = self._coarse_boundary_on_ray(
                direction=d,
                base_radius=radius,
            )

            record = {
                "index": int(j),
                "k": float(kval),
                **info,
            }

            if coarse is not None:
                pixel_l2 = self.rag_pixel_l2(
                    coarse
                )

                record[
                    "coarse_pixel_l2"
                ] = float(
                    pixel_l2
                )

                candidates.append(
                    {
                        "z": coarse.detach(),
                        "pixel_l2": float(
                            pixel_l2
                        ),
                        "record": record,
                    }
                )
            else:
                record[
                    "coarse_pixel_l2"
                ] = None

        if not candidates:
            return x_b.detach(), {
                "mode": "TTBA_QSFD",
                "L": L,
                "k_app": k_app,
                "bridge_threshold_queries": int(
                    threshold["queries"]
                ),
                "selector_queries": int(
                    self.all_queries - q_before
                ),
                "candidate_count": 0,
                "selected": "INCUMBENT",
                "candidate_ks": [
                    float(x)
                    for x in candidate_ks
                ],
            }

        best = min(
            candidates,
            key=lambda item: item[
                "pixel_l2"
            ],
        )

        refined = self._binary_search_known_adv(
            x_s,
            best["z"],
            category="boundary",
            max_steps=100,
        )

        refined_l2 = self.rag_pixel_l2(
            refined
        )

        if (
            refined_l2
            < incumbent_l2 - self.qsfd_accept_eps
        ):
            selected = "TTBA_QSFD"
            z_out = refined.detach()
        else:
            selected = "INCUMBENT"
            z_out = x_b.detach()

        return z_out, {
            "mode": "TTBA_QSFD",
            "L": L,
            "k_app": k_app,
            "bridge_threshold_queries": int(
                threshold["queries"]
            ),
            "selector_queries": int(
                self.all_queries - q_before
            ),
            "candidate_count": int(
                len(candidates)
            ),
            "selected": selected,
            "incumbent_pixel_l2": float(
                incumbent_l2
            ),
            "winner_k": float(
                best["record"]["k"]
            ),
            "winner_coarse_pixel_l2": float(
                best["pixel_l2"]
            ),
            "winner_refined_pixel_l2": float(
                refined_l2
            ),
            "candidate_ks": [
                float(x)
                for x in candidate_ks
            ],
            "candidates": [
                item["record"]
                for item in candidates
            ],
        }

    # ==================================================================
    # Normal estimator
    # ==================================================================
    def _choose_base_normal_with_qs_reserve(
        self,
        z_boundary,
        iteration_index,
    ):
        """
        Same fresh Gaussian estimator as BASE, but for QS methods we keep a
        small post-normal reserve so the iteration cannot spend its entire
        remaining budget on the normal and leave no selector query.

        Early/mid attack q_full is identical to BASE. This only matters near
        budget exhaustion.
        """
        q_full = int(
            self.N0
            * math.sqrt(
                iteration_index + 1.0
            )
        )

        q_full = max(
            1,
            q_full,
        )

        reserve = max(
            self.update_query_reserve,
            self.qsfd_min_post_normal_reserve,
        )

        available = max(
            0,
            self.remaining_queries()
            - reserve,
        )

        q_target = min(
            q_full,
            available,
        )

        if q_target <= 0:
            return None, None

        eta, q_used = self.estimate_normal(
            z_boundary,
            q_target,
        )

        return eta, {
            "action": "FULL_REFRESH",
            "q_full": int(
                q_full
            ),
            "q_target": int(
                q_target
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
    # Main comparison attack
    # ==================================================================
    def attack(self):
        # Exact validated reference.
        if self.requested_attack_method == self.METHOD_BASE:
            result = super().attack()
            result[
                "requested_attack_method"
            ] = self.METHOD_BASE
            result[
                "qsfd_diagnostics"
            ] = []
            return result

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
                "qsfd_diagnostics":
                    self.qsfd_diagnostics,
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
                <= self.qsfd_min_post_normal_reserve
            ):
                break

            before_l2 = self.rag_pixel_l2(
                z_b
            )

            eta, normal_info = (
                self._choose_base_normal_with_qs_reserve(
                    z_boundary=z_b,
                    iteration_index=i,
                )
            )

            if eta is None:
                break

            if self.remaining_queries() <= 0:
                break

            if (
                self.requested_attack_method
                == self.METHOD_QSFD
            ):
                z_candidate, step_info = self.step_qsfd(
                    self.z_source,
                    eta,
                    z_b,
                )

            elif (
                self.requested_attack_method
                == self.METHOD_COMBO
            ):
                z_candidate, step_info = (
                    self.step_ttba_qsfd(
                        self.z_source,
                        eta,
                        z_b,
                    )
                )

            else:
                raise RuntimeError(
                    "Unknown requested attack method."
                )

            candidate_l2 = self.rag_pixel_l2(
                z_candidate
            )

            # The QS methods are explicitly improvement-safe.
            if (
                candidate_l2
                < before_l2 - self.qsfd_accept_eps
            ):
                accepted = True
                z_next = z_candidate.detach()
                after_l2 = float(
                    candidate_l2
                )
            else:
                accepted = False
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

            diag = {
                "iteration": int(i),
                "query": int(
                    self.all_queries
                ),
                "method":
                    self.requested_attack_method,
                "pixel_l2_before": float(
                    before_l2
                ),
                "pixel_l2_candidate": float(
                    candidate_l2
                ),
                "pixel_l2_after": float(
                    after_l2
                ),
                "progress": float(
                    progress
                ),
                "accepted": int(
                    accepted
                ),
            }

            if normal_info is not None:
                diag.update(
                    {
                        f"normal_{key}": value
                        for key, value
                        in normal_info.items()
                    }
                )

            # Save a compact version in the normal diagnostic stream.
            compact_step = {
                key: value
                for key, value
                in step_info.items()
                if key != "candidates"
            }

            diag.update(
                {
                    f"step_{key}": value
                    for key, value
                    in compact_step.items()
                }
            )

            self.diagnostics.append(
                diag
            )

            self.qsfd_diagnostics.append(
                {
                    "iteration": int(i),
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
                print(
                    "         QSFD: "
                    f"method={self.requested_attack_method} "
                    f"qN={normal_info['q_used']}/"
                    f"{normal_info['q_target']} "
                    f"qSelector="
                    f"{step_info.get('selector_queries', 0)} "
                    f"cand="
                    f"{step_info.get('candidate_count', 0)} "
                    f"selected="
                    f"{step_info.get('selected')} "
                    f"progress={progress:.4f} "
                    f"accepted={int(accepted)}"
                )

                if (
                    self.requested_attack_method
                    == self.METHOD_COMBO
                ):
                    print(
                        "               "
                        f"L={step_info.get('L', 0.0):.4f} "
                        f"k_app="
                        f"{step_info.get('k_app', 0.0):.4f} "
                        f"winner_k="
                        f"{step_info.get('winner_k')}"
                    )

        best = min(
            self.history,
            key=lambda row: row[
                "l2_pix"
            ],
        )

        final = {
            "query": int(
                best["query"]
            ),
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
            "qsfd_diagnostics":
                self.qsfd_diagnostics,
            "requested_attack_method":
                self.requested_attack_method,
        }
