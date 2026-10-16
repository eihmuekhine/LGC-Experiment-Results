import math

import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBASCBTCompareAttack(RAGTtBAPGRAttack):
    """
    Fair comparison of the validated RAG_TTBA_BASE against a diagnostic
    Shrinking-Circle Boundary Tracking variant (RAG_TTBA_SCBT).

    Geometry
    --------
    Let z_s be the source latent, z_b the current adversarial boundary point,
    r = ||z_b-z_s||, v = (z_b-z_s)/r, and n the estimated boundary normal.

    The current update is viewed in span{v,n}.  The radius-r circle centered
    at z_s contains the known adversarial point z_b.  SCBT first tests the
    pure-normal point z_s + r n.

      * If the pure-normal point is adversarial, it is used immediately as a
        verified adversarial ray endpoint; no v--n bridge bisection is needed.

      * If the pure-normal point is clean, SCBT performs a true angular
        bisection on the circle between the known adversarial v endpoint and
        the clean n endpoint.  The last verified adversarial angular point is
        retained.

    The remaining selector-query budget is then spent on radial bisection
    between the known-clean source and that verified adversarial ray endpoint.
    Each such radial probe is equivalently a query on a smaller concentric
    circle.  Finally, the ordinary boundary search finishes from the tightened
    clean/adversarial bracket.

    Query fairness
    --------------
    The default selector budget is 10 victim queries per iteration, matching
    the ~10 bridge queries used by BASE with ttba_bridge_tol=1e-3.  Therefore
    SCBT reallocates, rather than simply adds, selector queries:

        1 pure-normal endpoint query
        + up to scbt_arc_max_queries angular queries
        + remaining selector queries for shrinking-circle radial tightening
        <= scbt_selector_budget.

    Threat model
    ------------
    * top-1 hard labels only;
    * no victim logits/probabilities used by attack logic;
    * no victim gradients/backward;
    * every victim evaluation is counted exactly once;
    * RAG pixel L2 is decoder-side FREE feedback used only as a monotone
      acceptance safeguard for the experimental method.

    Important scientific note
    -------------------------
    This is a diagnostic implementation.  It does NOT assume that the normal
    endpoint is clean.  The endpoint is explicitly queried.  If n is still
    adversarial, SCBT exploits it directly; if n is clean, it brackets the
    angular transition safely.
    """

    METHOD_BASE = "RAG_TTBA_BASE"
    METHOD_SCBT = "RAG_TTBA_SCBT"

    SUPPORTED_SCBT = {
        METHOD_BASE,
        METHOD_SCBT,
    }

    def __init__(
        self,
        *args,
        attack_method="RAG_TTBA_BASE",
        scbt_selector_budget=10,
        scbt_arc_max_queries=5,
        scbt_arc_tol_deg=0.5,
        scbt_accept_eps=1e-8,
        scbt_monotone_safety=True,
        scbt_antipodal_dot=-0.95,
        **kwargs,
    ):
        if attack_method not in self.SUPPORTED_SCBT:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported={sorted(self.SUPPORTED_SCBT)}"
            )

        requested = str(attack_method)

        # Keep all shared machinery, especially the normal estimator, on the
        # exact validated BASE path.
        super().__init__(
            *args,
            attack_method=self.METHOD_BASE,
            **kwargs,
        )

        self.requested_attack_method = requested

        self.scbt_selector_budget = int(scbt_selector_budget)
        if self.scbt_selector_budget < 1:
            raise ValueError("scbt_selector_budget must be >= 1.")

        self.scbt_arc_max_queries = int(scbt_arc_max_queries)
        if self.scbt_arc_max_queries < 0:
            raise ValueError("scbt_arc_max_queries must be >= 0.")

        self.scbt_arc_tol = math.radians(float(scbt_arc_tol_deg))
        if self.scbt_arc_tol <= 0.0:
            raise ValueError("scbt_arc_tol_deg must be > 0.")

        self.scbt_accept_eps = float(scbt_accept_eps)
        self.scbt_monotone_safety = bool(scbt_monotone_safety)
        self.scbt_antipodal_dot = float(scbt_antipodal_dot)

        # Exact extra accounting category for the smaller-concentric-circle
        # radial probes performed before the ordinary boundary finisher.
        if "track" not in self.q_breakdown:
            self.q_breakdown["track"] = 0

        self.scbt_diagnostics = []

    # ==================================================================
    # Plane / angular helpers
    # ==================================================================
    def _build_vn_plane(self, x_s, x_b, eta):
        """
        Return an orthonormal representation of span{v,n}.

        e1 = v
        e2 = normalized component of n orthogonal to v
        theta_n is the exact angle from v to n in this basis.

        When n ~= v, theta_n=0 and e2 is unnecessary.
        When n ~= -v, the 2-D plane orientation is numerically ambiguous; the
        caller falls back to the exact BASE step.
        """
        delta = x_b - x_s
        radius_t = torch.norm(delta)

        if radius_t.item() <= self.eps:
            return None

        v = delta / (radius_t + self.eps)
        n = eta / (torch.norm(eta) + self.eps)

        dot = float(
            torch.clamp(
                (v * n).sum(),
                min=-1.0,
                max=1.0,
            ).item()
        )

        if dot <= self.scbt_antipodal_dot:
            return {
                "degenerate": True,
                "reason": "NEAR_ANTIPODAL_V_N",
                "radius": float(radius_t.item()),
                "dot_vn": dot,
            }

        tangent = n - dot * v
        tangent_norm = torch.norm(tangent)

        if tangent_norm.item() <= 1e-10:
            # n ~= v (the antipodal case was handled above).
            return {
                "degenerate": False,
                "radius": float(radius_t.item()),
                "v": v.detach(),
                "n": n.detach(),
                "e2": None,
                "theta_n": 0.0,
                "dot_vn": dot,
            }

        e2 = tangent / (tangent_norm + self.eps)
        theta_n = math.atan2(
            float(tangent_norm.item()),
            dot,
        )

        return {
            "degenerate": False,
            "radius": float(radius_t.item()),
            "v": v.detach(),
            "n": n.detach(),
            "e2": e2.detach(),
            "theta_n": float(theta_n),
            "dot_vn": dot,
        }

    def _direction_at_angle(self, plane, theta):
        """Unit direction on the v--n circular arc."""
        theta = float(theta)

        if plane["e2"] is None:
            return plane["v"].detach()

        d = (
            math.cos(theta) * plane["v"]
            + math.sin(theta) * plane["e2"]
        )

        return d / (torch.norm(d) + self.eps)

    # ==================================================================
    # Radial shrinking-circle bracket tightening
    # ==================================================================
    def _tighten_radial_bracket(
        self,
        z_clean,
        z_adv,
        max_queries,
    ):
        """
        Spend at most max_queries hard-label probes tightening a known
        clean/adversarial radial bracket.

        Every midpoint corresponds to a smaller concentric circle centered at
        z_s.  This is just bisection, which is the query-efficient choice for a
        one-dimensional hard-label threshold once the ray is fixed.
        """
        clean = z_clean.detach().clone()
        adv = z_adv.detach().clone()

        used = 0

        for _ in range(int(max_queries)):
            if self.remaining_queries() <= 0:
                break

            if torch.norm(adv - clean).item() <= self.tol:
                break

            mid = 0.5 * (clean + adv)
            status = self.is_adversarial(
                mid,
                category="track",
            )

            if status == 0:
                break

            used += 1

            if status == 1:
                adv = mid
            else:
                clean = mid

        return clean.detach(), adv.detach(), int(used)

    # ==================================================================
    # Proposed SCBT step
    # ==================================================================
    def _step_scbt(self, x_s, eta, x_b):
        q_start = int(self.all_queries)
        bridge_start = int(self.q_breakdown.get("bridge", 0))
        track_start = int(self.q_breakdown.get("track", 0))

        plane = self._build_vn_plane(
            x_s,
            x_b,
            eta,
        )

        if plane is None:
            return x_b.detach(), {
                "mode": self.METHOD_SCBT,
                "selection_reason": "ZERO_RADIUS",
                "selected": "INCUMBENT",
                "selector_queries": 0,
                "bridge_queries": 0,
                "track_queries": 0,
                "theta_n_deg": 0.0,
                "theta_selected_deg": 0.0,
                "normal_endpoint_status": 0,
                "fallback_base": 0,
            }

        # Near-antipodal n and v do not define a stable unique minor plane arc.
        # Preserve the exact validated BASE update rather than inventing an
        # arbitrary orthogonal direction.
        if plane.get("degenerate", False):
            z_new, base_info = self.step_rag_ttba(
                x_s,
                eta,
                x_b,
            )

            return z_new.detach(), {
                "mode": self.METHOD_SCBT,
                "selection_reason": plane["reason"],
                "selected": "BASE_FALLBACK",
                "selector_queries": int(self.all_queries - q_start),
                "bridge_queries": int(
                    self.q_breakdown.get("bridge", 0) - bridge_start
                ),
                "track_queries": 0,
                "theta_n_deg": float("nan"),
                "theta_selected_deg": float("nan"),
                "normal_endpoint_status": 0,
                "fallback_base": 1,
                "base_L": float(base_info.get("L", 0.0)),
                "base_k_app": float(base_info.get("k_app", 0.0)),
                "dot_vn": float(plane.get("dot_vn", 0.0)),
            }

        radius = float(plane["radius"])
        theta_n = float(plane["theta_n"])

        # If n ~= v, there is no meaningful angular bridge to search.  The
        # current known adversarial point is the verified outer endpoint.
        if theta_n <= 1e-10:
            selected_theta = 0.0
            selected_adv = x_b.detach().clone()
            normal_status = 1
            reason = "N_ALIGNED_WITH_V"
        else:
            if self.remaining_queries() <= 0:
                return x_b.detach(), {
                    "mode": self.METHOD_SCBT,
                    "selection_reason": "NO_BUDGET",
                    "selected": "INCUMBENT",
                    "selector_queries": 0,
                    "bridge_queries": 0,
                    "track_queries": 0,
                    "theta_n_deg": math.degrees(theta_n),
                    "theta_selected_deg": 0.0,
                    "normal_endpoint_status": 0,
                    "fallback_base": 0,
                    "dot_vn": float(plane["dot_vn"]),
                }

            # Query the pure-normal point explicitly.  No assumption is made
            # about whether n is adversarial or clean.
            normal_point = x_s + radius * plane["n"]
            normal_status = self.is_adversarial(
                normal_point,
                category="bridge",
            )

            if normal_status == 1:
                selected_theta = theta_n
                selected_adv = normal_point.detach()
                reason = "PURE_NORMAL_ADVERSARIAL"

            elif normal_status == -1:
                # Known arc bracket:
                #   theta=0       -> x_b -> ADV (already known, zero query)
                #   theta=theta_n -> normal endpoint -> CLEAN (queried above)
                low = 0.0
                high = theta_n
                selected_theta = 0.0
                selected_adv = x_b.detach().clone()

                # Endpoint query already consumed one selector query.
                used_selector = int(self.all_queries - q_start)
                remaining_selector = max(
                    0,
                    self.scbt_selector_budget - used_selector,
                )

                arc_cap = min(
                    self.scbt_arc_max_queries,
                    remaining_selector,
                )

                for _ in range(int(arc_cap)):
                    if self.remaining_queries() <= 0:
                        break

                    if (high - low) <= self.scbt_arc_tol:
                        break

                    mid = 0.5 * (low + high)
                    d_mid = self._direction_at_angle(
                        plane,
                        mid,
                    )
                    p_mid = x_s + radius * d_mid

                    status = self.is_adversarial(
                        p_mid,
                        category="bridge",
                    )

                    if status == 0:
                        break

                    if status == 1:
                        low = mid
                        selected_theta = mid
                        selected_adv = p_mid.detach()
                    else:
                        high = mid

                reason = "ARC_LAST_VERIFIED_ADVERSARIAL"

            else:
                # No endpoint label because the global budget ended.
                selected_theta = 0.0
                selected_adv = x_b.detach().clone()
                reason = "BUDGET_AT_NORMAL_ENDPOINT"

        # Reallocate the rest of the fixed selector budget to direct radial
        # progress on concentric circles instead of spending all selector
        # queries on the angular bridge.
        used_selector = int(self.all_queries - q_start)
        radial_budget = max(
            0,
            self.scbt_selector_budget - used_selector,
        )
        radial_budget = min(
            radial_budget,
            self.remaining_queries(),
        )

        clean_bracket, adv_bracket, radial_used = self._tighten_radial_bracket(
            x_s,
            selected_adv,
            radial_budget,
        )

        # Finish to the same latent tolerance as BASE from the already-tight
        # verified clean/adversarial bracket.
        z_new = self._binary_search_known_adv(
            clean_bracket,
            adv_bracket,
            category="boundary",
            max_steps=100,
        )

        bridge_used = int(
            self.q_breakdown.get("bridge", 0) - bridge_start
        )
        track_used = int(
            self.q_breakdown.get("track", 0) - track_start
        )

        selected_name = (
            "NORMAL"
            if abs(selected_theta - theta_n) <= 1e-10
            and theta_n > 1e-10
            else "ARC"
        )

        return z_new.detach(), {
            "mode": self.METHOD_SCBT,
            "selection_reason": str(reason),
            "selected": selected_name,
            "selector_queries": int(bridge_used + track_used),
            "bridge_queries": int(bridge_used),
            "track_queries": int(track_used),
            "radial_track_queries": int(radial_used),
            "theta_n_deg": float(math.degrees(theta_n)),
            "theta_selected_deg": float(math.degrees(selected_theta)),
            "normal_endpoint_status": int(normal_status),
            "fallback_base": 0,
            "dot_vn": float(plane["dot_vn"]),
            "radius_before": float(radius),
            "radius_selected_outer": float(
                torch.norm(selected_adv - x_s).item()
            ),
            "radius_after_track_adv": float(
                torch.norm(adv_bracket - x_s).item()
            ),
            "total_step_queries": int(self.all_queries - q_start),
        }

    # ==================================================================
    # Main comparison loop
    # ==================================================================
    def attack(self):
        # Preserve the exact validated BASE implementation bit-for-bit through
        # the parent attack path.
        if self.requested_attack_method == self.METHOD_BASE:
            result = super().attack()
            result["requested_attack_method"] = self.METHOD_BASE
            result["scbt_diagnostics"] = []
            return result

        z_b = self._initialize_boundary()

        if z_b is None:
            return {
                "success": False,
                "final_z": self.z_source.detach(),
                "final": None,
                "checkpoints": self.checkpoint_summary(),
                "thresholds": self.threshold_summary(),
                "query_breakdown": dict(self.q_breakdown),
                "diagnostics": self.diagnostics,
                "scbt_diagnostics": self.scbt_diagnostics,
                "requested_attack_method": self.requested_attack_method,
            }

        self._record(z_b, "Initial")

        for i in range(self.iteration):
            if self.remaining_queries() <= self.update_query_reserve:
                break

            before_l2 = float(self.rag_pixel_l2(z_b))

            # Parent attack_method is BASE, hence exact fresh-normal schedule:
            # N_t = N0 * sqrt(t+1), capped only by the global query budget.
            eta, normal_info = self._choose_normal(
                z_b,
                i,
            )

            if eta is None or self.remaining_queries() <= 0:
                break

            z_candidate, step_info = self._step_scbt(
                self.z_source,
                eta,
                z_b,
            )

            candidate_l2 = float(self.rag_pixel_l2(z_candidate))

            if self.scbt_monotone_safety:
                accepted = (
                    candidate_l2
                    < before_l2 - self.scbt_accept_eps
                )
            else:
                accepted = True

            if accepted:
                z_next = z_candidate.detach()
                after_l2 = candidate_l2
            else:
                z_next = z_b.detach()
                after_l2 = before_l2

            progress = self._relative_progress(
                before_l2,
                after_l2,
            )

            z_b = z_next.detach()
            self.prev_progress = float(progress)

            self._record(
                z_b,
                f"Iter{i}",
            )

            diag = {
                "iteration": int(i),
                "query": int(self.all_queries),
                "method": self.requested_attack_method,
                "pixel_l2_before": float(before_l2),
                "pixel_l2_candidate": float(candidate_l2),
                "pixel_l2_after": float(after_l2),
                "progress": float(progress),
                "accepted": int(accepted),
            }

            if normal_info is not None:
                diag.update(
                    {
                        f"normal_{k}": v
                        for k, v in normal_info.items()
                    }
                )

            diag.update(
                {
                    f"scbt_{k}": v
                    for k, v in step_info.items()
                }
            )

            self.diagnostics.append(diag)
            self.scbt_diagnostics.append(
                {
                    "iteration": int(i),
                    "query": int(self.all_queries),
                    "accepted": int(accepted),
                    **step_info,
                }
            )

            if self.verbose == "Yes":
                print(
                    "         SCBT: "
                    f"qN={normal_info['q_used']}/{normal_info['q_full']} "
                    f"qArc={step_info.get('bridge_queries', 0)} "
                    f"qTrack={step_info.get('track_queries', 0)} "
                    f"thetaN={step_info.get('theta_n_deg', float('nan')):.2f} "
                    f"thetaUse={step_info.get('theta_selected_deg', float('nan')):.2f} "
                    f"nStatus={step_info.get('normal_endpoint_status', 0)} "
                    f"selected={step_info.get('selected', 'NA')} "
                    f"reason={step_info.get('selection_reason', 'NA')} "
                    f"progress={progress:.4f} "
                    f"accepted={int(accepted)}"
                )

        best = min(
            self.history,
            key=lambda row: row["l2_pix"],
        )

        final = {
            "query": int(best["query"]),
            "budget_consumed": int(self.all_queries),
            "l2_lat": float(best["l2_lat"]),
            "l2_pix": float(best["l2_pix"]),
            "ssim": float(best["ssim"]),
            "lpips": float(best["lpips"]),
            "asr": 1.0,
        }

        q_sum = int(
            sum(int(v) for v in self.q_breakdown.values())
        )

        return {
            "success": True,
            "final_z": z_b.detach(),
            "final": final,
            "checkpoints": self.checkpoint_summary(),
            "thresholds": self.threshold_summary(),
            "query_breakdown": dict(self.q_breakdown),
            "query_audit": {
                "all_queries": int(self.all_queries),
                "sum_breakdown": int(q_sum),
                "matches": bool(q_sum == int(self.all_queries)),
            },
            "diagnostics": self.diagnostics,
            "scbt_diagnostics": self.scbt_diagnostics,
            "requested_attack_method": self.requested_attack_method,
        }
