
import math

import numpy as np
import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBASubspaceAttack(RAGTtBAPGRAttack):
    """
    Matched-budget test of full-space TtBA versus a low-dimensional
    subspace normal estimator.

    Methods
    -------
    RAG_TTBA_BASE
        Validated baseline. Uses the repository's original full-dimensional
        hard-label normal estimator.

    RAG_TTBA_SUB16
        Uses exactly the same attack loop, N_t schedule, TtBA bridge search,
        boundary refinement, total victim-query budget, sigma and checkpoints
        as BASE. The ONLY intended mechanism change is normal estimation:

            full-space Gaussian u in R^D
                ->
            energy-matched Gaussian u restricted to an r-dimensional
            subspace S_r.

    Subspace
    --------
    At each current boundary point z_b,

        v = Normalize(z_b - z_s)

    and

        S_r = span{v, b_2, ..., b_r}.

    The tangent directions b_2,... are obtained by taking a FIXED master
    random matrix, projecting it orthogonally to the current v, and QR
    orthonormalizing it. This keeps the underlying random directions fixed
    across attack iterations while allowing the basis to adapt to v.

    This is deliberately more continuous than drawing a completely new
    random subspace every iteration.

    Energy matching
    ---------------
    Full-space Gaussian:
        u_full ~ N(0, I_D)
        E||u_full||^2 = D

    Subspace Gaussian:
        a ~ N(0, I_r)
        u = sqrt(D/r) * B_r a

    so:
        E||u||^2 = D

    Therefore the same sigma has approximately the same expected perturbation
    norm in BASE and SUB16.

    Strict threat model
    -------------------
    The subspace estimator uses top-1 hard-label decisions only via
    is_adversarial(). No logits, probabilities, gradients or victim backward
    pass are used.

    Query accounting
    ----------------
    Each subspace probe calls is_adversarial(..., category="normal") exactly
    once. Thus the normal-query schedule is matched to BASE:

        N_t = N0 * sqrt(t)

    subject to the same remaining-budget reserve.

    Diagnostics
    -----------
    Without extra victim queries, each SUB16 normal estimate reports:
      * split-half normal cosine;
      * phi = angle(v, n);
      * alpha = v^T n;
      * ||n - alpha v||;
      * adversarial-label fraction;
      * basis orthogonality error.
    """

    METHOD_BASE = "RAG_TTBA_BASE"
    METHOD_SUB16 = "RAG_TTBA_SUB16"

    SUPPORTED = {
        METHOD_BASE,
        METHOD_SUB16,
    }

    def __init__(
        self,
        *args,
        attack_method="RAG_TTBA_BASE",
        subspace_dim=16,
        subspace_basis_seed=4992,
        subspace_probe_seed=5992,
        **kwargs,
    ):
        if attack_method not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported={sorted(self.SUPPORTED)}"
            )

        # Keep the parent's internal method on the already validated BASE path.
        # Our only behavior switch is inside _choose_normal().
        super().__init__(
            *args,
            attack_method="RAG_TTBA_BASE",
            **kwargs,
        )

        self.requested_attack_method = str(
            attack_method
        )

        self.subspace_dim = int(
            subspace_dim
        )

        if self.subspace_dim < 2:
            raise ValueError(
                "subspace_dim must be >= 2."
            )

        self.subspace_basis_seed = int(
            subspace_basis_seed
        )

        self.subspace_probe_seed = int(
            subspace_probe_seed
        )

        # Fixed master random directions. Created lazily because the latent
        # dimension/device/dtype are known only when the first boundary arrives.
        self._master_random_tangent = None
        self._master_signature = None

        self.subspace_normal_diagnostics = []

    # ==================================================================
    # Small helpers
    # ==================================================================
    @staticmethod
    def _safe_unit(
        x,
    ):
        norm = torch.norm(
            x
        )

        if (
            not torch.isfinite(
                norm
            )
            or norm.item() < 1e-12
        ):
            return None, float(
                norm.item()
            )

        return (
            x / norm
        ).detach(), float(
            norm.item()
        )

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
            return None

        value = (
            torch.sum(
                a * b
            )
            / denom
        )

        value = torch.clamp(
            value,
            -1.0,
            1.0,
        )

        return float(
            value.item()
        )

    @staticmethod
    def _angle_deg_from_cos(
        cosine,
    ):
        if cosine is None:
            return None

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

    # ==================================================================
    # Fixed master random tangent directions
    # ==================================================================
    def _get_master_random_tangent(
        self,
        z_boundary,
    ):
        latent_dim = int(
            z_boundary.numel()
        )

        tangent_count = (
            self.subspace_dim
            - 1
        )

        if self.subspace_dim > latent_dim:
            raise ValueError(
                f"subspace_dim={self.subspace_dim} exceeds "
                f"latent_dim={latent_dim}."
            )

        signature = (
            latent_dim,
            str(
                z_boundary.device
            ),
            str(
                z_boundary.dtype
            ),
            tangent_count,
        )

        if (
            self._master_random_tangent is not None
            and self._master_signature
            == signature
        ):
            return self._master_random_tangent

        # Generate on CPU with a LOCAL generator so basis construction does not
        # disturb the attack's global RNG stream.
        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            self.subspace_basis_seed
        )

        raw_cpu = torch.randn(
            latent_dim,
            tangent_count,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )

        raw = raw_cpu.to(
            device=z_boundary.device,
            dtype=z_boundary.dtype,
        )

        self._master_random_tangent = (
            raw.detach()
        )

        self._master_signature = (
            signature
        )

        return self._master_random_tangent

    # ==================================================================
    # Build S_r = span(v, tangent directions)
    # ==================================================================
    def _build_subspace_basis(
        self,
        z_boundary,
    ):
        z_flat = (
            z_boundary.detach().reshape(
                -1
            )
        )

        source_flat = (
            self.z_source.detach().reshape(
                -1
            )
        )

        v_raw = (
            z_flat
            - source_flat
        )

        v, radius = self._safe_unit(
            v_raw
        )

        if v is None:
            return None

        latent_dim = int(
            v.numel()
        )

        raw = self._get_master_random_tangent(
            z_boundary
        )

        # Project the fixed master directions to v^\perp.
        radial_projection = torch.matmul(
            v.unsqueeze(
                0
            ),
            raw,
        )

        tangent = (
            raw
            - v.unsqueeze(
                1
            )
            * radial_projection
        )

        # Orthonormalize tangents.
        q_tangent, _ = torch.linalg.qr(
            tangent,
            mode="reduced",
        )

        # Remove numerical v leakage and QR once more.
        leakage = torch.matmul(
            v.unsqueeze(
                0
            ),
            q_tangent,
        )

        q_tangent = (
            q_tangent
            - v.unsqueeze(
                1
            )
            * leakage
        )

        q_tangent, _ = torch.linalg.qr(
            q_tangent,
            mode="reduced",
        )

        basis = torch.cat(
            [
                v.unsqueeze(
                    1
                ),
                q_tangent[
                    :,
                    : self.subspace_dim - 1,
                ],
            ],
            dim=1,
        ).detach()

        # Numerical sanity checks.
        gram = torch.matmul(
            basis.T,
            basis,
        )

        identity = torch.eye(
            self.subspace_dim,
            device=gram.device,
            dtype=gram.dtype,
        )

        orth_error = float(
            torch.max(
                torch.abs(
                    gram
                    - identity
                )
            ).item()
        )

        tangent_v_error = 0.0

        if self.subspace_dim > 1:
            tangent_v_error = float(
                torch.max(
                    torch.abs(
                        torch.matmul(
                            v.unsqueeze(
                                0
                            ),
                            basis[
                                :,
                                1:,
                            ],
                        )
                    )
                ).item()
            )

        energy_scale = math.sqrt(
            float(
                latent_dim
            )
            / float(
                self.subspace_dim
            )
        )

        return {
            "basis": basis,
            "v": v,
            "radius": float(
                radius
            ),
            "latent_dim": int(
                latent_dim
            ),
            "energy_scale": float(
                energy_scale
            ),
            "orth_error": float(
                orth_error
            ),
            "tangent_v_error": float(
                tangent_v_error
            ),
        }

    # ==================================================================
    # Query-matched one-sided subspace normal
    # ==================================================================
    def _estimate_subspace_normal(
        self,
        z_boundary,
        q_target,
        iteration_index,
    ):
        basis_info = self._build_subspace_basis(
            z_boundary
        )

        if basis_info is None:
            return None, 0, {
                "valid": False,
                "reason": "ZERO_V_VECTOR",
            }

        basis = basis_info[
            "basis"
        ]

        v = basis_info[
            "v"
        ]

        energy_scale = float(
            basis_info[
                "energy_scale"
            ]
        )

        r = int(
            self.subspace_dim
        )

        # Local CPU generator gives deterministic, method-specific probe
        # coefficients without altering the global RNG stream.
        generator = torch.Generator(
            device="cpu"
        )

        local_seed = int(
            self.subspace_probe_seed
            + 1_000_003
            * int(
                iteration_index
            )
        )

        generator.manual_seed(
            local_seed
        )

        eta_coeff = torch.zeros(
            r,
            device=basis.device,
            dtype=basis.dtype,
        )

        # Split-half estimators use NO extra victim queries.
        eta_first = torch.zeros_like(
            eta_coeff
        )

        eta_second = torch.zeros_like(
            eta_coeff
        )

        q_before = int(
            self.all_queries
        )

        used = 0
        adv_count = 0
        clean_count = 0

        q_target = int(
            q_target
        )

        split_point = max(
            1,
            q_target // 2,
        )

        for j in range(
            q_target
        ):
            if self.remaining_queries() <= 0:
                break

            a_cpu = torch.randn(
                r,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )

            a = a_cpu.to(
                device=basis.device,
                dtype=basis.dtype,
            )

            u_flat = (
                energy_scale
                * torch.matmul(
                    basis,
                    a,
                )
            )

            u = u_flat.reshape_as(
                z_boundary
            )

            status = self.is_adversarial(
                z_boundary
                + float(
                    self.sigma
                )
                * u,
                category="normal",
            )

            if status not in (
                -1,
                1,
            ):
                break

            used += 1

            if status == 1:
                adv_count += 1
            else:
                clean_count += 1

            contribution = (
                float(
                    status
                )
                * a
            )

            eta_coeff = (
                eta_coeff
                + contribution
            )

            if j < split_point:
                eta_first = (
                    eta_first
                    + contribution
                )
            else:
                eta_second = (
                    eta_second
                    + contribution
                )

        q_after = int(
            self.all_queries
        )

        q_used = max(
            0,
            q_after - q_before,
        )

        # Defensive consistency check. q_used is authoritative.
        if q_used != used:
            used = q_used

        eta_flat = torch.matmul(
            basis,
            eta_coeff,
        )

        eta_unit, eta_norm = self._safe_unit(
            eta_flat
        )

        if eta_unit is None:
            return None, int(
                q_used
            ), {
                "valid": False,
                "reason": "ZERO_ESTIMATOR",
                "q_used": int(
                    q_used
                ),
                "iteration": int(
                    iteration_index
                ),
            }

        # Return in latent tensor shape expected by the parent TtBA step.
        eta = eta_unit.reshape_as(
            z_boundary
        ).detach()

        # --------------------------------------------------------------
        # Free diagnostics from the same q_used probes.
        # --------------------------------------------------------------
        first_full = torch.matmul(
            basis,
            eta_first,
        )

        second_full = torch.matmul(
            basis,
            eta_second,
        )

        first_unit, _ = self._safe_unit(
            first_full
        )

        second_unit, _ = self._safe_unit(
            second_full
        )

        split_cos = None

        if (
            first_unit is not None
            and second_unit is not None
        ):
            split_cos = self._safe_cos(
                first_unit,
                second_unit,
            )

        alpha = float(
            torch.dot(
                eta_unit,
                v,
            ).item()
        )

        alpha = max(
            -1.0,
            min(
                1.0,
                alpha,
            ),
        )

        phi_deg = self._angle_deg_from_cos(
            alpha
        )

        tangent_part = (
            eta_unit
            - alpha * v
        )

        n_perp_norm = float(
            torch.norm(
                tangent_part
            ).item()
        )

        adv_fraction = (
            float(
                adv_count
            )
            / float(
                max(
                    1,
                    q_used,
                )
            )
        )

        diag = {
            "valid": True,
            "iteration": int(
                iteration_index
            ),
            "q_target": int(
                q_target
            ),
            "q_used": int(
                q_used
            ),
            "subspace_dim": int(
                r
            ),
            "latent_dim": int(
                basis_info[
                    "latent_dim"
                ]
            ),
            "sigma": float(
                self.sigma
            ),
            "energy_scale": float(
                energy_scale
            ),
            "radius_latent": float(
                basis_info[
                    "radius"
                ]
            ),
            "basis_orth_error": float(
                basis_info[
                    "orth_error"
                ]
            ),
            "basis_tangent_v_error": float(
                basis_info[
                    "tangent_v_error"
                ]
            ),
            "probe_seed": int(
                local_seed
            ),
            "eta_norm_raw": float(
                eta_norm
            ),
            "split_half_cos": (
                None
                if split_cos is None
                else float(
                    split_cos
                )
            ),
            "alpha_v_dot_n": float(
                alpha
            ),
            "phi_deg": (
                None
                if phi_deg is None
                else float(
                    phi_deg
                )
            ),
            "n_perp_norm": float(
                n_perp_norm
            ),
            "adv_count": int(
                adv_count
            ),
            "clean_count": int(
                clean_count
            ),
            "adv_fraction": float(
                adv_fraction
            ),
        }

        return eta, int(
            q_used
        ), diag

    # ==================================================================
    # Exact BASE schedule; only estimator differs for SUB16
    # ==================================================================
    def _choose_normal(
        self,
        z_boundary,
        iteration_index,
    ):
        q_full = int(
            self.N0
            * math.sqrt(
                iteration_index
                + 1.0
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

        # --------------------------------------------------------------
        # BASE: exact original full-space estimator.
        # --------------------------------------------------------------
        if (
            self.requested_attack_method
            == self.METHOD_BASE
        ):
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
                "normal_method": "FULL_PARENT",
            }

        # --------------------------------------------------------------
        # SUB16: identical q_full schedule, low-dimensional estimator.
        # --------------------------------------------------------------
        eta, q_used, diag = (
            self._estimate_subspace_normal(
                z_boundary=z_boundary,
                q_target=q_full,
                iteration_index=iteration_index,
            )
        )

        self.subspace_normal_diagnostics.append(
            diag
        )

        if self.verbose == "Yes":
            if diag.get(
                "valid",
                False,
            ):
                split_text = (
                    "NA"
                    if diag[
                        "split_half_cos"
                    ] is None
                    else f"{diag['split_half_cos']:.4f}"
                )

                print(
                    "         SUB16-N: "
                    f"qN={diag['q_used']}/{diag['q_target']} "
                    f"splitCos={split_text} "
                    f"phi={diag['phi_deg']:.2f}deg "
                    f"alpha={diag['alpha_v_dot_n']:.4f} "
                    f"perp={diag['n_perp_norm']:.4f} "
                    f"advFrac={diag['adv_fraction']:.3f}"
                )

            else:
                print(
                    "         SUB16-N: invalid "
                    f"reason={diag.get('reason')}"
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
            "normal_method": "SUBSPACE_ONE_SIDED",
            "subspace_dim": int(
                self.subspace_dim
            ),
            "split_half_cos": (
                None
                if not diag.get(
                    "valid",
                    False,
                )
                else diag.get(
                    "split_half_cos"
                )
            ),
        }

    # ==================================================================
    # Add method-specific diagnostics to parent attack result.
    # ==================================================================
    def attack(
        self,
    ):
        result = super().attack()

        valid = [
            row
            for row in self.subspace_normal_diagnostics
            if row.get(
                "valid",
                False,
            )
        ]

        split_values = [
            row[
                "split_half_cos"
            ]
            for row in valid
            if row.get(
                "split_half_cos"
            ) is not None
            and np.isfinite(
                row[
                    "split_half_cos"
                ]
            )
        ]

        perp_values = [
            row[
                "n_perp_norm"
            ]
            for row in valid
            if row.get(
                "n_perp_norm"
            ) is not None
        ]

        result[
            "requested_attack_method"
        ] = self.requested_attack_method

        result[
            "subspace_normal_diagnostics"
        ] = self.subspace_normal_diagnostics

        result[
            "subspace_normal_summary"
        ] = {
            "method": self.requested_attack_method,
            "subspace_dim": (
                int(
                    self.subspace_dim
                )
                if self.requested_attack_method
                == self.METHOD_SUB16
                else None
            ),
            "normal_updates": int(
                len(
                    valid
                )
            ),
            "mean_split_half_cos": (
                None
                if not split_values
                else float(
                    np.mean(
                        split_values
                    )
                )
            ),
            "median_split_half_cos": (
                None
                if not split_values
                else float(
                    np.median(
                        split_values
                    )
                )
            ),
            "mean_n_perp_norm": (
                None
                if not perp_values
                else float(
                    np.mean(
                        perp_values
                    )
                )
            ),
        }

        return result
