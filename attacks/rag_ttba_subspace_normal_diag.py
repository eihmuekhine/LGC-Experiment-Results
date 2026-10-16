
import copy
import math
import random
from itertools import combinations

import numpy as np
import torch

from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


class RAGTtBASubspaceNormalDiagnosticAttack(RAGTtBAPGRAttack):
    """
    OFFLINE diagnostic for low-dimensional hard-label normal estimation.

    The validated RAG_TTBA_BASE attack trajectory is kept unchanged.

    At selected boundary states z_b, the diagnostic compares:

      1) FULL_PARENT
         The repository's existing full-dimensional estimate_normal().

      2) SUBSPACE_ONE_SIDED
         One-sided hard-label estimation inside a fixed low-dimensional
         subspace.

      3) SUBSPACE_PAIRED
         Antithetic +/- hard-label estimation inside the SAME fixed subspace.

    The subspace is built as

        S_r = span{v, b_2, ..., b_r},

    where

        v = (z_b - z_s) / ||z_b - z_s||

    and b_2,... are deterministic random orthonormal tangent directions
    satisfying b_j^T v = 0.

    The dimensions are NESTED:
        S_16 subset S_32 subset S_64 subset S_128

    because one max-r basis is constructed and smaller experiments use prefixes.

    Energy matching
    ---------------
    A full-dimensional Gaussian u_full in R^D has

        E ||u_full||^2 = D.

    If a ~ N(0, I_r) and u = B_r a, then

        E ||u||^2 = r.

    To compare the same sigma fairly, subspace perturbations therefore use

        u = sqrt(D/r) * B_r a,

    giving

        E ||u||^2 = D.

    Thus sigma has approximately the same expected perturbation norm as the
    parent full-dimensional estimator.

    XY relevance
    ------------
    Besides normal repeat consistency, this diagnostic directly measures
    repeat consistency of the actual XY y-axis

        e_y = Normalize(n - (n^T v) v).

    This is more important for retrying XY than normal cosine alone.

    Query accounting
    ----------------
    All diagnostic queries are OFFLINE. They do NOT consume the attack's
    2000-query budget. Global RNG state is restored after every diagnostic
    operation so the BASE trajectory should reproduce unchanged.
    """

    FULL_PARENT = "FULL_PARENT"
    SUBSPACE_ONE = "SUBSPACE_ONE_SIDED"
    SUBSPACE_PAIRED = "SUBSPACE_PAIRED"

    def __init__(
        self,
        *args,
        subspace_dims=(16, 32, 64, 128),
        diagnostic_query_budgets=(60, 120),
        diagnostic_sigma=2e-4,
        normal_repeats=2,
        snapshot_iterations=(0,),
        diagnostic_seed=3992,
        **kwargs,
    ):
        kwargs.pop("attack_method", None)

        super().__init__(
            *args,
            attack_method="RAG_TTBA_BASE",
            **kwargs,
        )

        self.subspace_dims = tuple(
            sorted(
                {
                    int(x)
                    for x in subspace_dims
                }
            )
        )

        if not self.subspace_dims:
            raise ValueError(
                "At least one subspace dimension is required."
            )

        if any(x < 2 for x in self.subspace_dims):
            raise ValueError(
                "All subspace_dims must be >= 2 "
                "(one radial v axis + at least one tangent axis)."
            )

        self.diagnostic_query_budgets = tuple(
            sorted(
                {
                    int(x)
                    for x in diagnostic_query_budgets
                }
            )
        )

        if any(x < 2 for x in self.diagnostic_query_budgets):
            raise ValueError(
                "All diagnostic_query_budgets must be >= 2."
            )

        self.diagnostic_sigma = float(
            diagnostic_sigma
        )

        if self.diagnostic_sigma <= 0.0:
            raise ValueError(
                "diagnostic_sigma must be > 0."
            )

        self.normal_repeats = int(
            normal_repeats
        )

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

        self.diagnostic_seed = int(
            diagnostic_seed
        )

        self.subspace_diagnostics = []
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
            state["torch_cuda"] = (
                torch.cuda.get_rng_state_all()
            )

        return state

    @staticmethod
    def _restore_rng_state(state):
        random.setstate(
            state["python"]
        )

        np.random.set_state(
            state["numpy"]
        )

        torch.random.set_rng_state(
            state["torch_cpu"]
        )

        if (
            torch.cuda.is_available()
            and state["torch_cuda"] is not None
        ):
            torch.cuda.set_rng_state_all(
                state["torch_cuda"]
            )

    @staticmethod
    def _seed_all(seed):
        random.seed(
            int(seed)
        )

        np.random.seed(
            int(seed)
        )

        torch.manual_seed(
            int(seed)
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                int(seed)
            )

    # ==================================================================
    # Probe isolation
    # ==================================================================
    def _make_probe_copy(
        self,
        seed,
        query_budget,
        sigma,
    ):
        probe = copy.copy(
            self
        )

        if hasattr(
            self,
            "q_breakdown",
        ):
            try:
                probe.q_breakdown = type(
                    self.q_breakdown
                )(
                    (
                        key,
                        0,
                    )
                    for key
                    in self.q_breakdown.keys()
                )
            except Exception:
                probe.q_breakdown = {
                    key: 0
                    for key
                    in self.q_breakdown.keys()
                }

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
            "subspace_diagnostics",
        ):
            probe.subspace_diagnostics = []

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
                    (
                        int,
                        float,
                    ),
                )
            ):
                probe.__dict__[name] = 0

        if "max_queries" in probe.__dict__:
            probe.max_queries = int(
                query_budget + 64
            )

        if "seed" in probe.__dict__:
            probe.seed = int(
                seed
            )

        probe.sigma = float(
            sigma
        )

        return probe

    # ==================================================================
    # Geometry helpers
    # ==================================================================
    @staticmethod
    def _safe_unit(x):
        norm = torch.norm(
            x
        )

        value = float(
            norm.item()
        )

        if (
            not torch.isfinite(
                norm
            )
            or value < 1e-12
        ):
            return None, value

        return (
            x / norm
        ).detach(), value

    @staticmethod
    def _safe_cos(a, b):
        denom = (
            torch.norm(a)
            * torch.norm(b)
        )

        if denom.item() < 1e-12:
            return float(
                "nan"
            )

        value = (
            torch.sum(
                a * b
            )
            / denom
        )

        return float(
            torch.clamp(
                value,
                -1.0,
                1.0,
            ).item()
        )

    @staticmethod
    def _angle_deg_from_cos(c):
        if not math.isfinite(
            c
        ):
            return float(
                "nan"
            )

        c = max(
            -1.0,
            min(
                1.0,
                float(c),
            ),
        )

        return float(
            math.degrees(
                math.acos(
                    c
                )
            )
        )

    @staticmethod
    def _mean_unit(vectors):
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
            or norm.item() < 1e-12
        ):
            return None

        return (
            mean / norm
        ).detach()

    def _normal_to_xy_y(
        self,
        n_flat,
        v_flat,
    ):
        alpha = float(
            torch.dot(
                n_flat,
                v_flat,
            ).item()
        )

        tangent = (
            n_flat
            - alpha * v_flat
        )

        e_y, tangent_norm = (
            self._safe_unit(
                tangent
            )
        )

        phi = self._angle_deg_from_cos(
            alpha
        )

        return {
            "alpha_v_dot_n": float(
                alpha
            ),
            "phi_deg": float(
                phi
            ),
            "n_perp_norm": float(
                tangent_norm
            ),
            "e_y": e_y,
            "v_dot_ey": (
                None
                if e_y is None
                else float(
                    torch.dot(
                        v_flat,
                        e_y
                    ).item()
                )
            ),
        }

    # ==================================================================
    # Fixed nested basis
    # ==================================================================
    def _build_nested_basis(
        self,
        z_boundary,
        iteration_index,
    ):
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

        v_flat, radius = self._safe_unit(
            v_raw
        )

        if v_flat is None:
            return None

        latent_dim = int(
            v_flat.numel()
        )

        max_r = int(
            max(
                self.subspace_dims
            )
        )

        if max_r > latent_dim:
            raise ValueError(
                f"Requested max subspace dim {max_r} "
                f"but latent dimension is only {latent_dim}."
            )

        tangent_count = max_r - 1

        rng_state = self._capture_rng_state()

        try:
            basis_seed = int(
                self.diagnostic_seed
                + 100_000_000
                + 1_000_000
                * int(
                    iteration_index
                )
            )

            self._seed_all(
                basis_seed
            )

            # D x (max_r-1)
            random_cols = torch.randn(
                latent_dim,
                tangent_count,
                device=v_flat.device,
                dtype=v_flat.dtype,
            )

            # Remove radial component.
            projection = torch.matmul(
                v_flat.unsqueeze(
                    0
                ),
                random_cols,
            )

            random_cols = (
                random_cols
                - v_flat.unsqueeze(
                    1
                )
                * projection
            )

            # First tangent orthonormalization.
            q_tangent, _ = torch.linalg.qr(
                random_cols,
                mode="reduced",
            )

            # Re-project against v to remove numerical leakage, then QR again.
            leakage = torch.matmul(
                v_flat.unsqueeze(
                    0
                ),
                q_tangent,
            )

            q_tangent = (
                q_tangent
                - v_flat.unsqueeze(
                    1
                )
                * leakage
            )

            q_tangent, _ = torch.linalg.qr(
                q_tangent,
                mode="reduced",
            )

        finally:
            self._restore_rng_state(
                rng_state
            )

        max_basis = torch.cat(
            [
                v_flat.unsqueeze(
                    1
                ),
                q_tangent[
                    :,
                    :tangent_count,
                ],
            ],
            dim=1,
        ).detach()

        gram = torch.matmul(
            max_basis.T,
            max_basis,
        )

        identity = torch.eye(
            max_r,
            device=gram.device,
            dtype=gram.dtype,
        )

        orth_error = float(
            torch.max(
                torch.abs(
                    gram - identity
                )
            ).item()
        )

        tangent_v_error = float(
            torch.max(
                torch.abs(
                    torch.matmul(
                        v_flat.unsqueeze(
                            0
                        ),
                        max_basis[
                            :,
                            1:,
                        ],
                    )
                )
            ).item()
        )

        return {
            "latent_dim": int(
                latent_dim
            ),
            "radius_latent": float(
                radius
            ),
            "v_flat": v_flat.detach(),
            "max_basis": max_basis,
            "basis_seed": int(
                basis_seed
            ),
            "basis_orthogonality_max_abs_error":
                float(
                    orth_error
                ),
            "basis_tangent_v_max_abs_dot":
                float(
                    tangent_v_error
                ),
        }

    # ==================================================================
    # Full parent estimator control
    # ==================================================================
    def _estimate_full_parent(
        self,
        z_boundary,
        query_budget,
        seed,
    ):
        rng_state = self._capture_rng_state()

        try:
            self._seed_all(
                seed
            )

            probe = self._make_probe_copy(
                seed=seed,
                query_budget=query_budget,
                sigma=self.diagnostic_sigma,
            )

            q_before = int(
                getattr(
                    probe,
                    "all_queries",
                    0,
                )
            )

            eta, q_returned = (
                probe.estimate_normal(
                    z_boundary.detach(),
                    int(
                        query_budget
                    ),
                )
            )

            q_after = int(
                getattr(
                    probe,
                    "all_queries",
                    q_before,
                )
            )

            q_counter = max(
                0,
                q_after - q_before,
            )

            q_used = (
                q_counter
                if q_counter > 0
                else int(
                    q_returned
                )
            )

        finally:
            self._restore_rng_state(
                rng_state
            )

        if eta is None:
            return {
                "valid": False,
                "normal_flat": None,
                "q_used": int(
                    q_used
                    if "q_used" in locals()
                    else 0
                ),
                "eta_norm": None,
                "pair_count": None,
                "informative_pairs": None,
                "informative_pair_rate": None,
            }

        n, eta_norm = self._safe_unit(
            eta.reshape(
                -1
            )
        )

        return {
            "valid": n is not None,
            "normal_flat": n,
            "q_used": int(
                q_used
            ),
            "eta_norm": float(
                eta_norm
            ),
            "pair_count": None,
            "informative_pairs": None,
            "informative_pair_rate": None,
        }

    # ==================================================================
    # Subspace estimators
    # ==================================================================
    def _estimate_subspace(
        self,
        estimator,
        z_boundary,
        basis,
        latent_dim,
        query_budget,
        seed,
    ):
        r = int(
            basis.shape[
                1
            ]
        )

        energy_scale = math.sqrt(
            float(
                latent_dim
            )
            / float(
                r
            )
        )

        rng_state = self._capture_rng_state()

        try:
            self._seed_all(
                seed
            )

            probe = self._make_probe_copy(
                seed=seed,
                query_budget=query_budget,
                sigma=self.diagnostic_sigma,
            )

            eta_coeff = torch.zeros(
                r,
                device=basis.device,
                dtype=basis.dtype,
            )

            q_before = int(
                getattr(
                    probe,
                    "all_queries",
                    0,
                )
            )

            informative_pairs = 0
            pair_count = 0

            if estimator == self.SUBSPACE_ONE:
                for _ in range(
                    int(
                        query_budget
                    )
                ):
                    if probe.remaining_queries() < 1:
                        break

                    a = torch.randn(
                        r,
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

                    status = probe.is_adversarial(
                        z_boundary
                        + self.diagnostic_sigma
                        * u,
                        category="normal",
                    )

                    if status not in (
                        -1,
                        1,
                    ):
                        break

                    # Scaling by energy_scale is common to all samples and
                    # therefore irrelevant after normalization.
                    eta_coeff = (
                        eta_coeff
                        + float(
                            status
                        )
                        * a
                    )

            elif estimator == self.SUBSPACE_PAIRED:
                requested_pairs = int(
                    query_budget
                ) // 2

                for _ in range(
                    requested_pairs
                ):
                    if probe.remaining_queries() < 2:
                        break

                    a = torch.randn(
                        r,
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

                    s_plus = probe.is_adversarial(
                        z_boundary
                        + self.diagnostic_sigma
                        * u,
                        category="normal",
                    )

                    if s_plus not in (
                        -1,
                        1,
                    ):
                        break

                    s_minus = probe.is_adversarial(
                        z_boundary
                        - self.diagnostic_sigma
                        * u,
                        category="normal",
                    )

                    if s_minus not in (
                        -1,
                        1,
                    ):
                        break

                    pair_count += 1

                    weight = float(
                        s_plus
                        - s_minus
                    )

                    if weight != 0.0:
                        informative_pairs += 1
                        eta_coeff = (
                            eta_coeff
                            + weight * a
                        )

            else:
                raise ValueError(
                    f"Unknown subspace estimator: {estimator}"
                )

            q_after = int(
                getattr(
                    probe,
                    "all_queries",
                    q_before,
                )
            )

            q_used = max(
                0,
                q_after - q_before,
            )

        finally:
            self._restore_rng_state(
                rng_state
            )

        eta_full = torch.matmul(
            basis,
            eta_coeff,
        )

        n_flat, eta_norm = self._safe_unit(
            eta_full
        )

        info_rate = (
            float(
                informative_pairs
            )
            / float(
                pair_count
            )
            if pair_count > 0
            else None
        )

        return {
            "valid": n_flat is not None,
            "normal_flat": n_flat,
            "q_used": int(
                q_used
            ),
            "eta_norm": float(
                eta_norm
            ),
            "pair_count": (
                int(
                    pair_count
                )
                if estimator
                == self.SUBSPACE_PAIRED
                else None
            ),
            "informative_pairs": (
                int(
                    informative_pairs
                )
                if estimator
                == self.SUBSPACE_PAIRED
                else None
            ),
            "informative_pair_rate": (
                None
                if info_rate is None
                else float(
                    info_rate
                )
            ),
            "energy_scale": float(
                energy_scale
            ),
        }

    # ==================================================================
    # Repeat consistency
    # ==================================================================
    def _pairwise_consistency(
        self,
        vectors,
    ):
        signed = []

        for ia, ib in combinations(
            range(
                len(
                    vectors
                )
            ),
            2,
        ):
            c = self._safe_cos(
                vectors[
                    ia
                ],
                vectors[
                    ib
                ],
            )

            signed.append(
                c
            )

        if not signed:
            return {
                "signed_mean": None,
                "signed_min": None,
                "abs_mean": None,
                "angle_mean_deg": None,
            }

        return {
            "signed_mean": float(
                np.mean(
                    signed
                )
            ),
            "signed_min": float(
                np.min(
                    signed
                )
            ),
            "abs_mean": float(
                np.mean(
                    np.abs(
                        signed
                    )
                )
            ),
            "angle_mean_deg": float(
                np.mean(
                    [
                        self._angle_deg_from_cos(
                            c
                        )
                        for c in signed
                    ]
                )
            ),
        }

    # ==================================================================
    # Snapshot
    # ==================================================================
    def _run_subspace_snapshot(
        self,
        z_boundary,
        iteration_index,
    ):
        basis_info = self._build_nested_basis(
            z_boundary=z_boundary,
            iteration_index=iteration_index,
        )

        if basis_info is None:
            snapshot = {
                "iteration": int(
                    iteration_index
                ),
                "query_at_snapshot": int(
                    self.all_queries
                ),
                "valid": False,
                "reason": "ZERO_V_VECTOR",
                "rows": [],
                "repeat_consistency": [],
                "cross_dim": [],
                "full_vs_subspace": [],
                "diagnostic_queries": 0,
            }

            self.subspace_diagnostics.append(
                snapshot
            )

            return snapshot

        latent_dim = int(
            basis_info[
                "latent_dim"
            ]
        )

        v_flat = basis_info[
            "v_flat"
        ]

        max_basis = basis_info[
            "max_basis"
        ]

        snapshot = {
            "iteration": int(
                iteration_index
            ),
            "query_at_snapshot": int(
                self.all_queries
            ),
            "valid": True,
            "reason": "OK",
            "latent_dim": int(
                latent_dim
            ),
            "radius_latent": float(
                basis_info[
                    "radius_latent"
                ]
            ),
            "basis_seed": int(
                basis_info[
                    "basis_seed"
                ]
            ),
            "basis_orthogonality_max_abs_error":
                float(
                    basis_info[
                        "basis_orthogonality_max_abs_error"
                    ]
                ),
            "basis_tangent_v_max_abs_dot":
                float(
                    basis_info[
                        "basis_tangent_v_max_abs_dot"
                    ]
                ),
            "rows": [],
            "repeat_consistency": [],
            "cross_dim": [],
            "full_vs_subspace": [],
            "diagnostic_queries": 0,
        }

        # key = (estimator, r_or_0, budget)
        normals = {}
        y_axes = {}
        mean_normals = {}
        mean_y_axes = {}

        # --------------------------------------------------------------
        # FULL_PARENT control.
        # --------------------------------------------------------------
        for budget_index, budget in enumerate(
            self.diagnostic_query_budgets
        ):
            key = (
                self.FULL_PARENT,
                0,
                int(
                    budget
                ),
            )

            normals[
                key
            ] = []
            y_axes[
                key
            ] = []

            for repeat_index in range(
                self.normal_repeats
            ):
                seed = int(
                    self.diagnostic_seed
                    + 10_000_000
                    * int(
                        iteration_index
                    )
                    + 100_000
                    * int(
                        budget_index
                    )
                    + repeat_index
                )

                result = self._estimate_full_parent(
                    z_boundary=z_boundary,
                    query_budget=budget,
                    seed=seed,
                )

                q_used = int(
                    result[
                        "q_used"
                    ]
                )

                self.diagnostic_query_total += (
                    q_used
                )

                snapshot[
                    "diagnostic_queries"
                ] += q_used

                row = {
                    "estimator": self.FULL_PARENT,
                    "subspace_dim": None,
                    "latent_dim": int(
                        latent_dim
                    ),
                    "sigma": float(
                        self.diagnostic_sigma
                    ),
                    "query_budget": int(
                        budget
                    ),
                    "repeat": int(
                        repeat_index
                    ),
                    "seed": int(
                        seed
                    ),
                    "q_used": int(
                        q_used
                    ),
                    "valid": bool(
                        result[
                            "valid"
                        ]
                    ),
                    "eta_norm": result[
                        "eta_norm"
                    ],
                    "energy_scale": 1.0,
                    "alpha_v_dot_n": None,
                    "phi_deg": None,
                    "n_perp_norm": None,
                    "v_dot_ey": None,
                    "pair_count": None,
                    "informative_pairs": None,
                    "informative_pair_rate": None,
                }

                if (
                    result[
                        "valid"
                    ]
                    and result[
                        "normal_flat"
                    ] is not None
                ):
                    n = result[
                        "normal_flat"
                    ]

                    geom = self._normal_to_xy_y(
                        n_flat=n,
                        v_flat=v_flat,
                    )

                    for field in (
                        "alpha_v_dot_n",
                        "phi_deg",
                        "n_perp_norm",
                        "v_dot_ey",
                    ):
                        row[
                            field
                        ] = geom[
                            field
                        ]

                    normals[
                        key
                    ].append(
                        n
                    )

                    if geom[
                        "e_y"
                    ] is not None:
                        y_axes[
                            key
                        ].append(
                            geom[
                                "e_y"
                            ]
                        )

                snapshot[
                    "rows"
                ].append(
                    row
                )

        # --------------------------------------------------------------
        # Subspace estimators.
        # --------------------------------------------------------------
        subspace_estimators = (
            self.SUBSPACE_ONE,
            self.SUBSPACE_PAIRED,
        )

        for estimator_index, estimator in enumerate(
            subspace_estimators
        ):
            for dim_index, r in enumerate(
                self.subspace_dims
            ):
                basis = max_basis[
                    :,
                    :int(
                        r
                    ),
                ]

                for budget_index, budget in enumerate(
                    self.diagnostic_query_budgets
                ):
                    key = (
                        estimator,
                        int(
                            r
                        ),
                        int(
                            budget
                        ),
                    )

                    normals[
                        key
                    ] = []
                    y_axes[
                        key
                    ] = []

                    for repeat_index in range(
                        self.normal_repeats
                    ):
                        seed = int(
                            self.diagnostic_seed
                            + 20_000_000
                            * int(
                                iteration_index
                            )
                            + 5_000_000
                            * int(
                                estimator_index
                            )
                            + 100_000
                            * int(
                                dim_index
                            )
                            + 1_000
                            * int(
                                budget_index
                            )
                            + repeat_index
                        )

                        result = self._estimate_subspace(
                            estimator=estimator,
                            z_boundary=z_boundary,
                            basis=basis,
                            latent_dim=latent_dim,
                            query_budget=budget,
                            seed=seed,
                        )

                        q_used = int(
                            result[
                                "q_used"
                            ]
                        )

                        self.diagnostic_query_total += (
                            q_used
                        )

                        snapshot[
                            "diagnostic_queries"
                        ] += q_used

                        row = {
                            "estimator": estimator,
                            "subspace_dim": int(
                                r
                            ),
                            "latent_dim": int(
                                latent_dim
                            ),
                            "sigma": float(
                                self.diagnostic_sigma
                            ),
                            "query_budget": int(
                                budget
                            ),
                            "repeat": int(
                                repeat_index
                            ),
                            "seed": int(
                                seed
                            ),
                            "q_used": int(
                                q_used
                            ),
                            "valid": bool(
                                result[
                                    "valid"
                                ]
                            ),
                            "eta_norm": result[
                                "eta_norm"
                            ],
                            "energy_scale": float(
                                result[
                                    "energy_scale"
                                ]
                            ),
                            "alpha_v_dot_n": None,
                            "phi_deg": None,
                            "n_perp_norm": None,
                            "v_dot_ey": None,
                            "pair_count": result[
                                "pair_count"
                            ],
                            "informative_pairs": result[
                                "informative_pairs"
                            ],
                            "informative_pair_rate":
                                result[
                                    "informative_pair_rate"
                                ],
                        }

                        if (
                            result[
                                "valid"
                            ]
                            and result[
                                "normal_flat"
                            ] is not None
                        ):
                            n = result[
                                "normal_flat"
                            ]

                            geom = self._normal_to_xy_y(
                                n_flat=n,
                                v_flat=v_flat,
                            )

                            for field in (
                                "alpha_v_dot_n",
                                "phi_deg",
                                "n_perp_norm",
                                "v_dot_ey",
                            ):
                                row[
                                    field
                                ] = geom[
                                    field
                                ]

                            normals[
                                key
                            ].append(
                                n
                            )

                            if geom[
                                "e_y"
                            ] is not None:
                                y_axes[
                                    key
                                ].append(
                                    geom[
                                        "e_y"
                                    ]
                                )

                        snapshot[
                            "rows"
                        ].append(
                            row
                        )

        # --------------------------------------------------------------
        # Repeat consistency for every exact configuration.
        # --------------------------------------------------------------
        for key in normals.keys():
            estimator, r, budget = key

            normal_consistency = (
                self._pairwise_consistency(
                    normals[
                        key
                    ]
                )
            )

            y_consistency = (
                self._pairwise_consistency(
                    y_axes[
                        key
                    ]
                )
            )

            snapshot[
                "repeat_consistency"
            ].append(
                {
                    "estimator": estimator,
                    "subspace_dim": (
                        None
                        if r == 0
                        else int(
                            r
                        )
                    ),
                    "query_budget": int(
                        budget
                    ),
                    "valid_normal_repeat_count":
                        int(
                            len(
                                normals[
                                    key
                                ]
                            )
                        ),
                    "valid_y_repeat_count":
                        int(
                            len(
                                y_axes[
                                    key
                                ]
                            )
                        ),
                    "normal_cos_signed_mean":
                        normal_consistency[
                            "signed_mean"
                        ],
                    "normal_cos_signed_min":
                        normal_consistency[
                            "signed_min"
                        ],
                    "normal_cos_abs_mean":
                        normal_consistency[
                            "abs_mean"
                        ],
                    "normal_angle_mean_deg":
                        normal_consistency[
                            "angle_mean_deg"
                        ],
                    "y_cos_signed_mean":
                        y_consistency[
                            "signed_mean"
                        ],
                    "y_cos_signed_min":
                        y_consistency[
                            "signed_min"
                        ],
                    "y_cos_abs_mean":
                        y_consistency[
                            "abs_mean"
                        ],
                    "y_angle_mean_deg":
                        y_consistency[
                            "angle_mean_deg"
                        ],
                }
            )

            mean_n = self._mean_unit(
                normals[
                    key
                ]
            )

            mean_y = self._mean_unit(
                y_axes[
                    key
                ]
            )

            if mean_n is not None:
                mean_normals[
                    key
                ] = mean_n

            if mean_y is not None:
                mean_y_axes[
                    key
                ] = mean_y

        # --------------------------------------------------------------
        # Cross-dimension comparison.
        # Only meaningful because the basis is nested.
        # --------------------------------------------------------------
        for estimator in subspace_estimators:
            for budget in self.diagnostic_query_budgets:
                for r1, r2 in combinations(
                    self.subspace_dims,
                    2,
                ):
                    k1 = (
                        estimator,
                        int(
                            r1
                        ),
                        int(
                            budget
                        ),
                    )

                    k2 = (
                        estimator,
                        int(
                            r2
                        ),
                        int(
                            budget
                        ),
                    )

                    if (
                        k1 not in mean_normals
                        or k2 not in mean_normals
                    ):
                        continue

                    n_cos = self._safe_cos(
                        mean_normals[
                            k1
                        ],
                        mean_normals[
                            k2
                        ],
                    )

                    y_cos = None

                    if (
                        k1 in mean_y_axes
                        and k2 in mean_y_axes
                    ):
                        y_cos = self._safe_cos(
                            mean_y_axes[
                                k1
                            ],
                            mean_y_axes[
                                k2
                            ],
                        )

                    snapshot[
                        "cross_dim"
                    ].append(
                        {
                            "estimator": estimator,
                            "query_budget": int(
                                budget
                            ),
                            "dim_a": int(
                                r1
                            ),
                            "dim_b": int(
                                r2
                            ),
                            "normal_cos_signed": float(
                                n_cos
                            ),
                            "normal_cos_abs": float(
                                abs(
                                    n_cos
                                )
                            ),
                            "y_cos_signed": (
                                None
                                if y_cos is None
                                else float(
                                    y_cos
                                )
                            ),
                            "y_cos_abs": (
                                None
                                if y_cos is None
                                else float(
                                    abs(
                                        y_cos
                                    )
                                )
                            ),
                        }
                    )

        # --------------------------------------------------------------
        # Full-parent vs each subspace mean direction.
        # Descriptive only; full parent is known to be unstable.
        # --------------------------------------------------------------
        for budget in self.diagnostic_query_budgets:
            full_key = (
                self.FULL_PARENT,
                0,
                int(
                    budget
                ),
            )

            if full_key not in mean_normals:
                continue

            for estimator in subspace_estimators:
                for r in self.subspace_dims:
                    sub_key = (
                        estimator,
                        int(
                            r
                        ),
                        int(
                            budget
                        ),
                    )

                    if sub_key not in mean_normals:
                        continue

                    n_cos = self._safe_cos(
                        mean_normals[
                            full_key
                        ],
                        mean_normals[
                            sub_key
                        ],
                    )

                    y_cos = None

                    if (
                        full_key in mean_y_axes
                        and sub_key in mean_y_axes
                    ):
                        y_cos = self._safe_cos(
                            mean_y_axes[
                                full_key
                            ],
                            mean_y_axes[
                                sub_key
                            ],
                        )

                    snapshot[
                        "full_vs_subspace"
                    ].append(
                        {
                            "subspace_estimator":
                                estimator,
                            "subspace_dim": int(
                                r
                            ),
                            "query_budget": int(
                                budget
                            ),
                            "normal_cos_signed": float(
                                n_cos
                            ),
                            "normal_cos_abs": float(
                                abs(
                                    n_cos
                                )
                            ),
                            "y_cos_signed": (
                                None
                                if y_cos is None
                                else float(
                                    y_cos
                                )
                            ),
                            "y_cos_abs": (
                                None
                                if y_cos is None
                                else float(
                                    abs(
                                        y_cos
                                    )
                                )
                            ),
                        }
                    )

        self.subspace_diagnostics.append(
            snapshot
        )

        if self.verbose == "Yes":
            print()
            print(
                f"      [SUBSPACE-NORMAL pre-Iter{iteration_index}] "
                f"attackQ={self.all_queries} "
                f"D={latent_dim} "
                f"radius={basis_info['radius_latent']:.6f}"
            )

            print(
                "         basis sanity: "
                f"orthErr="
                f"{snapshot['basis_orthogonality_max_abs_error']:.3e} "
                f"vTangDot="
                f"{snapshot['basis_tangent_v_max_abs_dot']:.3e}"
            )

            for budget in self.diagnostic_query_budgets:
                full_rc = next(
                    row
                    for row in snapshot[
                        "repeat_consistency"
                    ]
                    if (
                        row[
                            "estimator"
                        ]
                        == self.FULL_PARENT
                        and row[
                            "query_budget"
                        ]
                        == int(
                            budget
                        )
                    )
                )

                print(
                    "         "
                    f"B={budget:<4} "
                    f"{self.FULL_PARENT:<21} "
                    f"nCos="
                    f"{'NA' if full_rc['normal_cos_signed_mean'] is None else f'{full_rc['normal_cos_signed_mean']:.4f}'} "
                    f"yCos="
                    f"{'NA' if full_rc['y_cos_signed_mean'] is None else f'{full_rc['y_cos_signed_mean']:.4f}'}"
                )

                for r in self.subspace_dims:
                    for estimator in subspace_estimators:
                        rc = next(
                            row
                            for row in snapshot[
                                "repeat_consistency"
                            ]
                            if (
                                row[
                                    "estimator"
                                ]
                                == estimator
                                and row[
                                    "subspace_dim"
                                ]
                                == int(
                                    r
                                )
                                and row[
                                    "query_budget"
                                ]
                                == int(
                                    budget
                                )
                            )
                        )

                        matched = [
                            row
                            for row in snapshot[
                                "rows"
                            ]
                            if (
                                row[
                                    "estimator"
                                ]
                                == estimator
                                and row[
                                    "subspace_dim"
                                ]
                                == int(
                                    r
                                )
                                and row[
                                    "query_budget"
                                ]
                                == int(
                                    budget
                                )
                                and row[
                                    "valid"
                                ]
                            )
                        ]

                        info_rate = None

                        if estimator == self.SUBSPACE_PAIRED:
                            values = [
                                row[
                                    "informative_pair_rate"
                                ]
                                for row in matched
                                if row[
                                    "informative_pair_rate"
                                ] is not None
                            ]

                            if values:
                                info_rate = float(
                                    np.mean(
                                        values
                                    )
                                )

                        n_text = (
                            "NA"
                            if rc[
                                "normal_cos_signed_mean"
                            ] is None
                            else f"{rc['normal_cos_signed_mean']:.4f}"
                        )

                        y_text = (
                            "NA"
                            if rc[
                                "y_cos_signed_mean"
                            ] is None
                            else f"{rc['y_cos_signed_mean']:.4f}"
                        )

                        if estimator == self.SUBSPACE_PAIRED:
                            info_text = (
                                "NA"
                                if info_rate is None
                                else f"{info_rate:.3f}"
                            )

                            print(
                                "         "
                                f"B={budget:<4} "
                                f"r={r:<4} "
                                f"{estimator:<21} "
                                f"nCos={n_text:<8} "
                                f"yCos={y_text:<8} "
                                f"info={info_text}"
                            )
                        else:
                            print(
                                "         "
                                f"B={budget:<4} "
                                f"r={r:<4} "
                                f"{estimator:<21} "
                                f"nCos={n_text:<8} "
                                f"yCos={y_text:<8}"
                            )

            print(
                "         "
                f"offlineDiagnosticQueries="
                f"{snapshot['diagnostic_queries']}"
            )

        return snapshot

    # ==================================================================
    # Validated BASE trajectory + isolated offline diagnostic
    # ==================================================================
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
                "subspace_diagnostics":
                    self.subspace_diagnostics,
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
                self._run_subspace_snapshot(
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
            "subspace_diagnostics":
                self.subspace_diagnostics,
            "diagnostic_query_total": int(
                self.diagnostic_query_total
            ),
        }
