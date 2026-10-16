
import math
import random
from collections import OrderedDict

import numpy as np
import torch


class RAGTtBAPGRAttack:
    """
    RAG-TtBA with step-by-step Progress-Gated Refresh (PGR) ablations.

    Supported methods
    -----------------
    RAG_TTBA_BASE
        Full fresh Gaussian normal estimate every iteration:
            N_t = N0 * sqrt(t)
        followed by the TtBA bridge update.

    RAG_TTBA_PGR_GATE
        Novelty Step 1:
        Event-triggered normal reuse using FREE RAG pixel-space progress.
        - Good previous progress -> reuse cached normal with ZERO normal queries.
        - Stalled previous progress -> refresh the normal.
        - A safety cap forces refresh after max_reuse_streak reuses.
        - Final boundary updates are accepted only when FREE RAG pixel L2 improves.

    RAG_TTBA_PGR_ADAPT
        Novelty Step 2:
        Step 1 + stall-adaptive refresh budget.
        Mild stall -> partial refresh.
        Severe stall -> full refresh.

    RAG_TTBA_PGR
        Novelty Step 3 / full candidate:
        Step 1 + Step 2 + progress-weighted historical normal blending.
        Mild stall retains more history; severe stall trusts the fresh normal.

    Threat model
    ------------
    * Victim access is top-1 hard label only.
    * No victim logits are used by the attack logic.
    * No victim gradients are used.
    * Decoder/RAG computations are attack-side computations.
    * Every victim evaluation performed here is counted exactly once.

    RAG map expected from model.get_pixel_image(z)
    ------------------------------------------------
        Phi(z) = clip(x0 + G(z) - G(z_s), 0, 1)
    """

    SUPPORTED = {
        "RAG_TTBA_BASE",
        "RAG_TTBA_PGR_GATE",
        "RAG_TTBA_PGR_ADAPT",
        "RAG_TTBA_PGR",
    }

    PGR_METHODS = {
        "RAG_TTBA_PGR_GATE",
        "RAG_TTBA_PGR_ADAPT",
        "RAG_TTBA_PGR",
    }

    ADAPTIVE_REFRESH_METHODS = {
        "RAG_TTBA_PGR_ADAPT",
        "RAG_TTBA_PGR",
    }

    HISTORY_METHODS = {
        "RAG_TTBA_PGR",
    }

    def __init__(
        self,
        model,
        z_source,
        x_orig,
        source_label,
        loss_fn_vgg,
        ssim_fn,
        attack_method="RAG_TTBA_BASE",
        target_seed=None,
        target_label=None,
        iteration=1500,
        initial_query=30,
        tol=1e-4,
        sigma=2e-4,
        seed=992,
        query_checkpoints=None,
        max_queries=10000,
        verbose="No",
        initial_query_offset=0,
        # TtBA
        ttba_bridge_tol=1e-3,
        grad_estimator_batch_size=40,
        update_query_reserve=18,
        # PGR Step 1
        pgr_progress_threshold=0.03,
        pgr_max_reuse_streak=3,
        pgr_accept_eps=1e-8,
        # PGR Step 2
        pgr_refresh_min_ratio=0.50,
        pgr_refresh_power=1.0,
        # PGR Step 3
        pgr_history_alpha_max=0.50,
        # Query-efficiency evaluation
        l2_thresholds=None,
    ):
        if attack_method not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported={sorted(self.SUPPORTED)}"
            )

        self.model = model
        self.z_source = z_source.detach().clone()
        self.x_orig = x_orig.detach().clone()
        self.source_label = int(source_label)

        self.target_seed = None if target_seed is None else target_seed.detach().clone()
        self.target_label = None if target_label is None else int(target_label)

        self.loss_fn_vgg = loss_fn_vgg
        self.ssim_fn = ssim_fn

        self.attack_method = attack_method
        self.iteration = int(iteration)
        self.N0 = int(initial_query)
        self.tol = float(tol)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.max_queries = int(max_queries)
        self.verbose = verbose

        self.checkpoints = sorted(
            query_checkpoints
            if query_checkpoints is not None
            else [500, 1000, 2000, 5000, 10000]
        )

        self.ttba_bridge_tol = float(ttba_bridge_tol)
        self.grad_estimator_batch_size = int(grad_estimator_batch_size)
        self.update_query_reserve = int(update_query_reserve)

        self.pgr_progress_threshold = float(pgr_progress_threshold)
        self.pgr_max_reuse_streak = int(pgr_max_reuse_streak)
        self.pgr_accept_eps = float(pgr_accept_eps)

        self.pgr_refresh_min_ratio = float(pgr_refresh_min_ratio)
        self.pgr_refresh_power = float(pgr_refresh_power)
        self.pgr_history_alpha_max = float(pgr_history_alpha_max)

        self.l2_thresholds = (
            [10.0, 5.0, 3.0, 2.5, 2.0]
            if l2_thresholds is None
            else [float(x) for x in l2_thresholds]
        )

        self.device = self.z_source.device
        self.eps = 1e-12

        self.all_queries = int(initial_query_offset)
        self.q_breakdown = OrderedDict(
            target_seed=int(initial_query_offset),
            init=0,
            normal=0,
            bridge=0,
            boundary=0,
        )

        self.history = []
        self.diagnostics = []

        # PGR state.
        self.cached_normal = None
        self.prev_progress = None
        self.reuse_streak = 0

        self._set_seed(self.seed)

    # ================================================================
    # Reproducibility / victim query accounting
    # ================================================================
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def remaining_queries(self):
        return max(0, self.max_queries - self.all_queries)

    def _query_labels(self, z_batch, category):
        if z_batch.ndim == self.z_source.ndim - 1:
            z_batch = z_batch.unsqueeze(0)

        n = int(z_batch.shape[0])

        if n <= 0 or self.remaining_queries() <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        if n > self.remaining_queries():
            z_batch = z_batch[: self.remaining_queries()]
            n = int(z_batch.shape[0])

        with torch.no_grad():
            labels = self.model(z_batch).argmax(dim=1)

        self.all_queries += n
        self.q_breakdown[category] += n

        return labels

    def _status_from_labels(self, labels):
        if self.target_label is None:
            return torch.where(
                labels != self.source_label,
                torch.ones_like(labels),
                -torch.ones_like(labels),
            )

        return torch.where(
            labels == self.target_label,
            torch.ones_like(labels),
            -torch.ones_like(labels),
        )

    def is_adversarial(self, z, category="bridge"):
        labels = self._query_labels(z, category)

        if labels.numel() == 0:
            return 0

        return int(self._status_from_labels(labels)[0].item())

    # ================================================================
    # Shared initialization
    # ================================================================
    def _find_random_adversarial(self):
        for j in range(1, 501):
            if self.remaining_queries() <= 0:
                break

            pert = torch.randn_like(self.z_source)
            cand = self.z_source + (0.02 * j) * pert

            if self.is_adversarial(cand, category="init") == 1:
                return cand.detach()

        return None

    def _binary_search_known_adv(
        self,
        z_clean,
        z_adv,
        category="boundary",
        max_steps=100,
    ):
        """
        Straight latent-space boundary search.

        Endpoint states are assumed known:
          z_clean = non-adversarial,
          z_adv   = adversarial.

        Therefore no extra endpoint verification query is performed here.
        """
        clean = z_clean.detach().clone()
        adv = z_adv.detach().clone()

        for _ in range(max_steps):
            if self.remaining_queries() <= 0:
                break

            if torch.norm(adv - clean).item() <= self.tol:
                break

            mid = 0.5 * (clean + adv)
            status = self.is_adversarial(mid, category=category)

            if status == 0:
                break

            if status == 1:
                adv = mid
            else:
                clean = mid

        return adv.detach()

    def _initialize_boundary(self):
        if self.target_label is None:
            z_adv = self._find_random_adversarial()

            if z_adv is None:
                return None

        else:
            if self.target_seed is None:
                raise ValueError(
                    "Targeted mode requires a hard-label-valid target_seed."
                )

            # The runner already verified this target seed using top-1 labels.
            z_adv = self.target_seed.detach()

        return self._binary_search_known_adv(
            self.z_source,
            z_adv,
            category="init",
            max_steps=100,
        )

    # ================================================================
    # Hard-label Gaussian normal estimator
    # ================================================================
    def estimate_normal(self, z_boundary, q):
        """
        Same sign convention as the current LGC/TtBA validation code.

        Untargeted:
            source-class probe -> +u
            adversarial probe  -> -u
            final normal       -> -mean(signed u)

        Targeted:
            non-target probe   -> +u
            target probe       -> -u
            final normal       -> -mean(signed u)
        """
        q = min(int(q), self.remaining_queries())

        if q <= 0:
            return None, 0

        noises = torch.randn(
            (q,) + tuple(self.z_source.shape[1:]),
            dtype=self.z_source.dtype,
            device=self.device,
        )

        accum = torch.zeros_like(self.z_source)
        done = 0

        while done < q:
            bs = min(
                self.grad_estimator_batch_size,
                q - done,
            )

            current = noises[done : done + bs]
            noisy = z_boundary + self.sigma * current

            labels = self._query_labels(
                noisy,
                category="normal",
            )

            if labels.numel() == 0:
                break

            current = current[: labels.shape[0]]
            labels = labels[: current.shape[0]]

            if self.target_label is None:
                sign = torch.where(
                    labels == self.source_label,
                    torch.ones_like(
                        labels,
                        dtype=current.dtype,
                    ),
                    -torch.ones_like(
                        labels,
                        dtype=current.dtype,
                    ),
                )
            else:
                sign = torch.where(
                    labels != self.target_label,
                    torch.ones_like(
                        labels,
                        dtype=current.dtype,
                    ),
                    -torch.ones_like(
                        labels,
                        dtype=current.dtype,
                    ),
                )

            view_shape = (
                sign.shape[0],
            ) + (1,) * (current.ndim - 1)

            accum = accum + (
                sign.view(view_shape) * current
            ).sum(dim=0, keepdim=True)

            done += int(labels.shape[0])

        if done <= 0:
            return None, 0

        eta = -(accum / float(done))
        norm = torch.norm(eta)

        if (
            not torch.isfinite(norm)
            or norm.item() < 1e-12
        ):
            return None, done

        return eta / (norm + self.eps), done

    # ================================================================
    # TtBA bridge
    # ================================================================
    def _ttba_attempt_unit(
        self,
        x_s,
        x_b,
        eta,
        k,
    ):
        normal_v = eta / (
            torch.norm(eta) + self.eps
        )

        boundary_v = (x_b - x_s) / (
            torch.norm(x_b - x_s) + self.eps
        )

        d = (
            k * normal_v
            + (1.0 - k) * boundary_v
        )

        return d / (
            torch.norm(d) + self.eps
        )

    @staticmethod
    def _ttba_piecewise_k(L):
        # Official TtBA bridge constants.
        low = 0.10
        JUMP = 0.90
        high = 0.20
        DIVE = 2.0 / 3.0

        kk = (
            DIVE * high
            - JUMP * low
        ) / (high - low)

        if L <= low:
            return JUMP * L

        if L >= high:
            return DIVE * L

        return (
            kk * (L - low)
            + JUMP * low
        )

    def step_rag_ttba(
        self,
        x_s,
        eta,
        x_b,
    ):
        """
        TtBA bridge in RAG latent coordinates.

        Returns
        -------
        z_new : refined adversarial boundary point
        info  : bridge diagnostics
        """
        radius = torch.norm(x_b - x_s)

        if radius.item() < 1e-12:
            return x_b.detach(), {
                "L": 0.0,
                "k_app": 0.0,
            }

        L, R = 0.0, 1.0

        while abs(R - L) > self.ttba_bridge_tol:
            if self.remaining_queries() <= 0:
                break

            mid = 0.5 * (L + R)

            d = self._ttba_attempt_unit(
                x_s,
                x_b,
                eta,
                mid,
            )

            cand = x_s + radius * d

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

        k_app = self._ttba_piecewise_k(L)

        d = self._ttba_attempt_unit(
            x_s,
            x_b,
            eta,
            k_app,
        )

        adv_trial = x_s + radius * d

        z_new = self._binary_search_known_adv(
            x_s,
            adv_trial,
            category="boundary",
            max_steps=100,
        )

        return z_new.detach(), {
            "L": float(L),
            "k_app": float(k_app),
        }

    # ================================================================
    # FREE RAG distortion feedback
    # ================================================================
    def rag_pixel_l2(self, z):
        """
        Decoder/RAG-only computation.
        ZERO victim classifier queries.
        """
        with torch.no_grad():
            x = self.model.get_pixel_image(z)
            return float(
                torch.norm(
                    x - self.x_orig,
                    p=2,
                ).item()
            )

    def _relative_progress(
        self,
        before,
        after,
    ):
        return float(
            (before - after)
            / (abs(before) + self.eps)
        )

    # ================================================================
    # PGR event-triggered normal controller
    # ================================================================
    def _stall_severity(self, progress):
        """
        0 = mild/no stall
        1 = complete/severe stall

        Only called when refresh is required.
        """
        tau = max(
            self.pgr_progress_threshold,
            self.eps,
        )

        if progress is None:
            return 1.0

        severity = (
            tau - float(progress)
        ) / tau

        return float(
            max(
                0.0,
                min(1.0, severity),
            )
        )

    def _choose_normal(
        self,
        z_boundary,
        iteration_index,
    ):
        """
        Returns
        -------
        eta : normal used for this TtBA update
        info: controller diagnostics

        BASE:
            always full refresh.

        PGR_GATE:
            good progress -> reuse
            stall         -> full refresh

        PGR_ADAPT:
            good progress -> reuse
            stall         -> adaptive refresh budget

        PGR:
            PGR_ADAPT + progress-weighted history when refreshing.
        """
        q_full = int(
            self.N0
            * math.sqrt(iteration_index + 1.0)
        )
        q_full = max(q_full, 1)

        # Keep room for bridge + boundary queries.
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

        # ------------------------------------------------------------
        # Baseline: fresh full normal every iteration
        # ------------------------------------------------------------
        if self.attack_method == "RAG_TTBA_BASE":
            eta, q_used = self.estimate_normal(
                z_boundary,
                q_full,
            )

            return eta, {
                "action": "FULL_REFRESH",
                "q_full": int(q_full),
                "q_target": int(q_full),
                "q_used": int(q_used),
                "prev_progress": (
                    None
                    if self.prev_progress is None
                    else float(self.prev_progress)
                ),
                "stall_severity": 1.0,
                "history_alpha": 0.0,
                "reuse_streak": 0,
            }

        # ------------------------------------------------------------
        # First PGR iteration: must establish a reliable cached normal
        # ------------------------------------------------------------
        if self.cached_normal is None:
            eta, q_used = self.estimate_normal(
                z_boundary,
                q_full,
            )

            if eta is not None:
                self.cached_normal = eta.detach()

            self.reuse_streak = 0

            return eta, {
                "action": "INITIAL_REFRESH",
                "q_full": int(q_full),
                "q_target": int(q_full),
                "q_used": int(q_used),
                "prev_progress": None,
                "stall_severity": 1.0,
                "history_alpha": 0.0,
                "reuse_streak": 0,
            }

        # ------------------------------------------------------------
        # Step 1: event-triggered ZERO-query reuse
        # ------------------------------------------------------------
        progress_good = (
            self.prev_progress is not None
            and self.prev_progress
            >= self.pgr_progress_threshold
        )

        reuse_allowed = (
            self.reuse_streak
            < self.pgr_max_reuse_streak
        )

        if progress_good and reuse_allowed:
            self.reuse_streak += 1

            return self.cached_normal.detach(), {
                "action": "REUSE_ZERO_QUERY",
                "q_full": int(q_full),
                "q_target": 0,
                "q_used": 0,
                "prev_progress": float(
                    self.prev_progress
                ),
                "stall_severity": 0.0,
                "history_alpha": 1.0,
                "reuse_streak": int(
                    self.reuse_streak
                ),
            }

        # ------------------------------------------------------------
        # A refresh is required.
        # ------------------------------------------------------------
        severity = self._stall_severity(
            self.prev_progress
        )

        # PGR_GATE uses a full refresh.
        q_target = q_full

        # ------------------------------------------------------------
        # Step 2: stall-adaptive refresh budget
        # ------------------------------------------------------------
        if (
            self.attack_method
            in self.ADAPTIVE_REFRESH_METHODS
        ):
            q_min = max(
                1,
                int(
                    round(
                        self.pgr_refresh_min_ratio
                        * q_full
                    )
                ),
            )

            scaled = (
                severity
                ** self.pgr_refresh_power
            )

            q_target = int(
                round(
                    q_min
                    + scaled
                    * (q_full - q_min)
                )
            )

            q_target = max(
                1,
                min(q_target, q_full),
            )

        fresh, q_used = self.estimate_normal(
            z_boundary,
            q_target,
        )

        if fresh is None:
            return None, None

        history_alpha = 0.0
        eta = fresh

        # ------------------------------------------------------------
        # Step 3: progress-weighted history at refresh
        # ------------------------------------------------------------
        if (
            self.attack_method
            in self.HISTORY_METHODS
            and self.cached_normal is not None
        ):
            # Mild stall => keep more history.
            # Severe stall => trust fresh normal.
            history_alpha = (
                self.pgr_history_alpha_max
                * (1.0 - severity)
            )

            history_alpha = float(
                max(
                    0.0,
                    min(
                        self.pgr_history_alpha_max,
                        history_alpha,
                    ),
                )
            )

            eta = (
                history_alpha
                * self.cached_normal
                + (1.0 - history_alpha)
                * fresh
            )

            eta = eta / (
                torch.norm(eta) + self.eps
            )

        self.cached_normal = eta.detach()
        self.reuse_streak = 0

        return eta, {
            "action": (
                "FORCED_REFRESH"
                if progress_good and not reuse_allowed
                else "STALL_REFRESH"
            ),
            "q_full": int(q_full),
            "q_target": int(q_target),
            "q_used": int(q_used),
            "prev_progress": (
                None
                if self.prev_progress is None
                else float(self.prev_progress)
            ),
            "stall_severity": float(
                severity
            ),
            "history_alpha": float(
                history_alpha
            ),
            "reuse_streak": 0,
        }

    # ================================================================
    # Metrics / query-efficiency reporting
    # ================================================================
    def _metrics_no_victim_query(self, z_adv):
        with torch.no_grad():
            x_adv = self.model.get_pixel_image(
                z_adv
            )

            delta_lat = (
                z_adv - self.z_source
            )
            delta_pix = (
                x_adv - self.x_orig
            )

            l2_lat = float(
                torch.norm(
                    delta_lat,
                    p=2,
                ).item()
            )

            l2_pix = float(
                torch.norm(
                    delta_pix,
                    p=2,
                ).item()
            )

            ssim_value = float(
                self.ssim_fn(
                    x_adv,
                    self.x_orig,
                    data_range=1.0,
                ).detach().item()
            )

            lpips_value = float(
                self.loss_fn_vgg(
                    x_adv * 2.0 - 1.0,
                    self.x_orig * 2.0 - 1.0,
                ).detach().mean().item()
            )

        return {
            "query": int(self.all_queries),
            "l2_lat": l2_lat,
            "l2_pix": l2_pix,
            "ssim": ssim_value,
            "lpips": lpips_value,
            "asr": 1.0,
        }

    def _record(
        self,
        z_adv,
        phase,
    ):
        row = self._metrics_no_victim_query(
            z_adv
        )
        row["phase"] = str(phase)
        self.history.append(row)

        if self.verbose == "Yes":
            print(
                f"      -> [{phase:<12}] "
                f"Q={row['query']:<5d} "
                f"| Pix L2={row['l2_pix']:.4f} "
                f"| Lat L2={row['l2_lat']:.4f} "
                f"| SSIM={row['ssim']:.5f} "
                f"| LPIPS={row['lpips']:.5f}"
            )

    def checkpoint_summary(self):
        n = len(self.checkpoints)

        out = {
            "checkpoint": np.asarray(
                self.checkpoints,
                dtype=np.int64,
            ),
            "state_query": np.full(
                n,
                -1,
                dtype=np.int64,
            ),
            "l2_lat": np.full(
                n,
                np.nan,
                dtype=np.float64,
            ),
            "l2_pix": np.full(
                n,
                np.nan,
                dtype=np.float64,
            ),
            "ssim": np.full(
                n,
                np.nan,
                dtype=np.float64,
            ),
            "lpips": np.full(
                n,
                np.nan,
                dtype=np.float64,
            ),
            "asr": np.zeros(
                n,
                dtype=np.float64,
            ),
        }

        for j, cp in enumerate(
            self.checkpoints
        ):
            eligible = [
                row
                for row in self.history
                if row["query"] <= cp
            ]

            if not eligible:
                continue

            best = min(
                eligible,
                key=lambda row: row[
                    "l2_pix"
                ],
            )

            out["state_query"][j] = int(
                best["query"]
            )

            for key in [
                "l2_lat",
                "l2_pix",
                "ssim",
                "lpips",
                "asr",
            ]:
                out[key][j] = float(
                    best[key]
                )

        return out

    def threshold_summary(self):
        """
        Earliest total victim-query count at which best-so-far pixel L2
        first reaches each threshold.
        """
        result = {}

        for threshold in self.l2_thresholds:
            best_so_far = float("inf")
            q_hit = -1

            for row in self.history:
                best_so_far = min(
                    best_so_far,
                    row["l2_pix"],
                )

                if best_so_far <= threshold:
                    q_hit = int(
                        row["query"]
                    )
                    break

            result[
                float(threshold)
            ] = q_hit

        return result

    # ================================================================
    # Main
    # ================================================================
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
            }

        self._record(
            z_b,
            "Initial",
        )

        for i in range(self.iteration):
            if (
                self.remaining_queries()
                <= self.update_query_reserve
            ):
                break

            before_l2 = self.rag_pixel_l2(
                z_b
            )

            eta, normal_info = self._choose_normal(
                z_b,
                i,
            )

            if eta is None:
                break

            if self.remaining_queries() <= 0:
                break

            z_candidate, bridge_info = self.step_rag_ttba(
                self.z_source,
                eta,
                z_b,
            )

            candidate_l2 = self.rag_pixel_l2(
                z_candidate
            )

            accepted = True
            z_next = z_candidate
            after_l2 = candidate_l2

            # --------------------------------------------------------
            # RAG feedback is used only by PGR variants.
            # It costs ZERO victim queries.
            # --------------------------------------------------------
            if self.attack_method in self.PGR_METHODS:
                accepted = (
                    candidate_l2
                    < before_l2
                    - self.pgr_accept_eps
                )

                if accepted:
                    z_next = z_candidate
                    after_l2 = candidate_l2
                else:
                    # Keep the known adversarial boundary point.
                    z_next = z_b
                    after_l2 = before_l2

            progress = self._relative_progress(
                before_l2,
                after_l2,
            )

            z_b = z_next.detach()

            # Progress controls the NEXT iteration.
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
                "method": self.attack_method,
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
                        f"normal_{k}": v
                        for k, v
                        in normal_info.items()
                    }
                )

            diag.update(
                {
                    f"bridge_{k}": v
                    for k, v
                    in bridge_info.items()
                }
            )

            self.diagnostics.append(
                diag
            )

            if self.verbose == "Yes":
                if normal_info is not None:
                    print(
                        "         PGR: "
                        f"action={normal_info['action']} "
                        f"qN={normal_info['q_used']}/"
                        f"{normal_info['q_full']} "
                        f"progress={progress:.4f} "
                        f"accepted={int(accepted)} "
                        f"reuse_streak="
                        f"{normal_info['reuse_streak']}"
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
        }
