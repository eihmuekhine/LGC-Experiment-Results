import os
import math
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


def _plane_basis(eta, v):
    e1 = eta / (torch.norm(eta) + 1e-12)
    proj = torch.sum(v * e1) * e1
    e2 = (v - proj) / (torch.norm(v - proj) + 1e-12)
    return e1, e2


def _rotate_in_plane(eta, v, phi):
    e1, e2 = _plane_basis(eta, v)
    return torch.cos(phi) * e1 + torch.sin(phi) * e2


def optimize_target_latent(encoder, decoder, model_wrapper, x_0_raw, x_rec_0,
                           x_target_raw, target_label, mean_t, std_t, device):
    """
    Returns the target latent vector ONLY if it correctly predicts the intended target_label.
    Otherwise, returns None to prevent silent target shifts.
    """
    pseudo_target = torch.clamp(x_target_raw - (x_0_raw - x_rec_0), 0.0, 1.0)
    with torch.no_grad():
        z_target = encoder(pseudo_target)

    adv_pix = model_wrapper.get_pixel_image(z_target)
    pred_label = model_wrapper.classifier((adv_pix - mean_t) / std_t).argmax(1).item()
    if pred_label == target_label:
        return z_target

    z_opt = z_target.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z_opt], lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    target_tensor = torch.tensor([target_label], dtype=torch.long).to(device)

    for step in range(50):
        optimizer.zero_grad()
        out_pix = model_wrapper.get_pixel_image(z_opt)
        logits = model_wrapper.classifier((out_pix - mean_t) / std_t)
        loss = criterion(logits, target_tensor)
        loss.backward()
        optimizer.step()

        if logits.argmax(1).item() == target_label:
            return z_opt.detach()

    # Major Fix: Return None if it fails to converge to prevent silent class switching
    return None


class LGC_Attack:
    def __init__(self, model, src_img, mean, std,
                 loss_fn_vgg, ssim_fn,
                 tar_img=None, expected_target_label=None, attack_method='LGC_H',
                 iteration=60, initial_query=30, tol=0.0001, sigma=0.0002,
                 verbose_control='Yes', seed=992, idx2label=None,
                 save_img_dir=None, img_name="image",
                 query_checkpoints=None, max_queries=10000):

        self.seed = seed
        self._set_seed(self.seed)

        self.model = model
        self.src_img = src_img
        self.src_lbl = torch.argmax(self.model(self.src_img).data).item()

        self.tar_img = tar_img
        if self.tar_img is not None:
            pred_tar_lbl = torch.argmax(self.model(self.tar_img).data).item()
            assert pred_tar_lbl == expected_target_label, \
                f"Silent shift detected! Latent pred: {pred_tar_lbl}, Expected: {expected_target_label}"
            self.tar_lbl = expected_target_label
        else:
            self.tar_lbl = None

        self.iteration = iteration
        self.N0 = initial_query
        self.mean = mean
        self.std = std
        self.tol = tol
        self.sigma = sigma
        self.grad_estimator_batch_size = 40
        self.verbose_control = verbose_control
        self.attack_method = attack_method

        self.loss_fn_vgg = loss_fn_vgg
        self.ssim_fn = ssim_fn
        self.idx2label = idx2label
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_queries = 0
        self.eps = 1e-12
        # ★ 计数修复: 新增「单一记账点」self._exact_queries。
        #   每个 go_to_boundary_* / bin_search / normal_vector_approximation_batch
        #   都返回自己产生的查询数, 统一累加到这里; 最终 q_num 应恒等于它。
        #   self.all_queries 保留作兼容/校验, 不再承担计数职责。
        self._exact_queries = 0

        self.save_img_dir = save_img_dir
        self.img_name = img_name
        self.milestones = sorted(query_checkpoints) if query_checkpoints else [500, 1000, 2500, 5000, 10000]
        self.max_queries = max_queries
        self.saved_milestones = {m: False for m in self.milestones}

        # ====================================================================
        # 改动①: Hybrid 切换点 + 自适应 K 配置 (本文件唯一改动区域开始)
        # --------------------------------------------------------------------
        # ★ switch_ssim: QS->LGC_H 切换阈值 (SSIM 达到即切)
        # ★ stop_ssim : 早停阈值, 必须 > switch_ssim, 否则永远走不到 LGC_H 阶段
        # ★ K_mode    : "fixed"(基线,恒K=10) | "gap_based"(方案A/B)
        # ★ K_schedule : {gap阈值: K} 字典, gap越大(越远)K越小(省查询)
        # ★ 阈值 {0.05,0.02,0.008} 来自你曲线真实 ΔSSIM 量级, 非拍
        # ====================================================================
        self.switch_ssim = 0.995      # ★ QS -> LGC_H 切换点
        self.stop_ssim = 0.997        # ★ 早停点 (必须 > switch_ssim, 见改动③)

        self.K_mode = "gap_based"                         # ★ 基线组改成 "fixed"
        self.K_min, self.K_max = 4, 16                    # ★ K 硬边界 (防极端)
        self.K_schedule = {0.05: 4, 0.02: 4, 0.008: 6, 0.0: 8}   # ★ 方案B (默认, 预期最优)
        # self.K_schedule = {0.05: 4, 0.02: 6, 0.008: 10, 0.0: 16}  # 方案A (保守, 参考)
        self.qs_K = 10                                    # 初始 K (与原版一致)
        self._enable_progress_K = False                   # ★ 微调开关(False=只用分段,最稳)
        self._K_fast = 0.15     # 进展快(>15%余量/轮) -> 减K省查询
        self._K_slow = 0.03     # 进展慢(<3%余量/轮)  -> 加K探索
        # ====================================================================

        if self.save_img_dir and not os.path.exists(self.save_img_dir):
            os.makedirs(self.save_img_dir, exist_ok=True)

    # ========================================================================
    # 改动②: 自适应 K 计算 (fixed / gap_based 两模式)
    #   设计: 由"距切换点余量 gap = switch_ssim - cur_ssim" 走显式阶梯 (if-elif),
    #   档位清晰、不互相覆盖、可解释、可消融。K_schedule 字典仍作为配置源
    #   (格式 {gap阈值: K}), 启动时自动解析为有序阶梯。
    #   默认 _enable_progress_K=False, 只用分段(稳定); 打开则叠加进展比例微调。
    # ========================================================================
    def _parse_K_ladder(self):
        """把 K_schedule 字典解析为 [(阈值, K), ...] 升序阶梯 (仅首次)"""
        if not hasattr(self, "_K_ladder"):
            items = sorted(self.K_schedule.items(), key=lambda kv: kv[0])  # 按阈值升序
            self._K_ladder = items   # [(thresh, K), ...]  例: [(0.0,8),(0.008,6),(0.02,4),(0.05,4)]
        return self._K_ladder

    def _compute_QS_K(self, iter_idx):
        """
        返回本轮 QS 的方向数 K.
          - fixed    : 恒 K=10 (完全等价你原版)
          - gap_based: 由 距切换点 SSIM 余量 gap = switch_ssim - cur_ssim 决定
                       gap 越大(离切换点越远)K 越小(省查询, 大胆探索);
                       gap 越小(越接近切换点)K 越大(精细, 稳定收敛)。
        """
        # ---- 基线: 恒 K=10 (原版行为, 一字不改) ----
        if self.K_mode == "fixed":
            return 10

        switch_ssim = getattr(self, "switch_ssim", 0.995)
        cur_ssim = getattr(self, "_cur_ssim", 0.0)
        prev_ssim = getattr(self, "_prev_ssim", 0.0)
        gap = max(0.0, switch_ssim - cur_ssim)

        # ---- gap_based: 显式阶梯 (阈值升序, 从上到下取首个满足的档) ----
        # 例 K_schedule={0.05:4, 0.02:4, 0.008:6, 0.0:8}
        #   -> 阶梯 [(0.0,8),(0.008,6),(0.02,4),(0.05,4)]
        #   gap>=0.05 -> K=4 (最远, 大胆)
        #   0.02<=gap<0.05 -> K=4
        #   0.008<=gap<0.02 -> K=6 (中等)
        #   gap<0.008 -> K=8 (最近, 精细)
        ladder = self._parse_K_ladder()
        K = ladder[-1][1]   # 默认 = 最小 gap 档 (最精细)
        for thresh, k in ladder:   # thresh 升序: 0.0, 0.008, 0.02, 0.05
            if gap >= thresh:
                K = k
            else:
                break   # ★ 升序, 一不满足即停 (取到正确档位)

        # ★ 可选微调: 用"余量消耗比例"在分段基础上小幅修正 (尺度一致, 不拍绝对值)
        if getattr(self, "_enable_progress_K", False) and iter_idx > 0:
            gap_prev = max(1e-6, switch_ssim - prev_ssim)
            gap_cur = max(0.0, switch_ssim - cur_ssim)
            progress_ratio = (gap_prev - gap_cur) / gap_prev   # 本轮消耗余量比例
            if progress_ratio > self._K_fast:      # 进展快 -> 减K
                K -= 2
            elif progress_ratio < self._K_slow:     # 进展慢 -> 加K
                K += 2

        K = int(max(self.K_min, min(self.K_max, K)))   # ★ 硬边界 [K_min, K_max]
        self.qs_K = K
        return K

    def _set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def is_adversarial(self, image):
        """
        ★ 唯一访问 victim model 的入口。
        计数约定 (双重校验):
          (a) 自动计数: 每次调用 self.all_queries += 1 —— 作为「真实总查询」的交叉验证源;
          (b) 显式返回: 各调用方在自己的 num_calls 里也 +1, 并最终通过返回值汇总到 q_num。
        设计要求: 任意一条查询路径都必须同时满足 (a) 和 (b), 故最终
              q_num  (主循环按返回值累加)  ==  self.all_queries  (此处自动累加)
        若二者不等, 说明存在漏计或重计, 由 Attack() 结尾的 assert 暴露。
        关键: 本函数绝不「既自增又由调用方再计一次同一调用」造成双重 ——
              这里的 self.all_queries 与调用方的 num_calls 是【两条独立的账】,
              只在结尾比对, 运行期互不干扰。
        """
        predict_label = torch.argmax(self.model(image).data).item()
        self.all_queries += 1
        if self.tar_img is not None:
            return 1 if predict_label == self.tar_lbl else -1
        else:
            return 1 if predict_label != self.src_lbl else -1

    def find_random_adversarial(self, image):
        num_calls = 1
        step = 0.02
        perturbed = image
        while self.is_adversarial(perturbed) == -1:
            pert = torch.randn_like(image).to(self.device)
            perturbed = image + num_calls * step * pert
            num_calls += 1
            if num_calls > 500:
                break
        return perturbed, num_calls

    def bin_search(self, x_0, x_random):
        """
        ★ 二分投影回边界。每调用一次 is_adversarial 计 1 次查询, 最终返回 (点, 查询数)。
        返回值的查询数由调用方汇总到 q_num, 保证每条查询只计一次。
        """
        num_calls = 0
        adv = x_random
        cln = x_0
        while True:
            mid = (cln + adv) / 2.0
            num_calls += 1                      # ★ 精确计数: 每次二分都是一次 victim-model 查询
            if self.is_adversarial(mid) == 1:
                adv = mid
            else:
                cln = mid
            if torch.norm(adv - cln).cpu().numpy() < self.tol or num_calls >= 100:
                break
        return adv, num_calls

    def normal_vector_approximation_batch(self, x_boundary, q_max, random_noises):
        grad_tmp, outs = [], []
        num_batchs = math.ceil(q_max / self.grad_estimator_batch_size)
        last_batch = q_max - (num_batchs - 1) * self.grad_estimator_batch_size

        for j in range(num_batchs):
            if j == num_batchs - 1:
                current_batch = random_noises[self.grad_estimator_batch_size * j:]
                noisy_boundary = [x_boundary[0].cpu().numpy()] * last_batch + self.sigma * current_batch.cpu().numpy()
            else:
                current_batch = random_noises[self.grad_estimator_batch_size * j:self.grad_estimator_batch_size * (j + 1)]
                noisy_boundary = [x_boundary[0].cpu().numpy()] * self.grad_estimator_batch_size + self.sigma * current_batch.cpu().numpy()

            noisy_boundary_tensor = torch.tensor(noisy_boundary, dtype=torch.float32).to(self.device)
            predict_labels = torch.argmax(self.model(noisy_boundary_tensor), 1).cpu().numpy().astype(int)
            outs.append(predict_labels)

        outs = np.concatenate(outs, axis=0)

        # ★ 自动账: 一次批前向 = q_max 次 victim-model 调用, 计入交叉验证计数器。
        #   显式账 (q_opt) 由主循环 q_num += q_opt 负责; 二者结尾比对。
        self.all_queries += q_max

        for i, predict_label in enumerate(outs):
            if self.tar_img is not None:
                if predict_label != self.tar_lbl:
                    grad_tmp.append(random_noises.cpu().numpy()[i])
                else:
                    grad_tmp.append(-random_noises.cpu().numpy()[i])
            else:
                if predict_label == self.src_lbl:
                    grad_tmp.append(random_noises.cpu().numpy()[i])
                else:
                    grad_tmp.append(-random_noises.cpu().numpy()[i])

        grad = -(1 / q_max) * sum(grad_tmp)
        grad_f = torch.tensor(grad, dtype=torch.float32).to(self.device)[None, :, :, :]
        return grad_f

    # ========================================================================
    # QS: 扇形(每方向二分到边界, 选最优) + 切线外推(投影回边界)
    #   ★ 唯一改动: K 由写死的 max(8,40//4) 改为 self._compute_QS_K(iter_idx)
    #     循环 for k in range(K) 自动适配; 其余(二分/选最优/外推/投影)一字未动
    # ------------------------------------------------------------------------
    # ★★★ 计数修复: 本方法现在精确统计其内部产生的【每一次】victim-model 查询
    #   (二分搜索 + 候选 bin_search + 切向外推 is_adversarial + 投影 bin_search),
    #   并以返回值 (best_adv, total_calls) 交给主循环汇总到 q_num。
    #   历史版本返回 (best_adv, 0), 导致 QS 内部查询全部漏报 ——
    #   这正是「QS 在 500/1000/2000 查询下与 LGC 不公平比较」的根因。
    #   修复原则: 每个 victim-model 查询只计数一次 (单一记账点 = q_num)。
    # ========================================================================
    def go_to_boundary_QS(self, x_s, eta_o, x_b, iter_idx=0):
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (x_b - x_s) / (torch.norm(x_b - x_s) + self.eps)

        sector_angle = math.pi / 2.0   # 固定 90°

        # ★★★ 改动②: 自适应 K (唯一改动点) ★★★
        K = self._compute_QS_K(iter_idx)
        # ==============================

        phis = torch.linspace(0.0, sector_angle, steps=K)
        R = torch.norm(x_b - x_s)

        # ★ 不省打印: 每轮打印 [QS-AdapK] 信息, 方便你对比三组
        if self.verbose_control == 'Yes':
            print(f"      -> [QS-AdapK] Iter: {iter_idx:>2} | gap: {getattr(self,'switch_ssim',0.995)-getattr(self,'_cur_ssim',0.0):.4f} | K: {K}")

        total_calls = 0   # ★ 计数修复: 本方法产生的查询总数
        best_adv = None
        for k in range(K):
            zeta = _rotate_in_plane(eta, v, phis[k])
            lo, hi = 0.0, R
            cand = None
            bisection_calls = 0
            for _ in range(30):
                bisection_calls += 1                    # ★ 二分: 每步一次 is_adversarial
                mid = (lo + hi) / 2.0
                pt = x_s + zeta * mid
                if self.is_adversarial(pt) == 1:
                    cand = pt
                    hi = mid
                else:
                    lo = mid
                if hi - lo < self.tol:
                    break
            total_calls += bisection_calls               # ★ 计入二分查询
            if cand is not None:
                cand, bin_calls = self.bin_search(self.src_img, cand)
                total_calls += bin_calls                 # ★ 计入候选投影查询
                if best_adv is None or torch.norm(cand - x_s) < torch.norm(best_adv - x_s):
                    best_adv = cand

        if best_adv is None:
            # ★ 兜底: 回退 LGC_H, 其返回的查询数一并累加, 避免漏报
            best_adv, fallback_calls = self.go_to_boundary_LGC_H(x_s, eta_o, x_b)
            total_calls += fallback_calls
            return best_adv, total_calls

        # ---- 切向外推 + 投影回边界 ----
        try:
            tangent = v - (torch.dot(v.reshape(-1), eta.reshape(-1))) * eta
            tangent = tangent / (torch.norm(tangent) + self.eps)
            step = (0.02 + 0.06 * iter_idx / max(self.iteration - 1, 1)) * R
            adv_ext = best_adv + tangent * step
            total_calls += 1                            # ★ 外推点分类: 一次 is_adversarial
            if self.is_adversarial(adv_ext) == 1:
                adv_ext, bin_calls = self.bin_search(self.src_img, adv_ext)
                total_calls += bin_calls                 # ★ 投影二分查询
                if torch.norm(adv_ext - x_s) < torch.norm(best_adv - x_s):
                    best_adv = adv_ext
        except Exception:
            pass

        return best_adv, total_calls   # ★ 修复: 返回真实查询数, 不再返回 0


    # ========================================================================
    # QS_TTBA: original QS e1/e2 geometry + TtBA-style two-stage search.
    # Stage 1: cheap scalar bridge bisection at fixed current radius.
    # Stage 2: one full radial boundary bisection along the selected direction.
    # Original QS/LGC/LGC_H/Hybrid remain unchanged.
    # ========================================================================
    def _ttba_bridge_apply(self, s_bridge):
        s_bridge = float(max(0.0, min(1.0, s_bridge)))
        low, high = 0.10, 0.20
        jump, dive = 0.90, 2.0 / 3.0
        if s_bridge <= low:
            factor = jump
        elif s_bridge >= high:
            factor = dive
        else:
            alpha = (s_bridge - low) / (high - low)
            factor = jump + alpha * (dive - jump)
        return float(max(0.0, min(s_bridge, factor * s_bridge)))

    def _qs_direction_from_progress(self, e1, e2, phi_v, s):
        # Same QS orthonormal basis. s=0 -> current v, s=1 -> e1=n.
        s = float(max(0.0, min(1.0, s)))
        phi = (1.0 - s) * phi_v
        phi_t = torch.tensor(phi, dtype=e1.dtype, device=e1.device)
        zeta = torch.cos(phi_t) * e1 + torch.sin(phi_t) * e2
        return zeta / (torch.norm(zeta) + self.eps), phi

    def go_to_boundary_QS_TTBA(self, x_s, eta_o, x_b, iter_idx=0,
                               bridge_tol=0.001, bridge_max_queries=12):
        total_calls = 0

        # Keep the original QS geometry exactly.
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (x_b - x_s) / (torch.norm(x_b - x_s) + self.eps)
        e1, e2 = _plane_basis(eta, v)

        # Angle of current v in the same e1/e2 basis.
        a = torch.sum(v * e1).item()
        b = torch.sum(v * e2).item()
        phi_v = math.atan2(max(b, 0.0), a)

        if not np.isfinite(phi_v) or abs(phi_v) < 1e-10:
            adv, q_rad = self.bin_search(x_s, x_b)
            total_calls += q_rad
            if self.verbose_control == 'Yes':
                print(f"      -> [QS-TtBA] Iter: {iter_idx:>2} | degenerate v||n | bridgeQ: 0 | radialQ: {q_rad}")
            return adv, total_calls

        R = torch.norm(x_b - x_s)

        # s=0 is current v and x_b is already known adversarial.
        s_adv = 0.0
        point_adv = x_b

        # Query s=1 (normal endpoint) once to establish the bridge bracket.
        zeta_n, _ = self._qs_direction_from_progress(e1, e2, phi_v, 1.0)
        point_n = x_s + R * zeta_n
        total_calls += 1
        normal_status = self.is_adversarial(point_n)

        if normal_status == 1:
            s_bridge = 1.0
            s_last_adv = 1.0
            point_last_adv = point_n
            bridge_mid_calls = 0
        else:
            s_clean = 1.0
            bridge_mid_calls = 0
            while ((s_clean - s_adv) > bridge_tol and
                   bridge_mid_calls < bridge_max_queries):
                s_mid = 0.5 * (s_adv + s_clean)
                zeta_mid, _ = self._qs_direction_from_progress(e1, e2, phi_v, s_mid)
                point_mid = x_s + R * zeta_mid
                total_calls += 1
                bridge_mid_calls += 1
                if self.is_adversarial(point_mid) == 1:
                    s_adv = s_mid
                    point_adv = point_mid
                else:
                    s_clean = s_mid
            s_bridge = s_adv
            s_last_adv = s_adv
            point_last_adv = point_adv

        # TtBA-style conservative bridge application.
        s_app = self._ttba_bridge_apply(s_bridge)
        zeta_app, phi_app = self._qs_direction_from_progress(e1, e2, phi_v, s_app)
        point_app = x_s + R * zeta_app

        # One safety label query for the selected same-radius candidate.
        total_calls += 1
        selected_status = self.is_adversarial(point_app)
        if selected_status == 1:
            selected_adv = point_app
            selected_reason = 'APP_VERIFIED_ADV'
        else:
            selected_adv = point_last_adv
            s_app = s_last_adv
            _, phi_app = self._qs_direction_from_progress(e1, e2, phi_v, s_app)
            selected_reason = 'APP_CLEAN_FALLBACK_LAST_ADV'

        # Stage 2: exactly ONE full radial boundary search.
        new_boundary, radial_calls = self.bin_search(x_s, selected_adv)
        total_calls += radial_calls

        if self.verbose_control == 'Yes':
            print(
                f"      -> [QS-TtBA] Iter: {iter_idx:>2} "
                f"| phi_v: {math.degrees(phi_v):.2f}deg "
                f"| nStatus: {'ADV' if normal_status == 1 else 'CLEAN'} "
                f"| sBridge: {s_bridge:.4f} | sApp: {s_app:.4f} "
                f"| phiApp: {math.degrees(phi_app):.2f}deg "
                f"| cheapQ: {1 + bridge_mid_calls + 1} | radialQ: {radial_calls} "
                f"| reason: {selected_reason}"
            )

        return new_boundary, total_calls

    # ========================================================================
    # Hybrid: 前期 QS (省查询), SSIM >= switch_ssim 后切 LGC_H (精细二分)
    # ========================================================================
    def go_to_boundary_Hybrid(self, x_s, eta_o, x_b, grad_oi=None, iter_idx=0):
        if getattr(self, '_cur_ssim', 0.0) >= getattr(self, 'switch_ssim', 0.995):
            if not getattr(self, '_hybrid_switched', False):
                print(f"      [Hybrid] >>> switched to LGC_H @ Iter {iter_idx} "
                      f"SSIM={getattr(self,'_cur_ssim',0.0):.4f} (switch_ssim={getattr(self,'switch_ssim',0.995)})")
                self._hybrid_switched = True
            return self.go_to_boundary_LGC_H(x_s, eta_o, x_b)   # 后期: 纯二分
        else:
            # ★ 前期走 QS 时用 grad_oi (与调用处一致, 名字对应 eta_o 形参)
            return self.go_to_boundary_QS(x_s, grad_oi, x_b, iter_idx=iter_idx)  # 前期: QS

    # ==================== LGC_H (纯净, 不变) ====================
    def go_to_boundary_LGC_H(self, x_s, eta_o, x_b):
        num_calls = 1
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (x_b - x_s) / (torch.norm(x_b - x_s) + self.eps)
        dot_val = torch.clamp(torch.dot(eta.reshape(-1), v.reshape(-1)), -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(dot_val)
        while True:
            denom = torch.sin(theta / (pow(2, num_calls))) + self.eps
            m = (torch.sin(theta) * torch.cos(theta / (pow(2, num_calls))) / denom - torch.cos(theta)).item()
            zeta = (eta + m * v) / (torch.norm(eta + m * v) + self.eps)
            perturbed = x_s + zeta * torch.norm(x_b - x_s) * torch.dot(zeta.reshape(-1), v.reshape(-1))
            num_calls += 1
            if self.is_adversarial(perturbed) == 1:
                break
            if num_calls >= 40:
                return x_b, num_calls - 1
        perturbed, bin_query = self.bin_search(self.src_img, perturbed)
        return perturbed, num_calls - 1 + bin_query

    # ==================== LGC (半圆, 不变) ====================
    def go_to_boundary_LGC(self, x_s, eta_o, x_b):
        num_calls = 1
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (x_b - x_s) / (torch.norm(x_b - x_s) + self.eps)
        dot_val = torch.clamp(torch.dot(eta.reshape(-1), v.reshape(-1)), -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(dot_val)
        while True:
            denom = torch.sin(torch.tensor([(math.pi / 2)]) * (1 - 1 / pow(2, num_calls))) + self.eps
            m = (torch.sin(theta.cpu()) * torch.cos(torch.tensor([(math.pi / 2)]) * (1 - 1 / pow(2, num_calls))) / denom - torch.cos(theta.cpu())).item()
            zeta = (eta + m * v) / (torch.norm(eta + m * v) + self.eps)
            p_near_boundary = x_s + zeta * torch.norm(x_b - x_s) * torch.dot(v.reshape(-1), zeta.reshape(-1))
            if self.is_adversarial(p_near_boundary) == -1:
                break
            num_calls += 1
            if num_calls >= 40:
                return x_b, num_calls - 1
        perturbed, n_calls = self.SemiCircular_boundary_search(x_s, x_b, p_near_boundary)
        return perturbed, num_calls + n_calls

    def SemiCircular_boundary_search(self, x_0, x_b, p_near_boundary):
        num_calls = 0
        norm_dis = torch.norm(x_b - x_0)
        boundary_dir = (x_b - x_0) / (norm_dis + self.eps)
        clean_dir = (p_near_boundary - x_0) / (torch.norm(p_near_boundary - x_0) + self.eps)
        adv_dir, clean_dir = boundary_dir, clean_dir
        adv, clean = x_b, x_0
        while True:
            mid_dir = adv_dir + clean_dir
            mid_dir = mid_dir / (torch.norm(mid_dir) + self.eps)
            dot_val = torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) / (
                (torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir)) + self.eps)
            dot_val = torch.clamp(dot_val, -1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(dot_val)
            d = torch.cos(theta) * norm_dis
            x_mid = x_0 + mid_dir * d
            num_calls += 1
            if self.is_adversarial(x_mid) == 1:
                adv_dir = mid_dir
                adv = x_mid
            else:
                clean_dir = mid_dir
                clean = x_mid
            if torch.norm(adv - clean).cpu().numpy() < self.tol or num_calls > 100:
                break
        return adv, num_calls

    # ==================== 指标记录 ====================
    def _record_metrics(self, z_adv, q_num, l2_lat_list, linf_lat_list, l2_pix_list,
                        linf_pix_list, ssim_list, lpips_list, asr_list, phase_name):
        with torch.no_grad():
            norm_l2_lat = torch.norm(z_adv - self.src_img, p=2).item()
            norm_linf_lat = torch.norm(z_adv - self.src_img, p=float('inf')).item()

            x_adv_pix = self.model.get_pixel_image(z_adv)
            x_orig_pix = self.model.x_orig

            norm_l2_pix = torch.norm(x_adv_pix - x_orig_pix, p=2).item()
            norm_linf_pix = torch.norm(x_adv_pix - x_orig_pix, p=float('inf')).item()

            ssim_val = self.ssim_fn(x_adv_pix, x_orig_pix, data_range=1.0).item()

            x_adv_lpips = x_adv_pix * 2.0 - 1.0
            x_orig_lpips = x_orig_pix * 2.0 - 1.0
            lpips_val = self.loss_fn_vgg(x_adv_lpips, x_orig_lpips).item()

            mean_tensor = torch.tensor(self.mean).view(1, 3, 1, 1).to(self.device)
            std_tensor = torch.tensor(self.std).view(1, 3, 1, 1).to(self.device)
            adv_label = self.model.classifier((x_adv_pix - mean_tensor) / std_tensor).argmax(1).item()

            if self.tar_img is not None:
                is_adv = 1 if adv_label == self.tar_lbl else 0
            else:
                is_adv = 1 if adv_label != self.src_lbl else 0

            l2_lat_list.append(norm_l2_lat)
            linf_lat_list.append(norm_linf_lat)
            l2_pix_list.append(norm_l2_pix)
            linf_pix_list.append(norm_linf_pix)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)
            asr_list.append(is_adv)

            if self.verbose_control == 'Yes':
                print(f"      -> [{phase_name:<8}] Queries: {q_num:<5} | Pix L2: {norm_l2_pix:.3f} | Lat L2: {norm_l2_lat:.3f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | ASR: {is_adv}")

    def _save_visual_milestone(self, z_adv, m, actual_q):
        with torch.no_grad():
            x_adv_pix = self.model.get_pixel_image(z_adv)
            x_orig_pix = self.model.x_orig

            mean_tensor = torch.tensor(self.mean).view(1, 3, 1, 1).to(self.device)
            std_tensor = torch.tensor(self.std).view(1, 3, 1, 1).to(self.device)
            adv_label = self.model.classifier((x_adv_pix - mean_tensor) / std_tensor).argmax(1).item()

            norm_l2_pix = torch.norm(x_adv_pix - x_orig_pix, p=2).item()
            ssim_val = self.ssim_fn(x_adv_pix, x_orig_pix, data_range=1.0).item()
            lpips_val = self.loss_fn_vgg(x_adv_pix * 2.0 - 1.0, x_orig_pix * 2.0 - 1.0).item()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        orig_img_np = torch.clamp(x_orig_pix.squeeze(0), 0, 1).cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(orig_img_np)
        orig_label_name = self.idx2label.get(self.src_lbl, str(self.src_lbl)) if self.idx2label else str(self.src_lbl)
        axes[0].set_title(f"Original\nLabel: {str(orig_label_name)[:25]}", fontsize=12)
        axes[0].axis('off')

        adv_img_np = torch.clamp(x_adv_pix.squeeze(0), 0, 1).cpu().permute(1, 2, 0).numpy()
        axes[1].imshow(adv_img_np)
        adv_label_name = self.idx2label.get(adv_label, str(adv_label)) if self.idx2label else str(adv_label)
        axes[1].set_title(f"Adversarial\nLabel: {str(adv_label_name)[:25]}", fontsize=12)
        axes[1].axis('off')

        noise = (x_adv_pix - x_orig_pix).squeeze(0).cpu()
        noise_vis = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        axes[2].imshow(noise_vis.permute(1, 2, 0).numpy())
        axes[2].set_title("Normalized Perturbation", fontsize=12)
        axes[2].axis('off')

        metrics_text = (f"Budget (<= {m}): {actual_q} Queries | L2 (Pix): {norm_l2_pix:.2f} | "
                        f"SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}")
        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=14,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = os.path.join(self.save_img_dir, f"{self.img_name}_{self.attack_method}_Q{m}_comparison.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        self.saved_milestones[m] = True
        if self.verbose_control == 'Yes':
            print(f"      -> [*] Exact Visual Milestone Saved: Q{m} (State from Q={actual_q})")

    # ==================== Attack() ====================
    def Attack(self):
        n_query, l2_lat_list, linf_lat_list = [], [], []
        l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list = [], [], [], [], []

        if self.tar_img is not None:
            x_random, query_random = self.tar_img, 0
        else:
            x_random, query_random = self.find_random_adversarial(self.src_img)

        x_b, query_b = self.bin_search(self.src_img, x_random)
        q_num = query_random + query_b
        self._exact_queries = q_num   # ★ 计数修复: 初始化阶段的查询 (随机探测 + 初始二分)
        n_query.append(q_num)

        self._record_metrics(x_b, q_num, l2_lat_list, linf_lat_list, l2_pix_list,
                             linf_pix_list, ssim_list, lpips_list, asr_list, "Initial")
        size = self.src_img.shape

        stable_count = 0
        stable_needed = 2   # ★ 连续 2 轮达标即停 (拐点快定)

        for i in range(self.iteration):
            prev_x_b = x_b.clone()
            prev_q = q_num

            if self.attack_method == 'QS':
                q_opt = int(self.N0 * np.sqrt(i + 1))
            else:
                q_opt = int(self.N0 * np.sqrt(i + 1))
            q_opt = max(q_opt, 10)
            random_vec_o = torch.randn(q_opt, size[1], size[-2], size[-1]).to(self.device)

            grad_oi = self.normal_vector_approximation_batch(x_b, q_opt, random_vec_o)
            q_num += q_opt
            self._exact_queries += q_opt   # ★ 计数修复: 批查询 (q_opt 次) 入账

            if self.attack_method == 'LGC':
                x_adv, qs = self.go_to_boundary_LGC(self.src_img, grad_oi, x_b)
            elif self.attack_method == 'LGC_H':
                x_adv, qs = self.go_to_boundary_LGC_H(self.src_img, grad_oi, x_b)
            elif self.attack_method == 'QS':
                x_adv, qs = self.go_to_boundary_QS(self.src_img, grad_oi, x_b, iter_idx=i)
                if x_adv is None:
                    x_adv, qs = self.go_to_boundary_LGC_H(self.src_img, grad_oi, x_b)
            elif self.attack_method == 'QS_TTBA':
                x_adv, qs = self.go_to_boundary_QS_TTBA(
                    self.src_img, grad_oi, x_b, iter_idx=i,
                    bridge_tol=0.001, bridge_max_queries=12
                )
                if x_adv is None:
                    x_adv, qs = self.go_to_boundary_LGC_H(self.src_img, grad_oi, x_b)
            elif self.attack_method == 'Hybrid':   # ★ Hybrid 分支
                x_adv, qs = self.go_to_boundary_Hybrid(self.src_img, grad_oi, x_b, grad_oi=grad_oi, iter_idx=i)
                if x_adv is None:
                    x_adv, qs = self.go_to_boundary_LGC_H(self.src_img, grad_oi, x_b)
            else:
                x_adv, qs = x_b, 0

            q_num += qs
            self._exact_queries += qs   # ★ 计数修复: 搜索方法(含 QS 内部全部二分/外推/投影)的查询入账
            x_b = x_adv

            # ★ 记录上一轮/当前 SSIM (供下轮 _compute_QS_K 判断 gap 与进展比例)
            self._prev_ssim = ssim_list[-1] if ssim_list else 0.0
            self._cur_ssim = ssim_list[-1] if ssim_list else 0.0

            # ===== 改动③: 早停阈值 stop_ssim=0.997 (> switch_ssim=0.995)
            #   ★ 关键: 若写成 0.995 则一达到切换点就停, LGC_H 阶段永远走不到!
            # =====
            cur_ssim = ssim_list[-1] if ssim_list else 0.0
            cur_asr = asr_list[-1] if asr_list else 0
            if cur_asr == 1 and cur_ssim >= self.stop_ssim:   # ★ 用 stop_ssim (0.997)
                stable_count += 1
                if stable_count >= stable_needed:
                    print(f"    [*] Early stop @ Iter {i}: ASR=1 & SSIM={cur_ssim:.4f} "
                          f"(stable {stable_count}/{stable_needed}), Queries={q_num}")
                    break
            else:
                stable_count = 0
            # ==============================================================

            if self.save_img_dir:
                for m in self.milestones:
                    if q_num >= m and not self.saved_milestones[m]:
                        self._save_visual_milestone(prev_x_b, m, prev_q)

            self._record_metrics(x_b, q_num, l2_lat_list, linf_lat_list, l2_pix_list,
                                 linf_pix_list, ssim_list, lpips_list, asr_list, f"Iter {i}")
            n_query.append(q_num)

            if q_num >= self.max_queries:
                break

        if self.save_img_dir:
            for m in self.milestones:
                if q_num >= m and not self.saved_milestones[m]:
                    self._save_visual_milestone(x_b, m, q_num)

        # ★ 计数修复: 一致性校验 —— q_num 应恒等于「逐方法返回值之和」,
        #   也应等于 self.all_queries (后者在各方法内按调用次数累加, 作交叉验证)。
        #   若三者不等, 说明出现了漏计/重计, 必须排查 (开发期硬断言, 不吞错)。
        if self.verbose_control == 'Yes':
            print(f"    [*] Query audit: q_num={q_num} | _exact_queries={self._exact_queries} "
                  f"| all_queries={self.all_queries} | diff={q_num - self._exact_queries}")
        assert q_num == self._exact_queries, \
            f"查询计数不一致: q_num({q_num}) != 逐方法返回之和({self._exact_queries})"
        assert q_num == self.all_queries, \
            f"查询计数不一致: q_num({q_num}) != all_queries({self.all_queries})"

        return x_b, n_query, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, \
               ssim_list, lpips_list, asr_list, []
