import os
import math
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

# RAG-PBTR: RAG-aware Pullback-metric Boundary Trust-Region optimizer.
# The victim model is used only through hard-label decisions. Decoder derivatives
# are white-box with respect to the attacker's own autoencoder, not the victim.

def _ensure_torch():
    global torch
    if torch is None:
        import torch as _t
        torch = _t

def optimize_target_latent(encoder, decoder, model_wrapper, x_0_raw, x_rec_0, x_target_raw, target_label, mean_t, std_t, device):
    """Strict hard-label-safe pseudo-target initializer.

    IMPORTANT: the previous version optimized ``z_target`` with cross-entropy
    gradients through the victim classifier. That violates a decision-based
    black-box threat model. This revision only checks the victim's top-1 label.
    If the encoded pseudo-target is not already in the requested target class,
    it returns ``None`` and the caller must use a hard-label-compatible
    initialization strategy (e.g. another valid target-class sample).
    """
    pseudo_target = torch.clamp(x_target_raw - (x_0_raw - x_rec_0), 0.0, 1.0)
    with torch.no_grad():
        z_target = encoder(pseudo_target)
        adv_pix = model_wrapper.get_pixel_image(z_target)
        pred_label = model_wrapper.classifier((adv_pix - mean_t) / std_t).argmax(1).item()
    return z_target if pred_label == target_label else None

class LGC_Attack:
    def __init__(self, model, src_img, mean, std, 
                 loss_fn_vgg, ssim_fn, 
                 tar_img=None, expected_target_label=None, attack_method='NGQS',
                 iteration=60, initial_query=30, tol=0.0001, sigma=0.0002,
                 verbose_control='Yes', seed=992, idx2label=None,
                 save_img_dir=None, img_name="image",
                 query_checkpoints=None, max_queries=2000,
                 pbtr_damping=1e-3, pbtr_cg_iters=6,
                 pbtr_trust_scale=0.12, pbtr_trust_min=0.02, pbtr_trust_max=0.50,
                 pbtr_expand=1.25, pbtr_shrink=0.50,
                 pbtr_refine_steps=8, pbtr_normal_queries=12,
                 pbtr_history_beta=0.50, pbtr_use_orthogonal_probes=True): 
        
        self.seed = seed
        self._set_seed(self.seed)

        self.model = model
        self.src_img = src_img
        self.src_lbl = torch.argmax(self.model(self.src_img).data).item()
        
        self.tar_img = tar_img
        if self.tar_img is not None:
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

        # ---------------- RAG-PBTR state ----------------
        # M(z) = J_G(z)^T J_G(z) + lambda I is never formed explicitly.
        self.pbtr_damping = float(pbtr_damping)
        self.pbtr_cg_iters = int(pbtr_cg_iters)
        self.pbtr_trust_scale = float(pbtr_trust_scale)
        self.pbtr_trust_min = float(pbtr_trust_min)
        self.pbtr_trust_max = float(pbtr_trust_max)
        self.pbtr_expand = float(pbtr_expand)
        self.pbtr_shrink = float(pbtr_shrink)
        self.pbtr_refine_steps = int(pbtr_refine_steps)
        self.pbtr_normal_queries = int(pbtr_normal_queries)
        self.pbtr_history_beta = float(pbtr_history_beta)
        self.pbtr_use_orthogonal_probes = bool(pbtr_use_orthogonal_probes)
        self._pbtr_prev_normal = None
        self.pbtr_diagnostics = []

        # Decoder derivatives must be deterministic during pullback products.
        if hasattr(self.model, 'decoder'):
            self.model.decoder.eval()

        # Cache G(z_s). In RAG, the unclipped chord is G(z)-G(z_s).
        with torch.no_grad():
            if hasattr(self.model, 'x_rec_0'):
                self._rag_base_rec = self.model.x_rec_0.detach().to(self.device)
            elif hasattr(self.model, 'decoder'):
                self._rag_base_rec = self.model.decoder(self.src_img).detach()
            else:
                self._rag_base_rec = None

        self.save_img_dir = save_img_dir
        self.img_name = img_name
        self.milestones = sorted(query_checkpoints) if query_checkpoints else [500, 1000, 1500, 2000]
        self.max_queries = max_queries
        self.saved_milestones = {m: False for m in self.milestones}
        
        if self.save_img_dir and not os.path.exists(self.save_img_dir):
            os.makedirs(self.save_img_dir, exist_ok=True)

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
        predict_label = torch.argmax(self.model(image).data).item()
        self.all_queries += 1
        if self.tar_img is not None: return 1 if predict_label == self.tar_lbl else -1
        else: return 1 if predict_label != self.src_lbl else -1

    def find_random_adversarial(self, image):
        num_calls = 0       
        step = 0.02
        perturbed = image        
        while True:          
            pert = torch.randn_like(image).to(self.device)
            num_calls += 1   
            perturbed = image + num_calls * step * pert
            if self.is_adversarial(perturbed) == 1:
                break
            if num_calls > 500: break
        return perturbed, num_calls 
    
    def bin_search(self, x_0, x_random):  
        num_calls = 0
        adv, cln = x_random, x_0      
        while True:         
            mid = (cln + adv) / 2.0
            num_calls += 1           
            if self.is_adversarial(mid) == 1: adv = mid
            else: cln = mid   
            if torch.norm(adv - cln).cpu().numpy() < self.tol or num_calls >= 100: break       
        return adv, num_calls 

    def bin_search_between(self, x_adv, x_nonadv):
        num_calls = 0
        adv, cln = x_adv.clone(), x_nonadv.clone()
        while True:
            mid = (adv + cln) / 2.0
            num_calls += 1
            if self.is_adversarial(mid) == 1:
                adv = mid
            else:
                cln = mid
            if torch.norm(adv - cln).cpu().numpy() < self.tol or num_calls >= 50:
                break
        return adv, num_calls
    
    def normal_vector_approximation_batch(self, x_boundary, q_max, random_noises, sigma_t):    
        grad_tmp, outs = [], []
        num_batchs = math.ceil(q_max / self.grad_estimator_batch_size)
        last_batch = q_max - (num_batchs - 1) * self.grad_estimator_batch_size        
        
        for j in range(num_batchs):            
            if j == num_batchs - 1:
                current_batch = random_noises[self.grad_estimator_batch_size * j:]
                noisy_boundary = [x_boundary[0].cpu().numpy()] * last_batch + sigma_t * current_batch.cpu().numpy()   
            else:
                current_batch = random_noises[self.grad_estimator_batch_size * j:self.grad_estimator_batch_size * (j + 1)]
                noisy_boundary = [x_boundary[0].cpu().numpy()] * self.grad_estimator_batch_size + sigma_t * current_batch.cpu().numpy()   
                
            noisy_boundary_tensor = torch.tensor(noisy_boundary, dtype=torch.float32).to(self.device)   
            predict_labels = torch.argmax(self.model(noisy_boundary_tensor), 1).cpu().numpy().astype(int)              
            outs.append(predict_labels)  
            
        outs = np.concatenate(outs, axis=0)
        self.all_queries += q_max
        
        for i, predict_label in enumerate(outs):
            if self.tar_img is not None:
                grad_tmp.append(random_noises.cpu().numpy()[i] if predict_label != self.tar_lbl else -random_noises.cpu().numpy()[i])
            else:
                grad_tmp.append(random_noises.cpu().numpy()[i] if predict_label == self.src_lbl else -random_noises.cpu().numpy()[i]) 
                    
        grad = -(1 / q_max) * sum(grad_tmp)       
        grad_f = torch.tensor(grad, dtype=torch.float32).to(self.device)[None, :, :, :]    
        return grad_f

    # ========================== METHOD: LGC ==========================
    def go_to_boundary_LGC(self, x_s, eta_o, x_b):
        num_calls = 0
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (x_b - x_s) / (torch.norm(x_b - x_s) + self.eps)
        
        dot_val = torch.clamp(torch.dot(eta.reshape(-1), v.reshape(-1)), -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(dot_val)
        
        loop_calls = 1
        while True:   
            denom = torch.sin(torch.tensor([(math.pi / 2)]) * (1 - 1 / pow(2, loop_calls))) + self.eps
            m = (torch.sin(theta.cpu()) * torch.cos(torch.tensor([(math.pi / 2)]) * (1 - 1 / pow(2, loop_calls))) / denom - torch.cos(theta.cpu())).item()
            zeta = (eta + m * v) / (torch.norm(eta + m * v) + self.eps)
            p_near_boundary = x_s + zeta * torch.norm(x_b - x_s) * torch.dot(v.reshape(-1), zeta.reshape(-1)) 
            
            num_calls += 1
            if self.is_adversarial(p_near_boundary) == -1: break
            loop_calls += 1
            if loop_calls >= 40: return x_b, num_calls
        
        num_calls_search = 0
        adv, clean = x_b, x_s
        norm_dis = torch.norm(x_b - x_s)
        boundary_dir = (x_b - x_s) / (norm_dis + self.eps)
        clean_dir = (p_near_boundary - x_s) / (torch.norm(p_near_boundary - x_s) + self.eps)
        adv_dir = boundary_dir
        
        while True:
            mid_dir = adv_dir + clean_dir
            mid_dir = mid_dir / (torch.norm(mid_dir) + self.eps)
            d = torch.cos(torch.acos(torch.clamp(torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)), -1.0, 1.0))) * norm_dis
            x_mid = x_s + mid_dir * d
            num_calls_search += 1
            if self.is_adversarial(x_mid) == 1:
                adv_dir, adv = mid_dir, x_mid  
            else:
                clean_dir, clean = mid_dir, x_mid                            
            if torch.norm(adv - clean).cpu().numpy() < self.tol or num_calls_search > 100: break      
        return adv, num_calls + num_calls_search

    # ========================== METHOD: RPGO ==========================
    def go_to_boundary_RPGO(self, x_s, eta_o, x_b):
        num_calls = 0
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (x_b - x_s) / (torch.norm(x_b - x_s) + self.eps)
        
        ortho_eta = eta - torch.dot(eta.reshape(-1), v.reshape(-1)) * v
        ortho_eta = ortho_eta / (torch.norm(ortho_eta) + self.eps)
        
        dot_val = torch.clamp(torch.dot(eta.reshape(-1), v.reshape(-1)), -1.0 + 1e-7, 1.0 - 1e-7)
        theta_max = torch.acos(dot_val).item()
        theta_min = 0.0
        
        perturbed = None
        loop_calls = 1
        
        while True:
            theta_mid = (theta_min + theta_max) / 2.0
            zeta = math.cos(theta_mid) * v + math.sin(theta_mid) * ortho_eta
            zeta = zeta / (torch.norm(zeta) + self.eps)
            
            r_new = torch.norm(x_b - x_s) * torch.clamp(torch.dot(zeta.reshape(-1), v.reshape(-1)), 0.0, 1.0)
            p_cand = x_s + zeta * r_new 
            
            num_calls += 1
            if self.is_adversarial(p_cand) == 1:
                theta_min, perturbed = theta_mid, p_cand
            else:
                theta_max = theta_mid
                
            loop_calls += 1
            if theta_max - theta_min < 1e-3 or loop_calls >= 40:
                if perturbed is None: perturbed = x_b  
                break
        
        if not torch.equal(perturbed, x_b):
            perturbed, bin_query = self.bin_search(self.src_img, perturbed)
            num_calls += bin_query
            
        return perturbed, num_calls

    # ========================== METHOD: LRCA v2 ==========================
    def go_to_boundary_LRCA(self, x_s, x_b, lr=0.1):
        num_calls = 0
        z = x_b.clone().detach().requires_grad_(True)
        delta = self.model.decoder(z) - self.model.x_rec_0
        chord_sq = (delta ** 2).sum()
        chord_sq.backward()
        g_nat = z.grad.detach() 
        
        g_norm = torch.norm(g_nat) + self.eps
        d = -g_nat / g_norm
        
        r_cur = torch.norm(x_b - x_s).item()
        base_step = lr * r_cur
        
        best_adv = None
        best_l2 = r_cur
        last_nonadv = None
        
        for scale in [1.0, 0.5, 0.25, 0.1]:
            z_cand = x_b + scale * base_step * d
            num_calls += 1
            if self.is_adversarial(z_cand) == 1:
                z_ref, bq = self.bin_search(self.src_img, z_cand)
                num_calls += bq
                l2_ref = torch.norm(z_ref - x_s).item()
                if l2_ref < best_l2:
                    best_l2 = l2_ref
                    best_adv = z_ref
                break
            else:
                last_nonadv = z_cand
                
        if best_adv is not None:
            return best_adv, num_calls
            
        if last_nonadv is not None:
            z_final, bq = self.bin_search_between(x_b, last_nonadv)
            num_calls += bq
            if torch.norm(z_final - x_s).item() < r_cur:
                return z_final, num_calls
                
        return x_b, num_calls

    # ========================== METHOD: NGQS ==========================
    def go_to_boundary_NGQS(self, x_s, x_b):
        num_calls = 0
        
        z = x_b.clone().detach().requires_grad_(True)
        delta = self.model.decoder(z) - self.model.x_rec_0
        chord_sq = (delta ** 2).sum()
        chord_sq.backward()
        grad = z.grad.detach()
        
        N_small = 10
        random_vecs = torch.randn(N_small, *x_b.shape[1:]).to(self.device)
        eta = self.normal_vector_approximation_batch(x_b, N_small, random_vecs, self.sigma)
        num_calls += N_small
        
        eta = eta / (torch.norm(eta) + self.eps)
        v = (x_b - x_s) / (torch.norm(x_b - x_s) + self.eps)
        
        eta_perp = eta - torch.dot(eta.reshape(-1), v.reshape(-1)) * v
        eta_perp = eta_perp / (torch.norm(eta_perp) + self.eps)
        
        g_v = torch.dot(grad.reshape(-1), v.reshape(-1)).item()
        g_eta = torch.dot(grad.reshape(-1), eta_perp.reshape(-1)).item()
        
        phi_opt = math.atan2(-g_eta, -g_v)
        if phi_opt < 0: phi_opt += math.pi
        if phi_opt > math.pi/2: phi_opt = math.pi - phi_opt
        phi_opt = max(0.0, min(math.pi/2, phi_opt))
        
        # FIX: Using math.cos and math.sin for python float scalar phi_opt to avoid TypeError
        zeta = math.cos(phi_opt) * v + math.sin(phi_opt) * eta_perp
        zeta = zeta / (torch.norm(zeta) + self.eps)
        
        R = torch.norm(x_b - x_s).item()
        lo, hi = 0.0, R
        cand = None
        
        for _ in range(12):
            mid = (lo + hi) / 2.0
            pt = x_s + zeta * mid
            num_calls += 1
            if self.is_adversarial(pt) == 1:
                cand = pt
                hi = mid
            else:
                lo = mid
            if hi - lo < self.tol:
                break
                
        if cand is not None:
            cand, bq = self.bin_search(self.src_img, cand)
            num_calls += bq
            if torch.norm(cand - x_s).item() < R:
                return cand, num_calls
                
        return x_b, num_calls

    # ===================== RAG-PBTR HELPERS =====================
    def _remaining_queries(self):
        return max(0, int(self.max_queries - self.all_queries))

    def _rag_chord_sq(self, z):
        """Unclipped RAG chord energy ||G(z)-G(z_s)||_2^2 (no victim query)."""
        if self._rag_base_rec is None:
            raise RuntimeError("RAG-PBTR requires model.decoder or model.x_rec_0.")
        with torch.no_grad():
            delta = self.model.decoder(z) - self._rag_base_rec
            return float((delta * delta).sum().item())

    def _rag_chord_grad(self, z):
        """g = grad_z 0.5||G(z)-G(z_s)||^2 using only the attacker's decoder."""
        if self._rag_base_rec is None:
            raise RuntimeError("RAG-PBTR requires model.decoder or model.x_rec_0.")
        with torch.enable_grad():
            z_req = z.detach().clone().requires_grad_(True)
            delta = self.model.decoder(z_req) - self._rag_base_rec
            loss = 0.5 * (delta * delta).sum()
            g = torch.autograd.grad(loss, z_req, create_graph=False, retain_graph=False)[0]
        return g.detach()

    def _decoder_jtj_vec(self, z, vec):
        """Matrix-free (J_G^T J_G + lambda I) @ vec.

        This differentiates only through the known autoencoder decoder. It does
        not access victim logits/gradients and therefore preserves the
        decision-based black-box threat model.
        """
        z0 = z.detach().clone().requires_grad_(True)
        v = vec.detach()
        with torch.enable_grad():
            # Jv via forward-mode AD.
            _, jv = torch.autograd.functional.jvp(
                self.model.decoder, (z0,), (v,), create_graph=False, strict=False
            )
            # J^T(Jv) via reverse-mode AD.
            y = self.model.decoder(z0)
            jtjv = torch.autograd.grad(
                y, z0, grad_outputs=jv.detach(), create_graph=False, retain_graph=False
            )[0]
        return jtjv.detach() + self.pbtr_damping * v

    def _cg_solve_pullback(self, z, b):
        """Approximately solve (J^T J + lambda I)x=b with conjugate gradient."""
        x = torch.zeros_like(b)
        r = b.detach().clone()
        p = r.clone()
        rs_old = torch.sum(r * r)
        bnorm = torch.sqrt(rs_old).item()
        if bnorm <= self.eps:
            return x

        for _ in range(max(1, self.pbtr_cg_iters)):
            Ap = self._decoder_jtj_vec(z, p)
            denom = torch.sum(p * Ap)
            if (not torch.isfinite(denom)) or abs(float(denom.item())) <= self.eps:
                break
            alpha = rs_old / (denom + self.eps)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r)
            if not torch.isfinite(rs_new):
                break
            if torch.sqrt(rs_new).item() <= 1e-3 * max(bnorm, self.eps):
                break
            beta = rs_new / (rs_old + self.eps)
            p = r + beta * p
            rs_old = rs_new
        return x.detach()

    def _pullback_norm(self, z, p):
        Mp = self._decoder_jtj_vec(z, p)
        val = torch.sum(p * Mp)
        return torch.sqrt(torch.clamp(val, min=self.eps))

    def _fixed_binary_refine(self, x_adv, x_nonadv, max_steps=None):
        """Query-capped binary refinement between known adversarial/clean endpoints."""
        steps = self.pbtr_refine_steps if max_steps is None else int(max_steps)
        steps = min(steps, self._remaining_queries())
        adv = x_adv.detach().clone()
        clean = x_nonadv.detach().clone()
        calls = 0
        for _ in range(max(0, steps)):
            mid = 0.5 * (adv + clean)
            calls += 1
            if self.is_adversarial(mid) == 1:
                adv = mid
            else:
                clean = mid
            if torch.norm(adv - clean).item() < self.tol:
                break
        return adv, calls

    def _orthogonal_gaussian_noises(self, q, latent_shape):
        """Generate approximately orthonormal Gaussian probes in latent space."""
        q = int(q)
        vecs = torch.randn(q, int(np.prod(latent_shape)), device=self.device)
        if not self.pbtr_use_orthogonal_probes or q <= 1:
            return vecs.reshape(q, *latent_shape)

        basis = []
        for i in range(q):
            v = vecs[i]
            for b in basis:
                v = v - torch.dot(v, b) * b
            n = torch.norm(v)
            if n.item() < 1e-8:
                v = torch.randn_like(v)
                n = torch.norm(v)
            v = v / (n + self.eps)
            basis.append(v)
        # Preserve the expected norm of an N(0,I) probe: E||u|| ~= sqrt(D).
        # Pure unit-norm orthogonal vectors would make sigma effectively too small.
        scale = math.sqrt(float(vecs.shape[1]))
        return (torch.stack(basis, dim=0) * scale).reshape(q, *latent_shape)

    def _pbtr_low_query_normal(self, x_b, sigma_t):
        """Small-query normal estimate + history prior for RAG_PBTR_LQ.

        This is an optional query-saving variant. RAG_PBTR itself uses the same
        normal-estimation budget as LGC for a cleaner optimizer-only ablation.
        """
        q = min(self.pbtr_normal_queries, self._remaining_queries())
        if q <= 0:
            return self._pbtr_prev_normal, 0

        noises = self._orthogonal_gaussian_noises(q, tuple(x_b.shape[1:]))
        fresh = self.normal_vector_approximation_batch(x_b, q, noises, sigma_t)
        fresh = fresh / (torch.norm(fresh) + self.eps)

        if self._pbtr_prev_normal is not None:
            prev = self._pbtr_prev_normal
            # Resolve the arbitrary sign before averaging.
            if torch.sum(fresh * prev).item() < 0:
                fresh = -fresh
            beta = min(max(self.pbtr_history_beta, 0.0), 0.99)
            eta = beta * prev + (1.0 - beta) * fresh
            eta = eta / (torch.norm(eta) + self.eps)
        else:
            eta = fresh

        self._pbtr_prev_normal = eta.detach()
        return eta.detach(), q

    # ================= METHOD: RAG-aware Pullback Boundary Trust Region =================
    def go_to_boundary_RAG_PBTR(self, x_s, eta_o, x_b):
        """RAG-aware pullback-metric latent boundary trust-region step.

        Local model:
            f(z) = 0.5 ||G(z)-G(z_s)||^2
            M(z) = J_G(z)^T J_G(z) + lambda I

        We compute the metric-projected natural descent direction
            p0 = -M^{-1}g + M^{-1}eta * (eta^T M^{-1}g)/(eta^T M^{-1}eta),
        which satisfies eta^T p0 ~= 0, then scale p0 to a trust region in
        the pullback metric. Only hard-label victim queries are used to restore
        the decision boundary; all Jacobian/gradient operations involve the
        attacker's decoder only.
        """
        if eta_o is None:
            return x_b, 0
        if self._remaining_queries() <= 0:
            return x_b, 0

        eta = eta_o.detach()
        eta = eta / (torch.norm(eta) + self.eps)
        current_obj = self._rag_chord_sq(x_b)
        g = self._rag_chord_grad(x_b)

        # Natural-gradient and metric-normal solves without explicitly forming J^T J.
        Minv_g = self._cg_solve_pullback(x_b, g)
        Minv_eta = self._cg_solve_pullback(x_b, eta)

        denom = torch.sum(eta * Minv_eta)
        if (not torch.isfinite(denom)) or abs(float(denom.item())) < 1e-10:
            # Euclidean tangent fallback.
            p = -g + torch.sum(g * eta) * eta
        else:
            coeff = torch.sum(eta * Minv_g) / (denom + self.eps)
            p = -Minv_g + coeff * Minv_eta

        if (not torch.isfinite(p).all()) or torch.norm(p).item() < 1e-10:
            p = -g + torch.sum(g * eta) * eta
        if torch.norm(p).item() < 1e-10:
            self.pbtr_trust_scale = max(self.pbtr_trust_min, self.pbtr_trust_scale * self.pbtr_shrink)
            return x_b, 0

        # Ensure first-order RAG descent.
        if torch.sum(g * p).item() > 0:
            p = -p

        # Trust-region radius is measured in the decoder-induced metric and
        # scaled relative to the current RAG chord magnitude.
        chord_norm = math.sqrt(max(current_obj, self.eps))
        target_metric_step = max(1e-4, self.pbtr_trust_scale * chord_norm)
        pnorm_M = self._pullback_norm(x_b, p).item()
        if (not math.isfinite(pnorm_M)) or pnorm_M < 1e-10:
            return x_b, 0
        p = p * (target_metric_step / (pnorm_M + self.eps))

        calls = 0
        best = x_b
        best_obj = current_obj
        accepted_scale = 0.0

        # Backtracking is zero-query until a decoder-improving trial is found.
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial = x_b + scale * p
            trial_obj = self._rag_chord_sq(trial)
            if trial_obj >= current_obj * (1.0 - 1e-7):
                continue
            if self._remaining_queries() <= 0:
                break

            calls += 1
            status = self.is_adversarial(trial)

            if status == 1:
                # Source is clean; trial is adversarial. Coarsely refine only
                # the promising ray instead of performing an expensive exact search.
                cand, qref = self._fixed_binary_refine(trial, x_s)
            else:
                # Current boundary is adversarial; trial is clean. Restore the
                # nearby boundary along the trust-region step.
                cand, qref = self._fixed_binary_refine(x_b, trial)
            calls += qref

            cand_obj = self._rag_chord_sq(cand)
            if cand_obj < best_obj * (1.0 - 1e-7):
                best = cand.detach()
                best_obj = cand_obj
                accepted_scale = scale
                break

        if accepted_scale > 0.0 and best_obj < current_obj:
            step = best - x_b
            predicted = max(float((-torch.sum(g * step)).item()), self.eps)
            actual = max(current_obj - best_obj, 0.0)
            rho = actual / predicted

            if rho > 0.75:
                self.pbtr_trust_scale = min(self.pbtr_trust_max, self.pbtr_trust_scale * self.pbtr_expand)
            elif rho < 0.25:
                self.pbtr_trust_scale = max(self.pbtr_trust_min, self.pbtr_trust_scale * self.pbtr_shrink)

            self.pbtr_diagnostics.append({
                'accepted': True,
                'queries': calls,
                'rag_obj_before': current_obj,
                'rag_obj_after': best_obj,
                'rho': float(rho),
                'trust_scale': float(self.pbtr_trust_scale),
                'trial_scale': float(accepted_scale),
                'normal_tangent_dot': float(torch.sum(eta * step).item()),
            })
            return best, calls

        self.pbtr_trust_scale = max(self.pbtr_trust_min, self.pbtr_trust_scale * self.pbtr_shrink)
        self.pbtr_diagnostics.append({
            'accepted': False,
            'queries': calls,
            'rag_obj_before': current_obj,
            'rag_obj_after': current_obj,
            'rho': 0.0,
            'trust_scale': float(self.pbtr_trust_scale),
            'trial_scale': 0.0,
            'normal_tangent_dot': 0.0,
        })
        return x_b, calls

    def _record_metrics(self, z_adv, q_num, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list, phase_name):
        with torch.no_grad():
            norm_l2_lat = torch.norm(z_adv - self.src_img, p=2).item()
            
            x_adv_pix = self.model.get_pixel_image(z_adv)
            x_orig_pix = self.model.x_orig
            
            norm_l2_pix = torch.norm(x_adv_pix - x_orig_pix, p=2).item()
            
            ssim_val = self.ssim_fn(x_adv_pix, x_orig_pix, data_range=1.0).item()
            x_adv_lpips = x_adv_pix * 2.0 - 1.0   
            x_orig_lpips = x_orig_pix * 2.0 - 1.0 
            lpips_val = self.loss_fn_vgg(x_adv_lpips, x_orig_lpips).item()
            
            mean_tensor = torch.tensor(self.mean).view(1,3,1,1).to(self.device)
            std_tensor = torch.tensor(self.std).view(1,3,1,1).to(self.device)
            adv_label = self.model.classifier((x_adv_pix - mean_tensor) / std_tensor).argmax(1).item()
            
            is_adv = 1 if (adv_label == self.tar_lbl if self.tar_img is not None else adv_label != self.src_lbl) else 0
            
            l2_lat_list.append(norm_l2_lat)
            linf_lat_list.append(0.0)
            l2_pix_list.append(norm_l2_pix)
            linf_pix_list.append(0.0)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)
            asr_list.append(is_adv)
            
            if self.verbose_control == 'Yes':
                print(f"      -> [{phase_name:<8}] Queries: {q_num:<5} | Pix L2: {norm_l2_pix:.3f} | Lat L2: {norm_l2_lat:.3f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | ASR: {is_adv}")

    def _save_visual_milestone(self, z_adv, m, actual_q):
        with torch.no_grad():
            x_adv_pix = self.model.get_pixel_image(z_adv)
            x_orig_pix = self.model.x_orig
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
        axes[1].set_title(f"Adversarial\nQ={actual_q}", fontsize=12)
        axes[1].axis('off')
        
        noise = (x_adv_pix - x_orig_pix).squeeze(0).cpu()
        noise_vis = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        axes[2].imshow(noise_vis.permute(1, 2, 0).numpy())
        axes[2].set_title("Normalized Perturbation", fontsize=12)
        axes[2].axis('off')
        
        metrics_text = (f"Budget (<= {m}): {actual_q} Queries | L2 (Pix): {norm_l2_pix:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}")
        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = os.path.join(self.save_img_dir, f"{self.img_name}_{self.attack_method}_Q{m}_comparison.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        self.saved_milestones[m] = True 
        if self.verbose_control == 'Yes':
            print(f"      -> [*] Exact Visual Milestone Saved: Q{m} (State from Q={actual_q})")

    def Attack(self):
        n_query, l2_lat_list, linf_lat_list = [], [], []
        l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list = [], [], [], [], []

        if self.tar_img is not None:
            x_random, query_random = self.tar_img, 0
        else:
            x_random, query_random = self.find_random_adversarial(self.src_img)

        x_b, query_b = self.bin_search(self.src_img, x_random)
        # all_queries is the authoritative attack-query counter. Metric-only
        # evaluation queries in _record_metrics are intentionally not charged.
        q_num = self.all_queries
        n_query.append(q_num)

        self._record_metrics(x_b, q_num, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list, "Initial")
        size = self.src_img.shape

        for i in range(self.iteration):
            if self.all_queries >= self.max_queries:
                break

            prev_x_b = x_b.clone()
            prev_q = self.all_queries
            grad_oi = None

            if self.attack_method in ['LRCA', 'NGQS']:
                # These legacy methods estimate their own search information.
                pass
            elif self.attack_method == 'RAG_PBTR_LQ':
                # Query-saving ablation: small orthogonal probe set + history prior.
                current_dist = torch.norm(x_b - self.src_img).item()
                sigma_t = max(self.sigma * min(1.0, current_dist), 1e-5)
                grad_oi, _ = self._pbtr_low_query_normal(x_b, sigma_t)
                if grad_oi is None:
                    break
            else:
                # Fair optimizer comparison: RAG_PBTR uses exactly the same
                # normal-estimation schedule as LGC unless the _LQ variant is selected.
                q_opt = int(self.N0 * np.sqrt(i + 1))
                q_opt = max(q_opt, 10)
                q_opt = min(q_opt, self._remaining_queries())
                if q_opt <= 0:
                    break

                random_vec_o = torch.randn(q_opt, size[1], size[-2], size[-1]).to(self.device)
                if self.attack_method in ['RPGO', 'RAG_PBTR']:
                    current_dist = torch.norm(x_b - self.src_img).item()
                    sigma_t = max(self.sigma * min(1.0, current_dist), 1e-5)
                else:
                    sigma_t = self.sigma
                grad_oi = self.normal_vector_approximation_batch(x_b, q_opt, random_vec_o, sigma_t)

            if self.all_queries >= self.max_queries:
                q_num = self.all_queries
                break

            if self.attack_method == 'LGC':
                x_adv, _ = self.go_to_boundary_LGC(self.src_img, grad_oi, x_b)
            elif self.attack_method == 'RPGO':
                x_adv, _ = self.go_to_boundary_RPGO(self.src_img, grad_oi, x_b)
            elif self.attack_method == 'LRCA':
                x_adv, _ = self.go_to_boundary_LRCA(self.src_img, x_b, lr=0.1)
            elif self.attack_method == 'NGQS':
                x_adv, _ = self.go_to_boundary_NGQS(self.src_img, x_b)
            elif self.attack_method in ['RAG_PBTR', 'RAG_PBTR_LQ']:
                x_adv, _ = self.go_to_boundary_RAG_PBTR(self.src_img, grad_oi, x_b)
            else:
                raise ValueError(
                    f"Unknown attack_method={self.attack_method}. Supported: "
                    "LGC, RPGO, LRCA, NGQS, RAG_PBTR, RAG_PBTR_LQ"
                )

            x_b = x_adv
            q_num = self.all_queries

            if self.save_img_dir:
                for m in self.milestones:
                    if q_num >= m and not self.saved_milestones[m]:
                        # State immediately before crossing the requested budget.
                        self._save_visual_milestone(prev_x_b, m, prev_q)

            self._record_metrics(x_b, q_num, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list, f"Iter {i}")
            n_query.append(q_num)

            if q_num >= self.max_queries:
                break

        if self.save_img_dir:
            for m in self.milestones:
                if q_num >= m and not self.saved_milestones[m]:
                    self._save_visual_milestone(x_b, m, q_num)

        return x_b, n_query, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list, self.pbtr_diagnostics
