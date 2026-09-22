import os
import math
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def optimize_target_latent(encoder, decoder, model_wrapper, x_0_raw, x_rec_0, x_target_raw, target_label, mean_t, std_t, device):
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
            # Major Fix: Strict target label enforcement
            pred_tar_lbl = torch.argmax(self.model(self.tar_img).data).item()
            assert pred_tar_lbl == expected_target_label, f"Silent shift detected! Latent pred: {pred_tar_lbl}, Expected: {expected_target_label}"
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

        self.save_img_dir = save_img_dir
        self.img_name = img_name
        self.milestones = sorted(query_checkpoints) if query_checkpoints else [500, 1000, 2500, 5000, 10000]
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
        num_calls = 0
        adv = x_random
        cln = x_0      
        while True:         
            mid = (cln + adv) / 2.0
            num_calls += 1           
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
            
            dot_val = torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) / ((torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir)) + self.eps)
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

    def _record_metrics(self, z_adv, q_num, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list, phase_name):
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
            
            mean_tensor = torch.tensor(self.mean).view(1,3,1,1).to(self.device)
            std_tensor = torch.tensor(self.std).view(1,3,1,1).to(self.device)
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
        """Saves the combined visual output strictly for the latest valid state <= B"""
        with torch.no_grad():
            x_adv_pix = self.model.get_pixel_image(z_adv)
            x_orig_pix = self.model.x_orig
            
            mean_tensor = torch.tensor(self.mean).view(1,3,1,1).to(self.device)
            std_tensor = torch.tensor(self.std).view(1,3,1,1).to(self.device)
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

    def Attack(self):
        n_query, l2_lat_list, linf_lat_list = [], [], []
        l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list = [], [], [], [], []
        
        if self.tar_img is not None:
            x_random, query_random = self.tar_img, 0
        else:
            x_random, query_random = self.find_random_adversarial(self.src_img)
            
        x_b, query_b = self.bin_search(self.src_img, x_random)
        q_num = query_random + query_b
        n_query.append(q_num)
        
        self._record_metrics(x_b, q_num, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list, "Initial")
        size = self.src_img.shape
        
        for i in range(self.iteration):
            # Track previous state for strict milestone observer (Issue 5)
            prev_x_b = x_b.clone()
            prev_q = q_num

            q_opt = int(self.N0 * np.sqrt(i + 1)) 
            random_vec_o = torch.randn(q_opt, size[1], size[-2], size[-1]).to(self.device)
            
            grad_oi = self.normal_vector_approximation_batch(x_b, q_opt, random_vec_o)
            q_num += q_opt
            
            if self.attack_method == 'LGC':
                x_adv, qs = self.go_to_boundary_LGC(self.src_img, grad_oi, x_b)
            elif self.attack_method == 'LGC_H':
                x_adv, qs = self.go_to_boundary_LGC_H(self.src_img, grad_oi, x_b)
            else:
                x_adv, qs = x_b, 0

            q_num += qs
            x_b = x_adv
            
            # Major Fix: Milestone observer checks if the NEW state overshoots the budget. 
            # If so, it saves the PREVIOUS valid state.
            if self.save_img_dir:
                for m in self.milestones:
                    if q_num >= m and not self.saved_milestones[m]:
                        # Use the latest accepted state where query <= m
                        self._save_visual_milestone(prev_x_b, m, prev_q)

            self._record_metrics(x_b, q_num, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list, f"Iter {i}")
            n_query.append(q_num)
            
            if q_num >= self.max_queries:
                break
        
        # Catch any remaining milestones that fall exactly on the final step
        if self.save_img_dir:
            for m in self.milestones:
                if q_num >= m and not self.saved_milestones[m]:
                    self._save_visual_milestone(x_b, m, q_num)

        return x_b, n_query, l2_lat_list, linf_lat_list, l2_pix_list, linf_pix_list, ssim_list, lpips_list, asr_list, []