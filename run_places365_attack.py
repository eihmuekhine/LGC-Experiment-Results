import torchvision.transforms as transforms
import torchvision.models as torch_models
import numpy as np
import torch
import os
import csv
from PIL import Image
import argparse
import glob
import random
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from models import builder
from attacks.lgc_attack import LGC_Attack, optimize_target_latent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=992):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_places365_model(model_name, device):
    if model_name == 'resnet50':
        model = torch_models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 365)
        url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    elif model_name == 'densenet161':
        model = torch_models.densenet161(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 365)
        url = 'http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar'
    else:
        raise ValueError(f"Unknown architecture: {model_name}")

    checkpoint = torch.hub.load_state_dict_from_url(url, map_location=device)
    state_dict = checkpoint['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        name = name.replace('norm.1.', 'norm1.')
        name = name.replace('norm.2.', 'norm2.')
        name = name.replace('conv.1.', 'conv1.')
        name = name.replace('conv.2.', 'conv2.')
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    return model.to(device).eval()

class LatentModelWrapper(torch.nn.Module):
    def __init__(self, decoder, classifier, mean, std):
        super().__init__()
        self.decoder = decoder
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def set_reference_images(self, x_orig, x_rec_0):
        self.x_orig = x_orig
        self.x_rec_0 = x_rec_0

    def get_pixel_image(self, z):
        delta = self.decoder(z) - self.x_rec_0
        return torch.clamp(self.x_orig + delta, 0.0, 1.0)

    def forward(self, z):
        x_final = self.get_pixel_image(z)
        return self.classifier((x_final - self.mean) / self.std)

def prepare_strict_splits(args, arch, image_dir, img2label, classifier, mean_t, std_t):
    splits_dir = os.path.join('splits', 'places365')
    os.makedirs(splits_dir, exist_ok=True)
    split_file = os.path.join(splits_dir, f"{arch}_{args.mode}.csv")
    
    pairs = []
    if os.path.exists(split_file):
        print(f"[*] Loading splits from {split_file}")
        with open(split_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                if args.mode == 'targeted':
                    pairs.append((row[0], int(row[1]), row[2], int(row[3])))
                else:
                    pairs.append((row[0], int(row[1]), None, None))
        return pairs[:args.num_samples]

    print(f"[*] Generating exact valid pairs for {arch}...")
    all_files = sorted(glob.glob(os.path.join(image_dir, "*.JPEG")))
    random.shuffle(all_files)
    
    for s_path in all_files:
        if len(pairs) >= args.num_samples: break
        
        s_name = os.path.basename(s_path)
        s_true_lbl = img2label.get(s_name)
        if s_true_lbl is None: continue
            
        x_s = transforms.ToTensor()(transforms.Resize((224, 224))(Image.open(s_path).convert('RGB'))).unsqueeze(0).to(device)
        with torch.no_grad():
            if classifier((x_s - mean_t) / std_t).argmax(1).item() != s_true_lbl: continue
            
        if args.mode == 'non_targeted':
            pairs.append((s_name, s_true_lbl, None, None))
            continue
            
        target_found = False
        candidates = [f for f in all_files if f != s_path]
        random.shuffle(candidates)
        
        for t_path in candidates:
            t_name = os.path.basename(t_path)
            t_true_lbl = img2label.get(t_name)
            if t_true_lbl is None or t_true_lbl == s_true_lbl: continue
            
            x_t = transforms.ToTensor()(transforms.Resize((224, 224))(Image.open(t_path).convert('RGB'))).unsqueeze(0).to(device)
            with torch.no_grad():
                if classifier((x_t - mean_t) / std_t).argmax(1).item() == t_true_lbl:
                    pairs.append((s_name, s_true_lbl, t_name, t_true_lbl))
                    target_found = True
                    break
        if not target_found: continue

    with open(split_file, 'w', newline='') as f:
        writer = csv.writer(f)
        if args.mode == 'targeted':
            writer.writerow(['source_image', 'source_label', 'target_image', 'target_label'])
        else:
            writer.writerow(['source_image', 'source_label'])
        for p in pairs:
            writer.writerow([p[0], p[1], p[2], p[3]] if args.mode == 'targeted' else [p[0], p[1]])
            
    return pairs

def main():
    set_seed(992)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['targeted', 'non_targeted'], required=True)
    parser.add_argument('--target_archs', nargs='+', type=str, default=['resnet50', 'densenet161'])
    parser.add_argument('--attack_methods', nargs='+', type=str, default=['NGQS', 'LRCA', 'RPGO', 'LGC'], help='List of methods to run')
    parser.add_argument('--ae_checkpoint', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='./data/Places365')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()

    image_dir = os.path.join(args.base_dir, 'images')
    labels_txt = os.path.join(args.base_dir, 'adv_test_labels.txt')
    categories_txt = os.path.join(args.base_dir, 'categories_places365.txt')
    
    img2label = {}
    if os.path.exists(labels_txt):
        with open(labels_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2: img2label[parts[0]] = int(parts[1])

    idx2label = {}
    if os.path.exists(categories_txt):
        with open(categories_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2: idx2label[int(parts[1])] = parts[0]

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean_t, std_t = torch.tensor(mean).view(1,3,1,1).to(device), torch.tensor(std).view(1,3,1,1).to(device)

    class MockArgs: arch, parallel, gpu = 'vgg16', 0, 0
    ae_model = builder.BuildAutoEncoder(MockArgs())
    sd = torch.load(args.ae_checkpoint, map_location=device)
    state_dict = sd['state_dict'] if 'state_dict' in sd else sd
    aligned_sd = {('module.' + k.replace('module.', '')): v for k, v in state_dict.items()}
    ae_model.load_state_dict(aligned_sd)
    ae_model = ae_model.to(device).eval()
    
    if isinstance(ae_model, torch.nn.DataParallel):
        decoder_to_use, encoder_to_use = ae_model.module.decoder, ae_model.module.encoder
    else:
        decoder_to_use, encoder_to_use = ae_model.decoder, ae_model.encoder

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    for arch in args.target_archs:
        print(f"\n{'='*60}")
        print(f" STARTING EVALUATION ON CLASSIFIER: {arch.upper()}")
        print(f"={'='*60}")

        classifier = load_places365_model(arch, device)
        model_wrapper = LatentModelWrapper(decoder_to_use, classifier, mean, std).to(device)

        pairs = prepare_strict_splits(args, arch, image_dir, img2label, classifier, mean_t, std_t)
        metrics_dict = {method: {k: [] for k in ['queries', 'l2_lat', 'l2_pix', 'ssim', 'lpips', 'asr']} for method in args.attack_methods}
        attacked_count = 0
        
        for pair in pairs:
            s_name, s_lbl, t_name, t_lbl = pair
            s_path = os.path.join(image_dir, s_name)
            
            x_0_raw = transforms.ToTensor()(transforms.Resize((224, 224))(Image.open(s_path).convert('RGB'))).unsqueeze(0).to(device)
            with torch.no_grad():
                z_source = encoder_to_use(x_0_raw)
                model_wrapper.set_reference_images(x_0_raw, decoder_to_use(z_source))

            z_target = None
            if args.mode == 'targeted':
                t_path = os.path.join(image_dir, t_name)
                x_t_raw = transforms.ToTensor()(transforms.Resize((224, 224))(Image.open(t_path).convert('RGB'))).unsqueeze(0).to(device)
                z_target = optimize_target_latent(encoder_to_use, decoder_to_use, model_wrapper, x_0_raw, model_wrapper.x_rec_0, x_t_raw, t_lbl, mean_t, std_t, device)
                if z_target is None:
                    print(f"[!] Skipping {s_name}: Adam optimization failed")
                    continue

            print(f"\n[{attacked_count+1}/{args.num_samples}] Attack: {s_name} -> target: {t_lbl if t_lbl is not None else 'Any'}")
            attacked_count += 1
            
            for attack_method in args.attack_methods:
                print(f" --- Running Method: {attack_method} ---")
                output_dir = f'Results/{args.mode}_{attack_method}/{arch}'
                vis_dir = os.path.join(output_dir, 'Visuals')
                os.makedirs(vis_dir, exist_ok=True)
                
                attack = LGC_Attack(model_wrapper, z_source, mean, std, loss_fn_vgg, ssim_fn, 
                                    tar_img=z_target, expected_target_label=t_lbl,
                                    attack_method=attack_method, iteration=60, 
                                    save_img_dir=vis_dir, img_name=s_name.split(".")[0],
                                    idx2label=idx2label, query_checkpoints=[500, 1000, 1500, 2000],
                                    max_queries=2000)
                
                z_adv, n_query, l2_lat, _, l2_pix, _, ssim_vals, lpips_vals, asr_vals, _ = attack.Attack()
                
                metrics_dict[attack_method]['queries'].append(n_query)
                metrics_dict[attack_method]['l2_lat'].append(l2_lat)
                metrics_dict[attack_method]['l2_pix'].append(l2_pix)
                metrics_dict[attack_method]['ssim'].append(ssim_vals)
                metrics_dict[attack_method]['lpips'].append(lpips_vals)
                metrics_dict[attack_method]['asr'].append(asr_vals)
                
                print(f"   => {attack_method} Final Status: ASR: {asr_vals[-1]} | Queries: {n_query[-1]} | Pix L2: {l2_pix[-1]:.3f} | SSIM: {ssim_vals[-1]:.4f} | LPIPS: {lpips_vals[-1]:.4f}")

        for attack_method in args.attack_methods:
            output_dir = f'Results/{args.mode}_{attack_method}/{arch}'
            os.makedirs(output_dir, exist_ok=True)
            metrics = metrics_dict[attack_method]
            
            np.savez(os.path.join(output_dir, 'metrics.npz'), **{k: np.array(v, dtype=object) for k, v in metrics.items()})
            
            print("\n" + "="*50)
            print(f" FINAL SUMMARY ({args.mode.upper()} - {attack_method} on {arch.upper()})")
            print("="*50)
            if attacked_count > 0:
                print(f" Total Evaluated   : {attacked_count} images")
                print(f" Overall ASR       : {np.mean([a[-1] for a in metrics['asr']])*100:.2f} %")
                print(f" Median Queries    : {np.median([q[-1] for q in metrics['queries']]):.1f}")
                print(f" Median Pixel L2   : {np.median([l[-1] for l in metrics['l2_pix']]):.3f}")
                print(f" Mean SSIM         : {np.mean([s[-1] for s in metrics['ssim']]):.4f}")
            print("="*50)

        del classifier
        del model_wrapper
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()