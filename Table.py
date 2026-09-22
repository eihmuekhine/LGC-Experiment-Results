import numpy as np
import pandas as pd
import os

# Results များကို အလိုအလျောက် သိမ်းဆည်းထားသော Folder လမ်းကြောင်းအမှန်
base_dir = "Results"
checkpoints = [500, 1000, 2000, 5000, 10000]
methods = ["RAG_TTBA_BASE", "RAG_TTBA_SLERP_OPTIMIZED"]

results = {}
for method in methods:
    # metrics_numeric.npz ကို တိုက်ရိုက်ခေါ်ဖတ်ပါမည်
    npz_path = os.path.join(base_dir, "ImageNet_ResNet50_TtBA_SLERP_Opt_NonTargeted.npz")
    
    if os.path.exists(npz_path):
        # ValueError မတက်စေရန် allow_pickle=True ဖြင့် ဖတ်ပါသည်
        data = np.load(npz_path, allow_pickle=True)
        
        # l2_pix သည် Shape (200, 5) ရှိသဖြင့် ပုံ ၂၀၀ စလုံးအတွက် Median (axis=0) ကို ရှာပါသည်
        median_l2 = np.nanmedian(data['l2_pix'], axis=0)
        results[method] = median_l2
    else:
        print(f"File not found: {npz_path}")

if results:
    df = pd.DataFrame(results, index=[f"Q <= {q}" for q in checkpoints])
    print("\n=== Median Pixel L2 for 200 Images ===")
    print(df.to_markdown())