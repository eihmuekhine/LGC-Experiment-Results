from openai import OpenAI
import os

# ၁။ သင့် Paper ၏ LaTeX ဖိုင်ကို ဖတ်ခြင်း
paper_file_path = "main.tex" 

try:
    with open(paper_file_path, "r", encoding="utf-8") as file:
        paper_content = file.read()
except FileNotFoundError:
    print(f"Error: {paper_file_path} ကို ရှာမတွေ့ပါ။")
    exit()

# ၂။ DeepSeek API သို့ ချိတ်ဆက်ခြင်း
client = OpenAI(
    api_key="sk-3b565018a53c4ae49367a5c478f1363b", # သင့် API Key ပြောင်းထည့်ရန်
    base_url="https://api.deepseek.com"
)

# ၃။ DeepSeek အတွက် အထူးပြုလုပ်ထားသော Prompt
system_prompt = """
You are a world-class researcher and top-tier reviewer in Computer Vision, Deep Learning, and Adversarial Attacks. 
Your expertise lies in hard-label decision-based black-box attacks, latent space manifold optimization, and generative models. 
Please provide your response in a mix of Myanmar language and English technical terms.
"""

user_prompt = f"""
I am revising my manuscript titled "Latent Geometric Chords for Query-Efficient Decision-Based Adversarial Attacks" which was recently rejected by IEEE T-IFS. 

Here is the LaTeX source code of my current manuscript for your deep understanding:
<paper_content>
{paper_content}
</paper_content>

Here are the critical comments from the T-IFS reviewer:
<reviewer_comments>
The paper is rejected due to its limited novelty. The experimental comparison is not adequate for a 2026 submission. The main compared methods are HSJA (2020), Sign-OPT (2020), and CGBA (2023). This is too narrow and outdated. The literature coverage also lacks recent 2024–2025/2026 works on decision-based, generative, diffusion-based, and semantic/latent-space attacks. The paper appears to be a limited extension of existing work.
</reviewer_comments>

My core technical problem:
My method successfully introduces the Residual-based Adversarial Generation (RAG) mechanism, but for the latent geometric optimization, it heavily relies on **CGBA** (Curvature-aware Geometric Black-box Attack). 
1. The reviewers perceive the use of CGBA as "limited innovation" (just applying an existing pixel-space attack to latent space).
2. In my experiments, this CGBA-based estimator struggles and stalls in regimes where the L2 norm is very low (the perturbation is minimal), likely due to high variance in normal vector estimation at highly curved boundaries.

Based on my exact paper content and the reviewer's comments, please provide:
1. **Critique:** A rigorous mathematical explanation of WHY applying CGBA in the latent space causes optimization issues in low L2 norm regimes.
2. **Novel Optimization Replacements:** Propose 3 entirely NOVEL mathematical mechanisms to **REPLACE CGBA** for latent geometric optimization in hard-label decision-based attacks. These should synergize with my RAG mechanism, solve the low L2 norm stalling issue, and provide strong theoretical novelty (e.g., Riemannian geometry, Score-guided diffusion priors, or Bayesian manifold optimization).
3. **Mathematical Formulation:** For the absolute best alternative among the 3, provide the detailed mathematical formulation (in LaTeX equations) and the step-by-step theoretical justification.
4. **SOTA Baselines:** Provide a detailed list of 5 recent SOTA baselines (published in 2024-2026 at CVPR/ICCV/ECCV/NeurIPS/ICLR) targeting hard-label, semantic, or diffusion-based adversarial attacks that I MUST compare against.
"""

print("DeepSeek သို့ Paper အချက်အလက်များ ပေးပို့နေပါသည်... မိနစ်အနည်းငယ် စောင့်ဆိုင်းပေးပါ။")

response = client.chat.completions.create(
    model="deepseek-reasoner", # Thinking Mode ကို မဖြစ်မနေသုံးပါ။
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

print("\n--- DeepSeek ၏ သုတေသန အကြံပြုချက် ---\n")
print(response.choices[0].message.content)