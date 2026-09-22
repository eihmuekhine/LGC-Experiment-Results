from openai import OpenAI

client = OpenAI(api_key="sk-3b565018a53c4ae49367a5c478f1363b", base_url="https://api.deepseek.com")

# Reviewer ၏ မှတ်ချက်များကို ဖြေရှင်းရန် တိကျသော Prompt
prompt_text = """
My current method (Latent Geometric Chords) uses CGBA for gradient estimation in a hard-label decision-based black-box attack, but it struggles with low L2 norm and the reviewers rejected it for limited innovation. 

1. Propose 3 novel alternative mechanisms for geometric optimization in latent space without relying on CGBA.
2. Suggest 5 state-of-the-art (2024-2025) baselines (e.g., Diffusion-based, Text-to-Boundary) to compare against instead of HSJA or Sign-OPT.
"""

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are an expert researcher in Computer Vision, Deep Learning, and Adversarial Attacks (specifically hard-label decision-based attacks). Provide answers in a mix of Myanmar and English technical terms."},
        {"role": "user", "content": prompt_text}
    ]
)

print(response.choices[0].message.content)