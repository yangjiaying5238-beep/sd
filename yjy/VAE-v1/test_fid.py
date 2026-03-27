import os, torch, json
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
 
# ===== 路径配置 =====
base_model_path = "/root/autodl-tmp/yjy/VAE-v1/sd15_ir_fulltune/best_pipeline"
lora_path       = "/root/autodl-tmp/yjy/VAE-v1/sd15_ir_r32_v4"
caption_json    = "/root/autodl-tmp/dataset/captions_final_v5.json"
output_dir      = "/root/autodl-tmp/yjy/VAE-v1/fid_lora_generated_v2"
os.makedirs(output_dir, exist_ok=True)
 
# ===== 推理参数 =====
num_inference_steps = 50
guidance_scale      = 5
device              = "cuda"
total_images        = 50000
 
# ===== 加载模型 =====
print("正在加载模型...")
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)
pipe.load_lora_weights(lora_path)
pipe.set_progress_bar_config(disable=True)
print("✓ 模型加载完成")
 
# ===== 读取caption =====
with open(caption_json, 'r') as f:
    captions = json.load(f)
 
# 把所有caption_short和caption_long都放进来循环使用
all_prompts = []
for fname, data in captions.items():
    all_prompts.append(data["caption_short"])
    all_prompts.append(data["caption_long"])
 
print(f"共有 {len(all_prompts)} 个prompt，循环使用生成 {total_images} 张")
 
# ===== 断点续传 =====
existing = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
print(f"已生成 {existing} 张，继续从第 {existing+1} 张开始")
 
# ===== 生成图像 =====
for i in tqdm(range(existing, total_images), desc="生成图像"):
    prompt = all_prompts[i % len(all_prompts)]
    generator = torch.Generator(device=device).manual_seed(i)
 
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
 
    save_path = os.path.join(output_dir, f"{i:06d}.png")
    image.save(save_path)
 
print(f"\n完成！共生成 {total_images} 张，保存在: {output_dir}")
 