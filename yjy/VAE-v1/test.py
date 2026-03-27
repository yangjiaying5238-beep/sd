import os, torch
from diffusers import StableDiffusionPipeline
 
# ===== 路径配置 =====
base_model_path = "/root/autodl-tmp/yjy/VAE-v1/sd15_ir_fulltune/best_pipeline"
lora_path       =  "/root/autodl-tmp/yjy/VAE-v1/sd15_ir_r32_v4"
output_dir      = "/root/autodl-tmp/yjy/VAE-v1/infrared_test_v5"
os.makedirs(output_dir, exist_ok=True)
 
# ===== 推理参数 =====
num_inference_steps = 50
guidance_scale      = 7.5
seed                = 42
device              = "cuda"
 
# ===== 加载模型 =====
print("正在加载模型...")
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)
 
# 加载LoRA权重
pipe.load_lora_weights(lora_path)
print("✓ 模型加载完成")
 
# ===== 生成测试图像 =====
test_prompts = [
    # 短caption风格
    "Empty street with parked bikes and buildings",
    "People walking on wet street",
    "Palm trees, street, people, bus, traffic light",
    "SUV parked on street with trees",
    "People walking, bicycles, street scene",
    
    # 长caption风格
    "A street with parked bicycles on the right side, a row of buildings on both sides, and a streetlight in the center",
    "A street scene with a bus on the left, a pedestrian crossing in the foreground, and two people standing on the right side of the street",
    "A person riding a bicycle on the left, a person walking on the right, and a group of people standing in the background",
    "A street scene with palm trees lining the sidewalk, a bus on the left, a pedestrian crossing the street",
    "The image shows a scene with a white car in the foreground, a pedestrian crossing the street, and another car in the background, all under the illumination of traffic lights",
]
 
for i, prompt in enumerate(test_prompts):
    print(f"生成第 {i+1} 张: '{prompt}'")
    generator = torch.Generator(device=device).manual_seed(seed + i)
 
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
 
    save_path = os.path.join(output_dir, f"generated_{i+1}.png")
    image.save(save_path)
    print(f"✓ 已保存: {save_path}")
 
print(f"\n完成！结果保存在: {output_dir}")