import argparse
import random
import logging
import math
import os
import json
import shutil
 
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
 
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
 
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
 
from transformers import CLIPTextModel, CLIPTokenizer
 
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_unet_state_dict_to_peft, convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from peft import LoraConfig
 
logger = get_logger(__name__, log_level="INFO")
 
 
# 1. 命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="红外图像SD1.5全参数微调")
 
    # 路径
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                         default="/root/autodl-tmp/yjy/VAE-v1/sd15_ir_fulltune/best_pipeline")
    parser.add_argument("--train_data_dir", type=str,
                        default="/root/autodl-tmp/dataset/images_clean")
        #清洗后的caption
    parser.add_argument("--caption_json", type=str,
                         default="/root/autodl-tmp/dataset/captions_final_v5.json")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/yjy/VAE-v1/sd15_ir_r32_v4")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
 
    # 训练超参
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
 
    # 优化器
    parser.add_argument("--use_8bit_adam", action="store_true", default=True)
 
    # 保存与日志
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])

    parser.add_argument("--lora_r", type=int, default=32, help="Lora rank, only used if use_lora is True")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha, only used if lora is True")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Lora dropout, only used if use_lora is True")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora is True",
    )
 
    return parser.parse_args()
 
 
# 2. 数据集

class InfraredDataset(Dataset):
    def __init__(self, data_dir, caption_json, tokenizer, resolution,
                 uncond_prob=0.1):  # uncond_prob=随机丢掉caption的概率
        with open(caption_json, 'r', encoding='utf-8') as f:
            captions = json.load(f)

        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        self.items = [
            (os.path.join(data_dir, f), captions[f])
            for f in sorted(os.listdir(data_dir))
            if f.lower().endswith(valid_ext) and f in captions
        ]
        self.uncond_prob = uncond_prob  # 新增
        self.tokenizer = tokenizer
        
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomCrop(resolution),     
            transforms.RandomHorizontalFlip(),      
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
 
    def __len__(self):
        return len(self.items)
 
    def __getitem__(self, idx):
        img_path, data = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        if random.random() < self.uncond_prob:
            # 无条件训练：空字符串
            caption = ""
        else:
            # 随机选长caption或短caption，不拼quality
            if random.random() < 0.5:
                caption = data["caption_long"]
            else:
                caption = data["caption_short"]

        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel_values, "input_ids": input_ids}

 

# 3. 主训练函数
def main():
    args = parse_args()
 
    # Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
    )
 
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
 
    if args.seed is not None:
        set_seed(args.seed)
 
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
 
    # 加载模型
    #噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    #文字分词器
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")
    #文字编码器
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet")

 
    #  冻结
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    UNET_TARGET_MODULES = [
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
        "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
    ]

    # UNET_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]

    unet_lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=UNET_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
 
    unet.enable_gradient_checkpointing()
 
    #优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("使用 AdamW8bit")
        except ImportError:
            raise ImportError("请安装: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW
 
    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
 
    # 数据集
    train_dataset = InfraredDataset(
        data_dir=args.train_data_dir,
        caption_json=args.caption_json,
        tokenizer=tokenizer,
        resolution=args.resolution,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
    )
 
    # 学习率调度
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
 
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
 
    # Accelerate prepare
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
 
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
 
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
 
    # 打印训练信息
    total_batch_size = (args.train_batch_size
                        * accelerator.num_processes
                        * args.gradient_accumulation_steps)
    logger.info("***** 开始训练 *****")
    logger.info(f"  数据集大小           = {len(train_dataset)}")
    logger.info(f"  Epochs              = {args.num_train_epochs}")
    logger.info(f"  单卡 batch size     = {args.train_batch_size}")
    logger.info(f"  梯度累积步数         = {args.gradient_accumulation_steps}")
    logger.info(f"  等效总 batch size   = {total_batch_size}")
    logger.info(f"  每epoch更新步数      = {num_update_steps_per_epoch}")
    logger.info(f"  总更新步数           = {max_train_steps}")
    logger.info(f"  学习率              = {args.learning_rate}")


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def cast_training_params(model: torch.nn.Module | list[torch.nn.Module], dtype=torch.float32):
        """
        Casts the training parameters of the model to the specified data type.

        Args:
            model: The PyTorch model whose parameters will be cast.
            dtype: The data type to which the model parameters will be cast.
        """
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for param in m.parameters():
                # only upcast trainable parameters into fp32
                if param.requires_grad:
                    param.data = param.to(dtype)

    #加载钩子
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionPipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                safe_serialization=True,
            )

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        # returns a tuple of state dictionary and network alphas
        lora_state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(input_dir)

        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            # throw warning if some unexpected keys are found and continue loading
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32
        if args.mixed_precision in ["fp16"]:
            cast_training_params([unet_], dtype=torch.float32)
 
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
 
    # 断点续训
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(args.output_dir)
                    if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        else:
            path = os.path.basename(args.resume_from_checkpoint)
 
        if path is None:
            logger.info("没有找到checkpoint，从头训练")
        else:
            logger.info(f"从checkpoint恢复: {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
 
    # 训练循环
    best_loss = float('inf')
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_main_process,
        desc="训练步数",
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                # VAE编码
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 加噪
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # 文本编码
                encoder_hidden_states = text_encoder(
                    batch["input_ids"], return_dict=False)[0]

                # 预测目标
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"未知 prediction_type: "
                        f"{noise_scheduler.config.prediction_type}")

                # UNet前向 + 损失
                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean")

                # 多卡loss收集
                avg_loss = accelerator.gather(
                    loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    # 计算当前epoch内的步数
                    steps_in_epoch = (global_step - first_epoch * num_update_steps_per_epoch)
                    avg_loss = train_loss / max(steps_in_epoch, 1)
                    logger.info(
                        f"step={global_step} | "
                        f"loss={avg_loss:.4f} | "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )


                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # 删除旧checkpoint
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for removing_checkpoint in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                                    logger.info(f"删除旧checkpoint: {removing_checkpoint}")

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Checkpoint保存: {save_path}")

        # 每轮结束保存最佳
        avg_epoch_loss = train_loss / len(train_dataloader)
        if accelerator.is_main_process and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            unwrapped_unet = accelerator.unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            StableDiffusionPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=False,
            )

            logger.info(f"新最佳 Loss={best_loss:.6f}，已保存")
    
 
    # 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=False,
        )
        # pipeline.save_lora_weights(args.output_dir)
        logger.info(f"训练完成！最终模型: {args.output_dir}")
 
    accelerator.end_training()
 
 
if __name__ == "__main__":
    main()