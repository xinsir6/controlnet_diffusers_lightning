import os
import random
import torch
from torch import nn
from typing import Any
import pytorch_lightning as pl

import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    ControlNetModel,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import EMAModel
from transformers import AutoTokenizer
from transformers import CLIPTextModel
from transformers import CLIPTextModelWithProjection
import sys
sys.path.append("..")
from annotator.pidinet.model import pidinet
from train_t2i_sdxl_bucket_dataset import ComicDataModule
from train_t2i_sdxl_adaptor import Adapter_XL, ControlNetHED_Apache2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.set_float32_matmul_precision('medium')

# initialize models
def initial_models(pretrained_model_name_or_path, pretrained_vae_model_name_or_path=None, pretrained_pidinet_path=None, revision=None):
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer_2", revision=revision, use_fast=False
    )

    # import correct text encoder classes
    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision
    )

    vae_path = (
        pretrained_model_name_or_path
        if pretrained_vae_model_name_or_path is None
        else pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path, subfolder="vae" if pretrained_vae_model_name_or_path is None else None, revision=revision
    )
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision
    )

    t2i_adaptor = Adapter_XL(channels=[320, 640, 1280, 1280], nums_rb=2, cin=256, ksize=1, sk=True, use_conv=False)

    softedge_net = pidinet()
    softedge_net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(pretrained_pidinet_path)['state_dict'].items()})

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, unet, t2i_adaptor, softedge_net, noise_scheduler

def random_threshold(edge, low_threshold=0.3, high_threshold=0.8):
    threshold = round(random.uniform(low_threshold, high_threshold), 1)
    edge_thre = edge > threshold
    # edge_thre = torch.tensor(edge_thre.clone().detach(), dtype=edge.dtype, device=edge.device)
    edge_thre = edge_thre.to(edge.dtype)
    return edge_thre

class StableDiffusionXL(pl.LightningModule):
    def __init__(self, unet, vae, t2i_adaptor, softedge_net, noise_scheduler, lr, use_ema=False, use_snr=False, snr_gamma=None) -> None:
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.t2i_adaptor = t2i_adaptor
        self.softedge_net = softedge_net
        self.noise_scheduler = noise_scheduler
        self.learning_rate = lr
        self.use_ema = use_ema
        self.use_snr = use_snr
        self.snr_gamma = snr_gamma
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.softedge_net.requires_grad_(False)
        # self.unet.enable_gradient_checkpointing()
        self.unet.enable_xformers_memory_efficient_attention()
        if self.use_ema:
            self.ema_control = EMAModel(self.t2i_adaptor.parameters(), model_cls=Adapter_XL, model_config=None) if use_ema else None

    def compute_time_ids(self, original_size, crops_coords_top_left, target_size):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids]).to(self.device)
        return add_time_ids

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    
    def training_step(self, batch, batch_idx):

        model_input = self.vae.encode(batch['pixel_values']).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor

        edge = self.softedge_net(batch['conditioning_pixel_values'])[-1]
        edge = random_threshold(edge)
        
        noise = torch.randn_like(model_input).to(self.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (model_input.shape[0],)).long().to(self.device)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        add_time_ids = torch.cat(
            [self.compute_time_ids(s, c, t) for s, c, t in zip(batch["original_sizes"], batch["crop_top_lefts"], batch['target_sizes'])]
        ).to(self.device)

        down_block_additional_residuals = self.t2i_adaptor(edge)
 
        model_pred = self.unet(
            noisy_model_input, 
            timesteps, 
            batch["prompt_embeds"], 
            added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": batch["pooled_prompt_embeds"]}, \
            down_block_additional_residuals=down_block_additional_residuals
        ).sample

        if not self.use_snr:
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        else:
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.AdamW(self.t2i_adaptor.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < 500:
            lr_scale = max(min(1.0, float(self.trainer.global_step + 1) / 500.0), 0.01)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

    def on_before_backward(self, loss):
        if self.use_ema:
            if not self.ema_control.cur_decay_value:
                self.ema_control.to(self.device)
            self.ema_control.step(self.t2i_adaptor.parameters())

    def on_train_epoch_end(self):
        if self.use_ema:
            self.ema_control.save_pretrained(self.trainer.checkpoint_callback.dirpath)

if __name__ == "__main__":
    file_txt = '/oss/comicai/qi.xin/wallpaper_1M_good.txt'
    pretrained_model_name_or_path = '/oss/comicai/zhiyuan.shi/models/stable-diffusion-xl-base-1.0'
    pretrained_vae_model_name_or_path = '/oss/comicai/zhiyuan.shi/models/sdxl-vae-fp16-fix'
    pretrained_pidinet_path = '/root/code/diffusers/examples/controlnet/annotator/ckpts/table5_pidinet.pth'
    logger_freq = 4000
    batch_size = 4
    learning_rate = 1e-5
    save_name = 'wallpaper_drawing'

    _, _, _, _, vae, unet, t2i_adaptor, softedge_net, noise_scheduler = initial_models(pretrained_model_name_or_path, pretrained_vae_model_name_or_path, pretrained_pidinet_path)

    ComicData = ComicDataModule(batch_size=batch_size, file_txt=file_txt)
    my_model = StableDiffusionXL(vae=vae, unet=unet, lr=learning_rate, t2i_adaptor=t2i_adaptor, softedge_net=softedge_net, noise_scheduler=noise_scheduler, use_ema=False)

    cur_dir = os.getcwd()
    checkpoint_callback = ModelCheckpoint(dirpath='%s/%s' % (cur_dir, save_name),
                                        monitor='epoch',
                                        save_top_k=20,  # 保存最好的 k 个 checkpoint
                                        every_n_epochs=1,      # 每隔2个 epoch 保存一次 checkpoint
                                        filename="%s-{epoch:02d}-{val_loss:.2f}" % save_name)

    tensorboard = TensorBoardLogger(save_dir='%s/%s' % (cur_dir, save_name))
    trainer = pl.Trainer(profiler='simple', accumulate_grad_batches=8, gradient_clip_val=1.0, max_epochs=20, \
                        logger=tensorboard, devices=4, precision=16, callbacks=[checkpoint_callback], num_nodes=1, strategy='deepspeed_stage_2')#
    # Train!
    trainer.fit(my_model, ComicData)
