import os
from typing import Any
import torch
from torch import nn
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
from train_controlnet_bucket_dataset import ComicDataModule
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# torch.set_float32_matmul_precision('medium')

# initialize models
def initial_models(pretrained_model_name_or_path, pretrained_vae_model_name_or_path=None, controlnet_model_name_or_path=None, \
                   revision=None):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, use_fast=False
    )

    # import correct text encoder classes
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
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
    if controlnet_model_name_or_path:
        controlnet = ControlNetModel.from_pretrained(controlnet_model_name_or_path)
    else:
        controlnet = ControlNetModel.from_unet(unet)

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    return tokenizer, text_encoder, vae, unet, controlnet, noise_scheduler

class StableDiffusionXL(pl.LightningModule):
    def __init__(self, unet, vae, text_encoder, tokenizer, controlnet, noise_scheduler, lr, use_ema=False, use_snr=False, snr_gamma=None) -> None:
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.controlnet = controlnet
        self.noise_scheduler = noise_scheduler
        self.learning_rate = lr
        self.use_ema = use_ema
        self.use_snr = use_snr
        self.snr_gamma = snr_gamma
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # self.tokenizer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.enable_xformers_memory_efficient_attention()
        self.controlnet.enable_gradient_checkpointing()
        self.controlnet.enable_xformers_memory_efficient_attention()
        if self.use_ema:
            self.ema_control = EMAModel(controlnet.parameters(), model_cls=ControlNetModel, model_config=controlnet.config) if use_ema else None

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

        noise = torch.randn_like(model_input).to(self.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (model_input.shape[0],)).long().to(self.device)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        inputs = self.tokenizer(
            batch['captions'], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        encoder_hidden_states = self.text_encoder(inputs.input_ids.to(self.device))[0]

        down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=batch['conditioning_pixel_values'],
                    return_dict=False,
                )
        
        # Predict the noise residual
        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        
        # for stable diffusion 1.5
        target = noise
        # for stable diffusion 2.1 (this is particularly important as you will not converge if you use the same with 1.5)
        # target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        
        if not self.use_snr:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
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
            self.ema_control.step(self.controlnet.parameters())

    def on_train_epoch_end(self):
        if self.use_ema:
            self.ema_control.save_pretrained(self.trainer.checkpoint_callback.dirpath)

if __name__ == "__main__":
    file_txt = '/root/code/ControlNet/real_human_laion_danbooru.txt'
    pretrained_model_name_or_path = '/oss/comicai/zhiyuan.shi/models/stable-diffusion-2-1'
    pretrained_vae_model_name_or_path = None
    logger_freq = 4000
    batch_size = 12
    learning_rate = 1e-5
    save_name = 'real_human_laion_danbooru'

    tokenizer, text_encoder, vae, unet, controlnet, noise_scheduler = initial_models(pretrained_model_name_or_path, pretrained_vae_model_name_or_path)

    ComicData = ComicDataModule(batch_size=batch_size, file_txt=file_txt)
    my_model = StableDiffusionXL(vae=vae, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, lr=learning_rate, controlnet=controlnet, \
                                 noise_scheduler=noise_scheduler, use_ema=False, use_snr=False, snr_gamma=5.0)

    cur_dir = os.getcwd()
    checkpoint_callback = ModelCheckpoint(dirpath='%s/%s' % (cur_dir, save_name),
                                        monitor='epoch',
                                        save_top_k=20,  # 保存最好的 k 个 checkpoint
                                        every_n_epochs=1,      # 每隔2个 epoch 保存一次 checkpoint
                                        filename="%s-{epoch:02d}-{val_loss:.2f}" % save_name)

    tensorboard = TensorBoardLogger(save_dir='%s/%s' % (cur_dir, save_name))
    trainer = pl.Trainer(resume_from_checkpoint=None, profiler='simple', accumulate_grad_batches=8, gradient_clip_val=1.0, max_epochs=20, \
                        logger=tensorboard, devices=4, precision=16, callbacks=[checkpoint_callback], num_nodes=1, strategy='deepspeed_stage_2')#
    # Train!
    trainer.fit(my_model, ComicData)
