import gc
import os
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from transformers import CLIPTextModel
from transformers import CLIPTextModelWithProjection
print("complete")

class XLTextEncoder(nn.Module):
    def __init__(self, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two):
        super().__init__()
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

    def forward(self, captions, device):
        text_inputs1 = self.tokenizer_one(
            captions,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds1 = self.text_encoder_one(
            text_inputs1.input_ids.to(device),
            output_hidden_states=True,
        )

        text_inputs2 = self.tokenizer_two(
            captions,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds2 = self.text_encoder_two(
            text_inputs2.input_ids.to(device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds2[0]
        prompt_embeds = torch.concat([prompt_embeds1.hidden_states[-2], prompt_embeds2.hidden_states[-2]], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

# initialize models
def get_encoder_model(pretrained_model_name_or_path='/oss/comicai/zhiyuan.shi/models/stable-diffusion-xl-base-1.0', revision=None):
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

    text_encoder_xl = XLTextEncoder(text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two)
    return text_encoder_xl

def get_prompts(file_txt='/root/code/ControlNet/real_human_laion_danbooru.txt', file_num=None):
    img_infos = open(file_txt, 'r').readlines()
    if file_num:
        img_infos = img_infos[:file_num]
    prompts = []
    paths = []
    for img_path in img_infos:
        target_path = img_path.strip()
        target_path = target_path.replace('/root/data', '/oss/comicai/qi.xin')
        postfix = target_path.split('.')[-1]
        paths.append(target_path.replace('.%s' % postfix, '_xl.npy'))
        if 'laion_image_folder_original' not in target_path:
            prompt_path = target_path.replace('.%s' % postfix, '.txt')
        else:
            prompt_path = target_path.replace('.%s' % postfix, '_niji.txt')

        if prompt_path and len(open(prompt_path, 'r').readlines()) > 0:
            prompt = open(prompt_path, 'r').readlines()[0].strip()
        else:
            prompt = ""
        prompts.append(prompt)
    return prompts, paths

def generate_prompt_embedding_single(file_txt=None, return_value=False, file_num=None):
    print(file_txt)
    prompts, paths = get_prompts(file_txt, file_num)
    if os.environ.get('LOCAL_RANK') is None:
        device = "cuda:0"
    else:
        device = f"cuda:{os.environ.get('LOCAL_RANK')}"
    text_encoder_xl = get_encoder_model().cuda(device)

    if return_value: # for small dataset
        prompt_embeds_list, pooled_prompt_embeds_list = [], []
        for batch_idx in range(0, len(prompts), 1000):
            print(batch_idx)
            sub_prompts = prompts[batch_idx:batch_idx+1000]
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = text_encoder_xl(sub_prompts, device)
            prompt_embeds_list.append(prompt_embeds.cpu())
            pooled_prompt_embeds_list.append(pooled_prompt_embeds.cpu())
        prompt_embeds_arr = np.concatenate([prompt_embeds_ for prompt_embeds_ in prompt_embeds_list])
        pooled_prompt_embeds_arr = np.concatenate([pooled_prompt_embeds_ for pooled_prompt_embeds_ in pooled_prompt_embeds_list])

    else: # for large dataset (just run once)
        for prompt, path in tqdm(zip(prompts, paths)):
            prompt_embeds, pooled_prompt_embeds = text_encoder_xl([prompt], device)
            np.save(path, np.array({"prompt_embeds": prompt_embeds.squeeze().cpu().detach().numpy(), "pooled_prompt_embeds":pooled_prompt_embeds.squeeze().cpu().detach().numpy()}))
    
    del prompt_embeds, pooled_prompt_embeds
    del text_encoder_xl
    torch.cuda.empty_cache()
    
    if return_value:
        return prompt_embeds_arr, pooled_prompt_embeds_arr
    else:
        return 0
    
if __name__ == '__main__':
    print("true")
    generate_prompt_embedding_single(file_txt='/oss/comicai/qi.xin/wallpaper_1M_good.txt')
    
# def inference_on_sublist(rank, world_size, model, prompts):
#     # 根据rank分割数据
#     num_prompts = len(prompts)
#     prompts_per_gpu = num_prompts // world_size
#     start_idx = rank * prompts_per_gpu
#     end_idx = start_idx + prompts_per_gpu if rank != world_size - 1 else num_prompts

#     sub_prompts = prompts[start_idx:end_idx]

#     model = model.to(rank)
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

#     all_outputs = []

#     for idx in range(0, len(sub_prompts), 1500):
#         # 将prompt转化为模型需要的输入格式
#         batch_prompts = sub_prompts[idx:idx+1500]
#         with torch.no_grad():
#             prompt_embeds, pooled_prompt_embeds = model(batch_prompts)
#             all_outputs.append([prompt_embeds.cpu()])

#     # 收集所有的输出到主进程
#     gathered_outputs = [torch.zeros_like(all_outputs[0]) for _ in range(world_size)]
#     dist.gather(all_outputs[0], gather_list=gathered_outputs, dst=0)

#     return gathered_outputs

# def generate_prompt_embedding(rank, world_size, prompts):
#     # 初始化进程组
#     dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

#     model = get_encoder_model()

#     outputs = inference_on_sublist(rank, world_size, model, prompts)
    
#     print(outputs[0].shape)

# def generate_prompt_embedding_distribute():
#     prompts = get_prompts()
#       # 或者其他加载prompts的方法
#     world_size = torch.cuda.device_count()
#     mp.spawn(generate_prompt_embedding, args=(world_size, prompts), nprocs=world_size, join=True)

# generate_prompt_embedding_distribute()