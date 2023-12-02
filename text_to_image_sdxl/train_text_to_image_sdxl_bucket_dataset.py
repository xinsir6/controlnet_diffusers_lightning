import cv2
import os
import math
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision import transforms
from collections import defaultdict
from itertools import chain, repeat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, BatchSampler, SequentialSampler, RandomSampler

# used for bucket dataset
def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

def get_prng(seed):
    return np.random.RandomState(seed)

class ComicDatasetBucket(Dataset):

    def __init__(self, file_path, prompt_embeds=None, pooled_prompt_embeds=None, max_size=(1024,1024), divisible=64, stride=16, min_dim=512, base_res=(1024,1024), max_ar_error=4, dim_limit=2048):
        self.base_res = base_res
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        max_tokens = (max_size[0]/stride) * (max_size[1]/stride)
        self.get_resolution(file_path)
        self.gen_buckets(min_dim, max_tokens, dim_limit, stride, divisible)
        self.assign_buckets(max_ar_error)
        self.gen_index_map()
        
    def get_resolution(self, file_path):
        file_arr_path = file_path.replace('.txt', '.npy')
        if os.path.exists(file_arr_path):
            self.res_map = np.load(file_arr_path, allow_pickle=True).item()
            return

        self.res_map = {}
        with open(file_path) as f:
            files = f.readlines()
        for _, file_ in tqdm(enumerate(files)):
            img = cv2.imread(file_.strip())
            h, w, _ = img.shape
            self.res_map[file_] = (w, h)

        np.save(file_path.replace('.txt', '.npy'), np.array(self.res_map))
        
    def gen_buckets(self, min_dim, max_tokens, dim_limit, stride=8, div=64):
        resolutions = []
        aspects = []
        w = min_dim
        while (w/stride) * (min_dim/stride) <= max_tokens and w <= dim_limit:
            h = min_dim
            got_base = False
            while (w/stride) * ((h+div)/stride) <= max_tokens and (h+div) <= dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += div
        h = min_dim
        while (h/stride) * (min_dim/stride) <= max_tokens and h <= dim_limit:
            w = min_dim
            got_base = False
            while (h/stride) * ((w+div)/stride) <= max_tokens and (w+div) <= dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += div

        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]

        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)

    def assign_buckets(self, max_ar_error=4):
        self.buckets = {}
        self.aspect_errors = []
        self.res_map_new = {}

        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w)/float(h)
            bucket_id = np.abs(self.aspects - aspect).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < max_ar_error:
                self.buckets[bucket_id].append(post_id)
                self.res_map_new[post_id] = tuple(self.resolutions[bucket_id])
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]

    def gen_index_map(self):
        self.id2path = {}
        self.id2shape = {}
        id = 0
        for path, shape in self.res_map_new.items():
            self.id2path[id] = path
            self.id2shape[id] = shape
            id += 1

    def __len__(self):
        return len(self.res_map)

    def __getitem__(self, idx):

        target_path = self.id2path[idx]
        W, H = self.res_map_new[target_path]
        target_path = target_path.strip()
        postfix = target_path.split('.')[-1]

        embeds_path = target_path.replace('.%s' % postfix, '_xl.npy')
        embeds_all = np.load(embeds_path, allow_pickle=True).item()
        prompt_embeds = embeds_all['prompt_embeds']
        pooled_prompt_embeds = embeds_all['pooled_prompt_embeds']

        if 'laion_image_folder_original' not in target_path:
            prompt_path = target_path.replace('.%s' % postfix, '.txt')
        else:
            prompt_path = target_path.replace('.%s' % postfix, '_niji.txt')
 
        if prompt_path and len(open(prompt_path, 'r').readlines()) > 0:
            prompt = open(prompt_path, 'r').readlines()[0].strip()
        else:
            prompt = ""

        target = cv2.imread(target_path)
        ori_H, ori_W, _ = target.shape
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (W, H))
        target = target.transpose((2,0,1))

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        
        return dict(pixel_values=target, original_sizes=(ori_W, ori_H), crop_top_lefts=(0, 0), target_sizes=(W, H), \
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds)


class GroupedBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=True):
        if not isinstance(sampler, Sampler):
            raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
        self.sampler = sampler
        self.group_ids = self.sampler.dataset.id2shape
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size
    
# using LightningDataModule
class ComicDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, file_txt, prompt_embeds, pooled_prompt_embeds):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        # self.dataset = dataset
        self.file_txt = file_txt
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
    
    def setup(self, stage):
        self.dataset = ComicDatasetBucket(file_path=self.file_txt, prompt_embeds=self.prompt_embeds, pooled_prompt_embeds=self.pooled_prompt_embeds)
        self.sampler = SequentialSampler(self.dataset)

    def train_dataloader(self):
        def collate_fn(examples):
            pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            target_sizes = [example["target_sizes"] for example in examples]
            prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
            pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

            return {
                "pixel_values": pixel_values,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
                "target_sizes": target_sizes
            }
        return DataLoader(self.dataset, batch_sampler=GroupedBatchSampler(sampler=self.sampler, batch_size=self.batch_size), num_workers=32, collate_fn=collate_fn)
