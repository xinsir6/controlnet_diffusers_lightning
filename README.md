# Code Repository

- Currently, ControlNet lacks efficient training code for stable diffusion 2.1 and stable diffusion XL versions. The original ControlNet code by zhanglvmin is friendly for the 1.5 model, but its training performance for 2.1 is not ideal. Diffusers provide controlnet training code, but it is redundant and uses `accelerate` for parallel acceleration, while mainstream T2I models typically use PyTorch Lightning for parallelization. This code repository references several open-source projects, refactors and organizes them to provide effective training code for various versions of ControlNet (1.5/2.1/SDXL). It retains the original diffusers' functionalities, offers faster training speeds, and adds bucket functionality.

# Issues with the Original Diffusers SDXL Version of Text to Image

- **Image VAE Encoder Preloading Issue**: After feature extraction, images lose their enhancement functionality.
- **Text Double Encoder Preloading Issue**: Feature extraction leads to memory explosion as the number of images increases. For a server with a memory limit of 1TB, the original script can handle about 600,000 images at most.
- **Multithreading Issue**: Multithreaded processing in Text Double Encoder and Image VAE Encoder results in repeated execution of the extraction process, once in the main thread and once in a duplicate thread.
- **SDXL Third-Stage Training**: The original version does not support SDXL's third-stage training method—multi-aspect ratio. Due to preloading of image features, the second-stage crop encoding cannot utilize data augmentation effectively.
- **EMA_UNET Saving Issue**: There is an error in the code. Reference the `train_text_to_image.py` script from SD 1.5/2.1 for comparison.
- **Multi-Node Training**: The original version does not clearly specify how to perform multi-node training. When combined with `accelerate` and `deepspeed`, modifying `deepspeed` config alongside training parameters is cumbersome.
- **Code Structure Issue**: The code structure is chaotic, and encapsulation of each functional part is incomplete. In distributed training, `accelerate` requires explicit `to device` placement, increasing code complexity.

# PyTorch Lightning Refactor Based on Diffusers

To address the aforementioned issues, we have made the following improvements:

- Image VAE Encoder is retained but no longer preloads image features.
- The feature extraction process of the Text Double Encoder is separated, trading space for time. PyTorch's Prefetch mechanism is used, and it does not increase training time.
- We removed the preprocessing loading process but retained small-batch data preloading functionality (ideally for less than 100,000 data points).
- Implemented multi-aspect ratio support for data augmentation.
- Corrected EMA_UNET saving mechanism, referencing scripts from SD 1.5/2.1 for guidance.
- Lightning and Deepspeed parameter settings can be done directly in the training script, no manual syncing of deepspeed config is needed. Lightning inherently supports multi-node training; refer to [this link](https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_2.html) for implementation details.
- The code structure has been optimized, reducing the number of lines by 50%.

# Current Code Overview

- **File Structure**: Split into three files: `train_text_to_image_sdxl_lightning.py`, `train_text_to_image_sdxl_bucket_dataset.py`, and `train_text_to_image_sdxl_generate_prompt_embedding.py`.
- **Code Approach**: Mainly adds VAE to the main network, utilizing Deepspeed stage 2 for GPU memory optimization. Training for 1024x1024 resolution images allows for a batch size increase to 8.
- **Training Speed**: When training with float16 precision, you must specify a float16-stable VAE. The refactored version achieves about 25% faster training speed compared to the original Diffusers. The current version can train without preloading, saving time and avoiding memory explosions.

# Areas to Modify
- The default RandomSampler and SequentialSampler in PyTorch have been modified to replace all instances of data_source with dataset. This change has been made to ensure consistency between
Lightning's multithreading in the warp process and the original Torch Sampler. Only dataset can be inherited by Lightning's Distributed Sampler.

```python
# Change instances of 'data_source' to 'dataset' in the following code snippets:

class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        dataset (Dataset): dataset to sample from  # Change 'data_source' to 'dataset'
    """
    dataset: Sized

    def __init__(self, dataset: Sized) -> None:
        self.dataset = dataset

    # ... (remaining code)

class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        dataset (Dataset): dataset to sample from  # Change 'data_source' to 'dataset'
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    dataset: Sized
    replacement: bool

    def __init__(self, dataset: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.dataset = dataset
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        # ... (remaining code)
        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.dataset)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples
```
- Please note that this file is typically located in your virtual environment, e.g., /root/anaconda3/envs/diffusers/lib/python3.8/site-packages/torch/utils/data/sampler.py

# Acknowledgments and References

- https://github.com/lllyasviel/ControlNet.git
- https://github.com/TencentARC/T2I-Adapter.git
- https://github.com/huggingface/diffusers.git
- https://github.com/lllyasviel/ControlNet-v1-1-nightly.git
- https://github.com/NovelAI/novelai-aspect-ratio-bucketing
