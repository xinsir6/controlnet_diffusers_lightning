# 代码库

- 当前ControlNet缺乏针对stable diffusion2.1以及stable diffusion XL版本的controlnet的高效的训练代码，原始的ControlNet作者zhanglvmin公布的代码对于1.5的模型是很友好的，
  但是对2.1的训练效果并不理想。Diffusers提供的controlnet训练代码十分冗余，并且使用accelerate做并行加速，但是主流的T2I模型基本使用pytorch lightning做并行加速。本代码库参
  考了若干开源的代码，并对其进行了整理和重构，提供了一个有效的训练Controlnet各个版本(1.5/2.1/sdxl)的模型的代码，参考zhanglvmin的原始controlnet代码的简洁性，保留了原始的
  diffusers的功能，速度比diffusers更快，并且增加了Bucket的功能，支持多节点并行训练，效果已经得到验证，特此开源以帮助其他对controlnet有训练需求的人。
 
# 原始diffusers的sdxl版本的text to image存在的问题

- **Image VAE Encoder预加载问题**: 提取特征后，图片丧失了增强功能。
- **Text Double Encoder预加载问题**: 提取特征会导致内存随着图片数量的增加而爆炸。对于服务器内存限制为1T的情况
下，原始脚本最多可以处理约60万图片。
- **多线程问题**: Text double encoder 和 Image VAE Encoder的多线程处理导致提取进程会被重复执行两次，一次是在>主线程，一次是在复制线程。
- **SDXL第三阶段训练**: 原版本不支持SDXL的第三阶段训练方法—multi aspect ratio。因为图片特征的预加载，第二阶段
的crop encoding无法发挥数据增强的效果。
- **EMA_UNET保存问题**: 代码有错误，可以参考SD1.5/2.1的`train_text_to_image.py`脚本进行比对。
- **多节点训练**: 原版本没有明确说明如何进行多节点训练。当结合accelerate和deepspeed使用时，deepspeed config需
要随训练参数一起修改，非常不方便。
- **代码结构问题**: 代码结构混乱，每个功能部分的封装都不够完善。在分布式训练时，accelerate需要显示`to device`，这增加了代码的复杂性。

# 基于diffusers的pytorch lightning重构版本

对于上述问题，我们进行了以下改进：

- 保留了Image VAE Encoder，但不再预加载图片特征。
- 独立出Text Double Encoder的特征提取过程，用空间换时间。利用Pytorch的Prefetch机制，不会增加训练时间。
- 去除了预处理加载过程，但保留了小批量数据预加载功能（最好小于10万数据）。
- 实现了multi aspect ratio，支持数据增强。
- 纠正了EMA_UNET的保存机制，参考了SD1.5/2.1的脚本。
- Lightning与Deepspeed的参数设置可以直接在训练脚本中完成，无需手动同步deepspeed config。Lightning天然支持多节
点训练，具体实现方式可参考[此链接](https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_2.html)。
- 代码结构得到了优化。保留了原始版本几乎所有的功能，如warmup、EMA、LR scheduler等，但代码行数减少了50%。

# 当前代码介绍

- **文件结构**: 分为三个文件：`train_text_to_image_sdxl_lightning.py`，`train_text_to_image_sdxl_bucket_dataset.py`以及`train_text_to_image_sdxl_generate_prompt_embedding.py`。
- **代码思路**: 主要是将VAE加到主网络中，使用Deepspeed stage2进行显存优化。训练1024x1024分辨率的图片，batch size可以增至8。
- **训练速度**: 使用float16精度进行训练时，必须指定float16 stable的VAE。重构后的版本比原始diffusers提高了约25%的训练速度。当前版本可以不使用预加载，节约时间，且不会有内存爆炸问题。

# 需要修改的地方
- Torch默认的RandomSampler，SequentialSampler, 将所有的data_source修改为dataset，这是为了解决lightning多线程warp过程中和原始的torch的Sampler继承一致性，只有dataset能被lightning的Distributed Sampler继承
```python
class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    dataset: Sized

    def __init__(self, dataset: Sized) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
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
- 需要注意的是，这个文件通常在你的虚拟环境下，例如/root/anaconda3/envs/diffusers/lib/python3.8/site-packages/torch/utils/data/sampler.py
  
# 参考的工作，感谢他们的贡献
- https://github.com/lllyasviel/ControlNet.git
- https://github.com/TencentARC/T2I-Adapter.git
- https://github.com/huggingface/diffusers.git
- https://github.com/lllyasviel/ControlNet-v1-1-nightly.git
- https://github.com/NovelAI/novelai-aspect-ratio-bucketing
