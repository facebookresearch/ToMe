# Token Merging: Your ViT but Faster

Official PyTorch implemention of **ToMe** from our paper: [Token Merging: Your ViT but Faster](https://arxiv.org/abs/2210.09461).  
Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman.

## What is ToMe?
![ToMe Concept Figure](examples/images/concept_figure.png)

Token Merging (ToMe) allows you to take an existing Vision Transformer architecture and efficiently merge tokens inside of the network for **2-3x** faster evaluation (see [benchmark script](examples/1_benchmark_timm.ipynb)). ToMe is tuned to seamlessly fit inside existing vision transformers, so you can use it without having to do additional training (see [eval script](examples/0_validation_timm.ipynb)). And if you *do* use ToMe during training, you can reduce the accuracy drop even further while also speeding up training considerably.

## What ToMe does

![ToMe Visualization](examples/images/image_vis.png)

ToMe merges tokens based on their similarity, implicitly grouping parts of objects together. This is in contrast to token pruning, which only removes background tokens. ToMe can get away with reducing more tokens because we can merge redundant foreground tokens in addition to background ones. Visualization of merged tokens on ImageNet-1k val using a trained ViT-H/14 MAE model with ToMe. See [this example](examples/2_visualization_timm.ipynb) for how to produce these visualizations. For more, see the paper appendix.


## News
 - **[2023.03.30]** Daniel has released his implementation of ToMe for diffusion [here](https://github.com/dbolya/tomesd). Check it out! (Note: this is an external implementation not affiliated with Meta in any way).
 - **[2023.02.08]** We are delighted to announce that the Meta Research Blog has highlighted our work, Token Merging! Check out the article at [Meta Research Blog](https://research.facebook.com/blog/2023/2/token-merging-your-vit-but-faster/) for more information.
 - **[2023.01.31]** We are happy to announce that our paper has been accepted for an oral presentation at ICLR 2023.
 - **[2023.01.30]** We've released checkpoints trained with ToMe for DeiT-Ti, DeiT-S, ViT-B, ViT-L, and ViT-H!
 - **[2022.10.18]** Initial release.

## Installation
See [INSTALL.md](INSTALL.md) for installation details.

## Usage

This repo does not include training code. Instead, we provide a set of tools to patch existing vision transformer implementations. Then, you can use those implementations out of the box. Currently, we support the following ViT implementations:
 - [x] [🔗](#using-timm-models) [timm](https://github.com/rwightman/pytorch-image-models)
 - [x] [🔗](#using-swag-models-through-torch-hub) [swag](https://github.com/facebookresearch/SWAG)
 - [x] [🔗](#training-with-mae) [mae](https://github.com/facebookresearch/mae)
 
See the `examples/` directory for a set of usage examples.

<!-- - [ ] [pyslowfast](https://github.com/facebookresearch/SlowFast) (coming at some point) -->
ToMe has also been implemented externally for other applications:
 - [x] [🔗 diffusion](https://github.com/dbolya/tomesd)
 
**Note:** these external implementations aren't associated with Meta in any way.


### Using timm models

[Timm](https://github.com/rwightman/pytorch-image-models) is a commonly used implementation for vision transformers in PyTorch. As of version 0.4.12 it currently uses [AugReg](https://github.com/google-research/vision_transformer) weights.

```py
import timm, tome

# Load a pretrained model, can be any vit / deit model.
model = timm.create_model("vit_base_patch16_224", pretrained=True)
# Patch the model with ToMe.
tome.patch.timm(model)
# Set the number of tokens reduced per layer. See paper for details.
model.r = 16
```

Here are some expected results when using the timm implementation *off-the-shelf* on ImageNet-1k val using a V100:

| Model          | original acc | original im/s |  r | ToMe acc | ToMe im/s |
|----------------|-------------:|--------------:|:--:|---------:|----------:|
| ViT-S/16       |        81.41 |           953 | 13 |    79.30 |      1564 |
| ViT-B/16       |        84.57 |           309 | 13 |    82.60 |       511 |
| ViT-L/16       |        85.82 |            95 |  7 |    84.26 |       167 |
| ViT-L/16 @ 384 |        86.92 |            28 | 23 |    86.14 |        56 |

See the paper for full results with all models and all values of `r`.

We've trained some DeiT (v1) models using [the official implementation](https://github.com/facebookresearch/deit]). To use, instantiate a DeiT timm model, patch it with the timm patch (`prop_attn=True`), and use ImageNet mean and variance for data loading.

| Model      | original acc | original im/s | r  | ToMe acc | ToMe im/s | Checkpoint                                                                  |
|------------|--------------|---------------|----|----------|-----------|-----------------------------------------------------------------------------|
| DeiT-S/16  | 79.8         | 930           | 13 | 79.36    | 1550      | [deit_S_r13](https://dl.fbaipublicfiles.com/tome/f367470145_deit_S_r13.pth) |
| DeiT-Ti/16 | 71.8         | 2558          | 13 | 71.27    | 3980      | [deit_T_r13](https://dl.fbaipublicfiles.com/tome/f367470145_deit_T_r13.pth) |

### Using SWAG models through Torch Hub

[SWAG](https://github.com/facebookresearch/SWAG) is a repository of massive weakly-supervised ViT models. They are available from Torch Hub and we include a function to patch its implementation.

```py
import torch, tome

# Load a pretrained model, can be one of ["vit_b16_in1k", "vit_l16_in1k", or "vit_h14_in1k"].
model = torch.hub.load("facebookresearch/swag", model="vit_b16_in1k")
# Patch the model with ToMe.
tome.patch.swag(model)
# Set the amount of reduction. See paper for details.
model.r = 45
```

Here are some results using these SWAG models *off-the-shelf* on ImageNet-1k val using a V100:

| Model          | original acc | original im/s |  r | ToMe acc | ToMe im/s |
|----------------|-------------:|--------------:|:--:|---------:|----------:|
| ViT-B/16 @ 384 |        85.30 |          85.7 | 45 |    84.59 |     167.7 |
| ViT-L/16 @ 512 |        88.06 |          12.8 | 40 |    87.80 |      26.3 |
| ViT-H/14 @ 518 |        88.55 |           4.7 | 40 |    88.25 |       9.8 |

Full results for other values of `r` are available in the paper appendix.


### Training with MAE

We fine-tune models models pretrained with MAE using the [official MAE codebase](https://github.com/facebookresearch/mae). Apply the patch as shown in [this example](examples/4_example_mae.py) and set `r` as desired (see paper appendix for full list of accuracies vs `r`). Then, follow the instructions in the MAE code-base to fine tune your model from pretrained weights.

Here are some results *after training* on ImageNet-1k val using a V100 for evaluation:

| Model    | original acc | original im/s | r  | ToMe acc | ToMe im/s | Checkpoint                                                                      |
|----------|--------------|---------------|----|----------|-----------|---------------------------------------------------------------------------------|
| ViT-B/16 | 83.62        | 309           | 16 | 81.91    | 603       | [vit_B_16_r16](https://dl.fbaipublicfiles.com/tome/f367082919_vit_B_16_r16.pth) |
| ViT-L/16 | 85.66        | 93            | 8  | 85.09    | 183       | [vit_L_16_r8](https://dl.fbaipublicfiles.com/tome/f366894475_vit_L_16_r8.pth)   |
| ViT-H/14 | 86.88        | 35            | 7  | 86.46    | 63        | [vit_H_14_r7](https://dl.fbaipublicfiles.com/tome/f366895717_vit_H_14_r7.pth)   |

To use the checkpoints, apply the MAE patch (`tome.patch.mae`) to an MAE model from the [official MAE codebase](https://github.com/facebookresearch/mae) as shown in [this example](examples/4_example_mae.py). Pass `global_pool=True` to the vit mae constructors and use ImageNet mean for data loading. For the models we trained (above checkpoints), we used `prop_attn=True` when patching with ToMe, but leave that as `False` for off-the-shelf models. Note that the original models in this table were also trained by us.

As a sanity check, here is our baseline result *without training* using the off-the-shelf ViT-L model available [here](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md) as described in Table 1 of the paper:

| Model          | original acc | original im/s |  r | ToMe acc | ToMe im/s |
|----------------|-------------:|--------------:|:--:|---------:|----------:|
| ViT-L/16       |        85.96 |            93 |  8 |    84.22 |       183 |


## License and Contributing

Please refer to the [CC-BY-NC 4.0](LICENSE). For contributing, see [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citation
If you use ToMe or this repository in your work, please cite:
```
@inproceedings{bolya2022tome,
  title={Token Merging: Your {ViT} but Faster},
  author={Bolya, Daniel and Fu, Cheng-Yang and Dai, Xiaoliang and Zhang, Peizhao and Feichtenhofer, Christoph and Hoffman, Judy},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
