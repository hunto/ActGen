# Active Generation for Image Classification

Official implementation for paper "[Active Generation for Image Classification](https://arxiv.org/abs/2403.06517)", ECCV 2024.

## ImageNet / CIFAR Classification

Please see [classification/README.md](classification/README.md) for environment setup.

### Training command

```
cd classification
sh tools/dist_train.sh 8 ${CONFIG} ${MODEL} --experiment ${EXP_NAME} --gen_images
```

For example, for ViT-S/16, the MODEL should be `timm_vit_small_patch16_224`, CONFIG should be `configs/strategies/ActGen/imagenet/vit_s_16.yaml`.

### Tips for understanding the code
* The core code for generating images is in [classification/lib/gen/guided_gen.py](classification/lib/gen/guided_gen.py)
* The code for integrating ActGen into training code is in [classification/tools/train.py](classification/tools/train.py): Please find keyword `args.gen_images` to see how we implement the training.