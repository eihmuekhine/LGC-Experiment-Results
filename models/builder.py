"""Autoencoder builder with safe CPU/single-GPU behavior."""

import torch
import torch.nn as nn
import torch.nn.parallel as parallel

from . import vgg, resnet


def BuildAutoEncoder(args):
    if args.arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(args.arch)
        model = vgg.VGGAutoEncoder(configs)
    elif args.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(args.arch)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
    else:
        raise ValueError(f"Unsupported autoencoder architecture: {args.arch}")

    if getattr(args, "parallel", 0) == 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DistributedDataParallel requested but CUDA is unavailable")
        gpu = int(getattr(args, "gpu", 0))
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = parallel.DistributedDataParallel(
            model.to(torch.device(f"cuda:{gpu}")),
            device_ids=[gpu],
            output_device=gpu,
        )
    else:
        # Preserve the original DataParallel layout on CUDA so existing
        # checkpoints remain convenient, but do not hard-fail on CPU.
        if torch.cuda.is_available():
            gpu = int(getattr(args, "gpu", 0))
            model = nn.DataParallel(model).to(torch.device(f"cuda:{gpu}"))

    return model
