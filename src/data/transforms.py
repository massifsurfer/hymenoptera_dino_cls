import torch
from omegaconf import DictConfig
from torchvision.transforms import v2


def get_train_transform(cfg: DictConfig) -> v2.Compose:
    normalize = v2.Normalize(mean=list(cfg.mean), std=list(cfg.std))
    train_cfg = cfg.train

    return v2.Compose(
        [
            v2.RandomResizedCrop(
                cfg.img_size,
                scale=tuple(train_cfg.crop_scale),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=train_cfg.flip_p),
            v2.RandomRotation(degrees=train_cfg.rotation_degrees),
            v2.ColorJitter(
                brightness=train_cfg.jitter_brightness,
                contrast=train_cfg.jitter_contrast,
                saturation=train_cfg.jitter_saturation,
                hue=train_cfg.jitter_hue,
            ),
            v2.RandomGrayscale(p=train_cfg.grayscale_p),
            v2.GaussianBlur(
                kernel_size=tuple(train_cfg.blur_kernel),
                sigma=tuple(train_cfg.blur_sigma),
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )


def get_val_transform(cfg: DictConfig) -> v2.Compose:
    normalize = v2.Normalize(
        mean=list(cfg.mean),
        std=list(cfg.std),
    )

    return v2.Compose(
        [
            v2.Resize(
                (cfg.img_size, cfg.img_size),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
