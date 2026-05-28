import torch
from omegaconf import DictConfig
from torchvision.transforms import v2


def get_train_transform(cfg: DictConfig) -> v2.Compose:
    """Generates an aggressive data augmentation and normalization pipeline for model training.

    This function constructs a Torchvision V2 processing pipeline tailored for image
    classification models. It chains together structural transformations, color
    distortions, and pixel normalization:
    1. Extracts normalization constants and specific training augmentation parameters
       from the configuration.
    2. Incorporates random scaling, cropping, spatial reflections, and rotations.
    3. Adds photometric perturbations via color jitter, stochastic grayscale conversions,
       and Gaussian blurring.
    4. Casts raw images to floating-point tensors and shifts their distributions
       using preset channel-wise means and standard deviations.

    Args:
        cfg (DictConfig): A subset of the Hydra configuration containing general
            image parameters (size, mean, std) and training-specific hyper-parameters.

    Returns:
        v2.Compose: A callable pipeline that applies stochastic transformations
            to training image batches.
    """

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
    """Generates a deterministic preprocessing and normalization pipeline for model validation.

    This function creates a minimal, reproducible sequence of operations designed
    to evaluate model performance without data distortion:
    1. Extracts standard normalization constants from the input parameters.
    2. Resizes target images to uniform spatial dimensions using high-fidelity
       bicubic interpolation.
    3. Converts native image instances into generic PyTorch image containers.
    4. Casts data types to floating-point tensors and scales intensity values before
       applying static channel-wise standardization.

    Args:
        cfg (DictConfig): A subset of the Hydra configuration containing uniform
            image target sizes alongside channel mean and standard deviation matrices.

    Returns:
        v2.Compose: A callable pipeline that standardizes validation or test images
            prior to inference.
    """

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
