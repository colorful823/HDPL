from torchvision import transforms
from torchvision.transforms.functional import resize, normalize
from PIL import Image
import numpy as np
import torch


class MinMaxResize:
    def __init__(self, shorter=800, longer=1333):
        self.min = shorter
        self.max = longer

    def __call__(self, x):
        w, h = x.size
        scale = self.min / min(w, h)
        if h < w:
            newh, neww = self.min, scale * w
        else:
            newh, neww = scale * h, self.min

        if max(newh, neww) > self.max:
            scale = self.max / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // 32 * 32, neww // 32 * 32

        return x.resize((neww, newh), resample=Image.BICUBIC)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# This is simple maximum entropy normalization performed in Inception paper
inception_normalize = transforms.Compose(
    [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)

# ViT uses simple non-biased inception normalization
# https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py#L132
inception_unnormalize = transforms.Compose(
    [UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)


def norm255_np(mat: np.ndarray, dtype=np.uint8) -> np.ndarray:
    min_v = np.min(mat)
    return ((mat - min_v) / (np.max(mat) - min_v) * 255).astype(dtype)


def norm255_ts(mat: torch.Tensor, dtype=torch.uint8) -> torch.Tensor:
    min_v = torch.min(mat)
    return ((mat - min_v) / (torch.max(mat) - min_v) * 255).type(dtype)


def pixelbert_np(image: np.ndarray, image_size: int, rzt=None, tnm=None,
                 dtype=None, normalize=True, crop=224) -> list[torch.Tensor, ]:
    if rzt is None:
        rzt = transforms.Compose(
            [
                MinMaxResize(image_size, int((1333 / 800) * image_size)),
                transforms.ToTensor(),
            ]
        )

    if normalize:
        if tnm is not None:
            assert len(tnm.mean) == len(tnm.std) == image.shape[-1]
        else:
            tnm = transforms.Normalize(mean=[0.5] * image.shape[-1], std=[0.5] * image.shape[-1])
        if dtype is not None:
            return [tnm(torch.stack([rzt(Image.fromarray(image[:, :, c_ind]))[0] for c_ind in range(image.shape[-1])])).type(dtype)]
        return [tnm(torch.stack([rzt(Image.fromarray(image[:, :, c_ind]))[0] for c_ind in range(image.shape[-1])]))]
    else:
        if dtype is not None:
            return [torch.stack([rzt(Image.fromarray(image[:, :, c_ind]))[0] for c_ind in range(image.shape[-1])]).type(dtype)]
        return [torch.stack([rzt(Image.fromarray(image[:, :, c_ind]))[0] for c_ind in range(image.shape[-1])])]


def pixelbert_ts(image: torch.Tensor, image_size: int, rzt=None, tnm=None, ts2pil=None) -> list[torch.Tensor, ]:
    if rzt is None:
        rzt = transforms.Compose(
            [
                MinMaxResize(image_size, int((1333 / 800) * image_size)),
                transforms.ToTensor(),
            ]
        )
    if tnm is not None:
        assert len(tnm.mean) == len(tnm.std) == image.shape[-1]
    else:
        tnm = transforms.Normalize(mean=[0.5] * image.shape[-1], std=[0.5] * image.shape[-1])
    if ts2pil is None:
        ts2pil = transforms.ToPILImage()

    return [tnm(torch.stack([rzt(ts2pil(image[:, :, c_ind]))[0] for c_ind in range(image.shape[-1])]))]


def modmis_pixelbert(image: np.ndarray, crop_rect: tuple[float], image_size: int,
                     layer_by_layer=True, norm_mean_std=None, dtype=None):
    x, y, dw, dh = crop_rect
    H, W, C = image.shape
    if x <= 1:
        x, y, dw, dh = int(W * x), int(H * y), int(W * dw), int(H * dh)
    else:
        x, y, dw, dh = int(x), int(y), int(dw), int(dh)
    image = torch.from_numpy(image[y:y+dh, x:x+dw, :]).permute(2, 0, 1)  # 裁剪
    image = resize(image, size=(image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=None)  # 放缩
    if layer_by_layer:  # 归一化
        min_values = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        max_values = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        max_values = torch.clamp(max_values, min=1e-7)
        image = (image - min_values) / (max_values - min_values)
    else:
        min_v = torch.min(image)
        image = (image - min_v) / (torch.max(image) - min_v)
    if norm_mean_std is not None:  # 标准化
        image = normalize(image, *norm_mean_std, inplace=True)
    if dtype is not None:
        image = image.type(dtype)
    return image
