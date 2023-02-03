from typing import Any

import ignite.distributed as idist
import numpy as np
import torchvision
import torchvision.transforms as T
from PIL import Image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)


def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def clip_boxes_(boxes, img_size):
    height, width = img_size
    clip_upper = np.array([height, width] * 2, dtype=boxes.dtype)
    np.clip(boxes, 0, clip_upper, out=boxes)


def resolve_fill_color(fill_color, img_mean=IMAGENET_DEFAULT_MEAN):
    if isinstance(fill_color, tuple):
        assert len(fill_color) == 3
        fill_color = fill_color
    else:
        try:
            int_color = int(fill_color)
            fill_color = (int_color,) * 3
        except ValueError:
            assert fill_color == 'mean'
            fill_color = tuple([int(round(255 * x)) for x in img_mean])
    return fill_color


class ImageToNumpy:
    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img, annotations


class ResizePad:
    def __init__(
        self,
        target_size: int,
        interpolation: str = 'bilinear',
        fill_color: tuple = (0, 0, 0),
    ):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img, anno: dict):
        width, height = img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        new_img = Image.new(
            "RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color
        )
        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_w, scaled_h), interp_method)
        new_img.paste(img)  # pastes at 0,0 (upper-left corner)

        if 'bbox' in anno:
            bbox = anno['bbox']
            bbox[:, :4] *= img_scale
            bbox_bound = (
                min(scaled_h, self.target_size[0]),
                min(scaled_w, self.target_size[1]),
            )
            clip_boxes_(
                bbox, bbox_bound
            )  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1.0 / img_scale  # back to original

        return new_img, anno


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict):
        for t in self.transforms:
            img, annotations = t(img, annotations)
        return img, annotations


def transforms_coco_eval(
    img_size=224,
    interpolation='bilinear',
    use_prefetcher=False,
    fill_color='mean',
    mean=IMAGENET_DEFAULT_MEAN,
    # std=IMAGENET_DEFAULT_STD
):
    fill_color = resolve_fill_color(fill_color, mean)

    image_tfl = [
        ResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color
        ),
        ImageToNumpy(),
    ]

    # assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf


def setup_data(config: Any):
    """Download datasets and create dataloaders

    Parameters
    ----------
    config: needs to contain `data_path`, `train_batch_size`, `eval_batch_size`, and `num_workers`
    """

    if config.dataset == 'coco':
        transform = transforms_coco_eval()

        dataset_train = torchvision.datasets.CocoDetection(
            root=config.train_path,
            annFile=config.train_ann_path,
            transform=transform,
            transforms=None,
        )
        dataset_eval = torchvision.datasets.CocoDetection(
            root=config.validation_path,
            annFile=config.validation_ann_path,
            transform=transform,
            transforms=None,
        )
    else:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset_train = torchvision.datasets.CIFAR10(
            root=config.data_path,
            train=True,
            download=True,
            transform=transform,
        )
        dataset_eval = torchvision.datasets.CIFAR10(
            root=config.data_path,
            train=False,
            download=True,
            transform=transform,
        )

    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return dataloader_train, dataloader_eval
