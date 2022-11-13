import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from .base_methods import BaseMethod
import collections

try:
    import accimage
except ImportError:
    accimage = None
import random

"""
This file defines some transform examples.
Each transform method is defined by using BaseMethod class
"""


class TransToPIL(BaseMethod):
    """
    Transform method to convert images as PIL Image.
    """

    def __init__(self):
        BaseMethod.__init__(self)
        self.to_pil = transforms.ToPILImage()

    def __call__(self, data_item):
        self.set_data(data_item)

        if not self._is_pil_image(self.img):
            data_item['img'] = self.to_pil(self.img)
        if not self._is_pil_image(self.depth):
            data_item['depth'] = self.to_pil(self.depth)
        if 'depth_interp' in data_item:
            if not self._is_pil_image(self.depth_interp):
                data_item['depth_interp'] = self.to_pil(self.depth_interp)

        return data_item


class Scale(BaseMethod):
    def __init__(self, mode, size):
        BaseMethod.__init__(self, mode)
        self.scale = transforms.Resize(size, transforms.InterpolationMode.BICUBIC)

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode in ["pair", "Img"]:
            data_item['img'] = self.scale(self.img)
        if self.mode in ["pair", "depth"]:
            data_item['depth'] = self.scale(self.depth)
            if 'depth_interp' in data_item:
                data_item['depth_interp'] = self.scale(self.depth_interp)
        return data_item


class CenterCrop(BaseMethod):
    def __init__(self, mode, size):
        BaseMethod.__init__(self, mode)
        self.crop = transforms.CenterCrop(size)

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode in ["pair", "Img"]:
            data_item['img'] = self.crop(self.img)
        if self.mode in ["pair", "depth"]:
            data_item['depth'] = self.crop(self.depth)
            if 'depth_interp' in data_item:
                data_item['depth_interp'] = self.crop(self.depth_interp)
        return data_item


class RandomHorizontalFlip(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def __call__(self, data_item):
        self.set_data(data_item)

        if random.random() < 0.5:
            data_item['img'] = self.img.transpose(Image.FLIP_LEFT_RIGHT)
            data_item['depth'] = self.depth.transpose(Image.FLIP_LEFT_RIGHT)

            if 'depth_interp' in data_item:
                data_item['depth_interp'] = self.depth_interp.transpose(Image.FLIP_LEFT_RIGHT)

        return data_item


class RandomRotate(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    @staticmethod
    def rotate_pil_func():
        degree = random.randrange(-500, 500) / 100
        return lambda pil, interp: F.rotate(pil, degree, interp)

    def __call__(self, data_item):
        self.set_data(data_item)

        if random.random() < 0.5:
            rotate_pil = self.rotate_pil_func()
            data_item['img'] = rotate_pil(self.img, interp=transforms.InterpolationMode.BICUBIC)
            data_item['depth'] = rotate_pil(self.depth, interp=transforms.InterpolationMode.BICUBIC)

            if 'depth_interp' in data_item:
                data_item['depth_interp'] = rotate_pil(self.depth_interp, interp=transforms.InterpolationMode.BICUBIC)

        return data_item


class ImgAug(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    @staticmethod
    def adjust_pil(pil):
        brightness = random.uniform(0.8, 1.0)
        contrast = random.uniform(0.8, 1.0)
        saturation = random.uniform(0.8, 1.0)

        pil = F.adjust_brightness(pil, brightness)
        pil = F.adjust_contrast(pil, contrast)
        pil = F.adjust_saturation(pil, saturation)

        return pil

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item['img'] = self.adjust_pil(self.img)

        return data_item


class ToTensor(BaseMethod):
    def __init__(self, mode):
        BaseMethod.__init__(self, mode=mode)
        self.totensor = transforms.ToTensor()

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode == "Img":
            data_item['img'] = self.totensor(self.img)
        if self.mode in "depth":
            data_item['depth'] = np.array(data_item['depth'])
            data_item['depth'] = self.totensor(self.depth)
            if 'depth_interp' in data_item:
                data_item['depth_interp'] = self.totensor(self.depth_interp)

        return data_item


class ImgNormalize(BaseMethod):
    def __init__(self, mean, std):
        BaseMethod.__init__(self)
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item['img'] = self.normalize(self.img)

        return data_item


class RandomCrop(BaseMethod):
    """Crop randomly the image
    Args:
        output_size (list or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        BaseMethod.__init__(self)

        assert isinstance(output_size, (int, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data_item):
        # image, depth = data_item['img'], data_item['depth']
        self.set_data(data_item)

        w, h = self.img.size
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        data_item['img'] = self.img.crop((left, top, left + new_w, top + new_h))
        data_item['depth'] = self.depth.crop((left, top, left + new_w, top + new_h))

        return data_item



