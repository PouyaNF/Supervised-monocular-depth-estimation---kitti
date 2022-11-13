import torch
import numpy as np
import torchvision.transforms as transforms
import Transformer.custom_methods as augmethods
from .base_transformer import BaseTransformer


class CustTransformer(BaseTransformer):
    """
    An example of Custom Transformer.
    This class should work with custom transform methods which defined in custom_methods.py
    """
    def __init__(self, phase):
        BaseTransformer.__init__(self, phase)
        if not self.phase in ["train", "test", "val"]:
            raise ValueError("Panic::Invalid phase parameter")
        else:
            pass



    def get_joint_transform(self):


        if self.phase == "train":
            return transforms.Compose([
                                        augmethods.TransToPIL(),
                                        # augmethods.CenterCrop("pair", [224, 672]),
                                        augmethods.Scale("pair", [224, 672]),  # augmethods.Scale("pair", [224, 721]),
                                        # augmethods.RandomCrop([224, 224]),
                                        augmethods.RandomHorizontalFlip(),
                                        augmethods.RandomRotate()
                                        ])
        else:
            return transforms.Compose([
                                            augmethods.TransToPIL(),
                                            augmethods.Scale("pair", [224, 672]),     # augmethods.Scale("pair", [224, 721]),
                                            #augmethods.RandomCrop([224, 224]),
                                            #augmethods.CenterCrop("pair", [224, 672]),
                                        ])



    def get_img_transform(self):
        if self.phase == "train":
            return transforms.Compose([augmethods.ImgAug(),
                                       augmethods.ToTensor("Img"),
                                        # augmethods.ImgNormalize([0.5, .5, .5], [.5, .5, .5])
                                       augmethods.ImgNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                       ])

        else:
            return transforms.Compose([# augmethods.Scale("Img", [224, 224]),
                                       augmethods.ToTensor("Img"),
                                       # augmethods.ImgNormalize([0.5, .5, .5], [.5, .5, .5])
                                       augmethods.ImgNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])

    def get_depth_transform(self):
        return transforms.Compose([augmethods.ToTensor("depth")])


