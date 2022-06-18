#  -*- coding: utf-8 -*-
#  Copyright (c) 1, 2022.
#  Author: This Code has been developed by Miguel Mejía (mmejiam@eafit.edu.co)
#  Repository: pruning (git clone git@github.com:mmejiam-eafit/pruning.git)
#  Last Modified: 1/11/22, 10:51 AM by Miguel Mejía
from typing import List

import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from torch.utils.data import DataLoader

from DatasetGenerator import DatasetGenerator


class DataTransformerFactory:

    def get_transforms(self, **kwargs) -> transforms.Compose:
        assert 'type' in kwargs.keys(), "Missing transform type"
        assert kwargs['type'] in ['train', 'val', 'test'], "Invalid Transform Type"

        transform_list = []

        normalize = transforms.Normalize([0.52, 0.52, 0.52], [0.23, 0.23, 0.23])

        if kwargs['type'] == 'train' or kwargs['type'] == 'val':
            transform_list = [
                transforms.RandomResizedCrop(kwargs['trans_crop']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        elif kwargs['type'] == 'test':

            def to_tensor(crops: List[Image]):
                return torch.stack([transforms.ToTensor()(crop) for crop in crops])

            def normal(crops: List[Image]):
                return torch.stack([normalize(crop) for crop in crops])

            transform_list = [
                transforms.Resize(kwargs['trans_resize']),
                transforms.TenCrop(kwargs['trans_crop']),
                transforms.Lambda(to_tensor),
                transforms.Lambda(normal)
            ]

        return transforms.Compose(transform_list)


class DataloaderFactory:

    def __init__(self, image_dir: str, batch_size: int):
        self._image_dir = image_dir
        self._batch_size = batch_size
        self._transform_factory = DataTransformerFactory()

    def create(self, type: str, dataset_file: str, **kwargs) -> DataLoader:
        assert type in ['train', 'val', 'test'], "Invalid loader Type"

        transforms = self._transform_factory.get_transforms(type=type, **kwargs)
        data_loader = self._get_data_loader(image_dir=self._image_dir, batch_size=self._batch_size,
                                            transform_sequence=transforms, dataset_file=dataset_file)

        return data_loader

    def _get_data_loader(self, image_dir: str, batch_size: int, transform_sequence: transforms.Compose,
                         dataset_file: str) -> DataLoader:
        dataset = DatasetGenerator(img_dir=image_dir, transform=transform_sequence, dataset_file=dataset_file)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                 pin_memory=True)

        return data_loader
