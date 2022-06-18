import os
from typing import Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import Compose

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetGenerator(Dataset):
    """

    """

    def __init__(self, img_dir: str, dataset_file: str, transform: Compose):
        """

        :param self:
        :param img_dir:
        :param dataset_file:
        :param transform:
        :return:
        """
        self._list_image_paths = []
        self._list_image_labels = []
        self._transform = transform

        # ---- Open file, get image paths and labels

        with open(dataset_file, "r") as file:
            for line in file:
                line_items = line.split()

                image_path = os.path.join(img_dir, line_items[0])
                image_label = line_items[1:]
                image_label = [int(i) for i in image_label]

                self._list_image_paths.append(image_path)
                self._list_image_labels.append(image_label)

    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.FloatTensor]:
        """

        :param self:
        :param index:
        :return:
        """
        image_path = self._list_image_paths[index]

        image_data = Image.open(image_path).convert('RGB')
        image_label = torch.FloatTensor(self._list_image_labels[index])

        if self._transform is not None:
            image_data = self._transform(image_data)

        return image_data, image_label

    def __len__(self) -> int:
        """

        :param self:
        :return:
        """
        return len(self._list_image_paths)
