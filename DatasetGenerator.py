import os
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
# --------------------------------------------------------------------------------

class DatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, imgDir, datasetFile, transform):

        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        # ---- Open file, get image paths and labels

        with open(datasetFile, "r") as file:
            for line in file:
                lineItems = line.split()

                imagePath = os.path.join(imgDir, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])

        if self.transform != None: imageData = self.transform(imageData)

        return imageData, imageLabel

    # --------------------------------------------------------------------------------

    def __len__(self):

        return len(self.listImagePaths)

# --------------------------------------------------------------------------------
