from os import path
from random import seed as random_seed

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.prune import Identity, remove

from models import DenseNet121

EXPERIMENT_NAME = f"chexnet_experiment__iterative_pruning__s30__iter3__20221001_040054"
RANDOM_SEED = 0
CLASS_COUNT = 14

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random_seed(RANDOM_SEED)
if __name__ == '__main__':

    model = DenseNet121(classCount=CLASS_COUNT, isTrained=False)
    model = nn.DataParallel(model).cuda()

    modules_d1 = [
        model.module.densenet121.features.denseblock1.denselayer1.conv2,
        model.module.densenet121.features.denseblock1.denselayer2.conv2,
        model.module.densenet121.features.denseblock1.denselayer3.conv2,
        model.module.densenet121.features.denseblock1.denselayer4.conv2,
        model.module.densenet121.features.denseblock1.denselayer5.conv2,
        model.module.densenet121.features.denseblock1.denselayer6.conv2,
    ]

    modules_d2 = [
        model.module.densenet121.features.denseblock2.denselayer1.conv2,
        model.module.densenet121.features.denseblock2.denselayer2.conv2,
        model.module.densenet121.features.denseblock2.denselayer3.conv2,
        model.module.densenet121.features.denseblock2.denselayer4.conv2,
        model.module.densenet121.features.denseblock2.denselayer5.conv2,
        model.module.densenet121.features.denseblock2.denselayer6.conv2,
        model.module.densenet121.features.denseblock2.denselayer7.conv2,
        model.module.densenet121.features.denseblock2.denselayer8.conv2,
        model.module.densenet121.features.denseblock2.denselayer9.conv2,
        model.module.densenet121.features.denseblock2.denselayer10.conv2,
        model.module.densenet121.features.denseblock2.denselayer11.conv2,
        model.module.densenet121.features.denseblock2.denselayer12.conv2,
    ]

    modules_d3 = [
        model.module.densenet121.features.denseblock3.denselayer2.conv2,
        model.module.densenet121.features.denseblock3.denselayer4.conv2,
        model.module.densenet121.features.denseblock3.denselayer6.conv2,
        model.module.densenet121.features.denseblock3.denselayer8.conv2,
        model.module.densenet121.features.denseblock3.denselayer10.conv2,
        model.module.densenet121.features.denseblock3.denselayer12.conv2,
        model.module.densenet121.features.denseblock3.denselayer14.conv2,
        model.module.densenet121.features.denseblock3.denselayer16.conv2,
        model.module.densenet121.features.denseblock3.denselayer18.conv2,
        model.module.densenet121.features.denseblock3.denselayer20.conv2,
        model.module.densenet121.features.denseblock3.denselayer22.conv2,
        model.module.densenet121.features.denseblock3.denselayer24.conv2,
        model.module.densenet121.features.denseblock3.denselayer1.conv2,
        model.module.densenet121.features.denseblock3.denselayer3.conv2,
        model.module.densenet121.features.denseblock3.denselayer5.conv2,
        model.module.densenet121.features.denseblock3.denselayer7.conv2,
        model.module.densenet121.features.denseblock3.denselayer9.conv2,
        model.module.densenet121.features.denseblock3.denselayer11.conv2,
        model.module.densenet121.features.denseblock3.denselayer13.conv2,
        model.module.densenet121.features.denseblock3.denselayer15.conv2,
        model.module.densenet121.features.denseblock3.denselayer17.conv2,
        model.module.densenet121.features.denseblock3.denselayer19.conv2,
        model.module.densenet121.features.denseblock3.denselayer21.conv2,
        model.module.densenet121.features.denseblock3.denselayer23.conv2,
    ]

    modules_d4 = [
        model.module.densenet121.features.denseblock4.denselayer1.conv2,
        model.module.densenet121.features.denseblock4.denselayer3.conv2,
        model.module.densenet121.features.denseblock4.denselayer5.conv2,
        model.module.densenet121.features.denseblock4.denselayer7.conv2,
        model.module.densenet121.features.denseblock4.denselayer9.conv2,
        model.module.densenet121.features.denseblock4.denselayer11.conv2,
        model.module.densenet121.features.denseblock4.denselayer13.conv2,
        model.module.densenet121.features.denseblock4.denselayer15.conv2,
        model.module.densenet121.features.denseblock4.denselayer2.conv2,
        model.module.densenet121.features.denseblock4.denselayer4.conv2,
        model.module.densenet121.features.denseblock4.denselayer6.conv2,
        model.module.densenet121.features.denseblock4.denselayer8.conv2,
        model.module.densenet121.features.denseblock4.denselayer10.conv2,
        model.module.densenet121.features.denseblock4.denselayer12.conv2,
        model.module.densenet121.features.denseblock4.denselayer14.conv2,
        model.module.densenet121.features.denseblock4.denselayer16.conv2,
    ]

    prune_layers = modules_d4 + modules_d3 + modules_d2 + modules_d1

    prune_identity = Identity()

    model_path = path.join("experiments", EXPERIMENT_NAME, f"{EXPERIMENT_NAME}.pth.tar")
    prune_tuples = []

    for module in prune_layers:
        prune_tuples.append((module, "weight"))
        prune_identity.apply(module, "weight")
        if module.bias is not None:
            prune_tuples.append((module, "bias"))
            prune_identity.apply((module, "bias"))

    model_params_dict = torch.load(model_path)

    model.load_state_dict(model_params_dict["state_dict"])

    for module, param in prune_tuples:
        remove(module, param)

    torch.save({'state_dict': model.state_dict(), 'best_loss': model_params_dict['best_loss'],
                'optimizer': model_params_dict['optimizer']},
               path.join(f"experiments", EXPERIMENT_NAME, f"{EXPERIMENT_NAME}_remove_mask.pth.tar"))
