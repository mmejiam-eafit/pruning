from os import path
from random import seed as random_seed

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.prune import Identity, remove
from torchmetrics import AUROC
from torchsummary import summary

from dataloader import DataloaderFactory
from evaluation import TestEvaluator
from hooks import HookManager
from logger import logger
from models import DenseNet121

RANDOM_SEED = 0
CLASS_COUNT = 14
TIME_FORMAT = "%Y%m%d_%H%M%S"
DATASET_DIR = "./dataset"
IMG_DIR = "./database"
IMG_TRANS_CROP = 299
IMG_TRANS_RESIZE = 320

EXPERIMENT_NAME = f"chexnet_experiment__iterative_pruning__s30__iter3__20221001_040054"
BATCH_SIZE = 6

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random_seed(0)
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
    for module in prune_layers:
        prune_identity.apply(module, "weight")
        if module.bias is not None:
            prune_identity.apply((module, "bias"))
    state_dict = torch.load(model_path)["state_dict"]

    total_params = 0
    for key, val in state_dict.items():
        # total_params += (val.detach().cpu().abs() > 0).sum().item()
        if "mask" in key:
            total_params += (val.detach().cpu() - 1).abs().sum().item()
    model.load_state_dict(state_dict)
    print(f"Total params = 0 -> {total_params}")
    hook_manager = HookManager()

    for module in prune_layers:
        remove(module, "weight")
        if module.bias is not None:
            remove((module, "bias"))

    summary(model, input_size=(3, 224, 224))

    model.eval()
    macro_auroc_evaluator = AUROC(num_classes=CLASS_COUNT, average='macro', pos_label=1)
    weighted_auroc_evaluator = AUROC(num_classes=CLASS_COUNT, average='weighted', pos_label=1)

    test_evaluator = TestEvaluator(metric=macro_auroc_evaluator)
    test_evaluator.logger = logger
    dataloader_factory = DataloaderFactory(batch_size=BATCH_SIZE, image_dir=IMG_DIR)
    transform_params = {'trans_crop': IMG_TRANS_CROP, 'trans_resize': IMG_TRANS_RESIZE}

    denseblocks = [
        ("denseblock1", model.module.densenet121.features.denseblock1),
        ("denseblock2", model.module.densenet121.features.denseblock2),
        ("denseblock3", model.module.densenet121.features.denseblock3),
        ("denseblock4", model.module.densenet121.features.denseblock4),
    ]

    # for name, layer in denseblocks:
    #     hook_manager.addHook(module=layer, hook_name=f"hook_layer_{name}",
    #                          fn=get_propagated_info_generator_function(
    #                              module_str=name,
    #                              image_dir=f"tests/{EXPERIMENT_NAME}/activations",
    #                              images_per_batch=1
    #                          ))

    targets, predictions = test_evaluator.run(model,
                                              data_loader=dataloader_factory.create(type="test",
                                                                                    dataset_file=path.join(
                                                                                        DATASET_DIR,
                                                                                        "test_1.txt"),
                                                                                    **transform_params),
                                              batch_marker=93)
    result = test_evaluator.get_evaluation(target=targets.int(), output=predictions)
    logger.info(f"MACRO AUROC = {result}")
    test_evaluator.metric = weighted_auroc_evaluator
    result = test_evaluator.get_evaluation(target=targets.int(), output=predictions)
    logger.info(f"WEIGHTED AUROC = {result}")
