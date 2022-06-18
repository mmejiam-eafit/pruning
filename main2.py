import time
from os import path
from random import seed as random_seed

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import AUROC

from dataloader import DataloaderFactory
from evaluation import TrainEvaluator, ValEvaluator, TestEvaluator
from models import DenseNet121
from training import ModelPrunerIncremental
from utils import StatsPlotter

RANDOM_SEED = 0
CLASS_COUNT = 14
TIME_FORMAT = "%d%m%Y_%H%M%S"
DATASET_DIR = "./dataset"
IMG_DIR = "./database"
IMG_TRANS_CROP = 299
IMG_TRANS_RESIZE = 320

MODEL_NAME = f"training_test_{time.strftime(TIME_FORMAT)}"
BATCH_SIZE = 4
NUM_EPOCHS = 10
EPOCHS_EARLY_STOP = 3

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

    plotter = StatsPlotter(save_path="./plots")

    dataloader_factory = DataloaderFactory(batch_size=BATCH_SIZE, image_dir=IMG_DIR)

    loss = nn.BCELoss(reduction='sum')

    macro_auroc_evaluator = AUROC(num_classes=CLASS_COUNT, average='macro', pos_label=1)
    weighted_auroc_evaluator = AUROC(num_classes=CLASS_COUNT, average='weighted', pos_label=1)

    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, mode='min')

    train_evaluator = TrainEvaluator(loss=loss)
    val_evaluator = ValEvaluator(loss=loss)

    test_evaluator = TestEvaluator(metric=macro_auroc_evaluator)
    params = {'loss': loss, 'num_epochs': NUM_EPOCHS, 'early_stop': EPOCHS_EARLY_STOP, 'optimizer': optimizer,
              'scheduler': scheduler, 'batch_marker': 100, 'prune_percent': 30, 'prune_layers': prune_layers,
              'prune_limit': 50, 'prune_step': 5, 'prune_early_stop': 5, 'num_pruning': 8}

    transform_params = {'trans_crop': IMG_TRANS_CROP, 'trans_resize': IMG_TRANS_RESIZE}

    trainer = ModelPrunerIncremental(
        model=model,
        model_name=f"{MODEL_NAME}.pth.tar",
        train_dl=dataloader_factory.create(type="train", dataset_file=path.join(DATASET_DIR, "train_2.txt"),
                                           **transform_params),
        val_dl=dataloader_factory.create(type="val", dataset_file=path.join(DATASET_DIR, "val_2.txt"),
                                         **transform_params),
        train_evaluator=train_evaluator,
        val_evaluator=val_evaluator,
        image_dir=IMG_DIR
    )

    trainer.train(**params)

    plotter.plot_graph([trainer.accuracies['train'], trainer.accuracies['val']], label=['Train acc', 'Val acc'],
                       legend=True, save_graph=True, plot_name='accuracies', model_name=MODEL_NAME)
    plotter.plot_graph([trainer.losses['train'], trainer.losses['val']], label=['Train loss', 'Val loss'], legend=True,
                       save_graph=True, plot_name='losses', model_name=MODEL_NAME)

    targets, predictions = test_evaluator.run(model, data_loader=dataloader_factory.create(type="test",
                                                                                           dataset_file=path.join(
                                                                                               DATASET_DIR,
                                                                                               "test_2.txt"),
                                                                                           **transform_params))
    result = test_evaluator.get_evaluation(target=targets.int(), output=predictions)
    print(f"MACRO AUROC = {result}")
    test_evaluator.metric = weighted_auroc_evaluator
    result = test_evaluator.get_evaluation(target=targets.int(), output=predictions)
    print(f"WEIGHTED AUROC = {result}")
