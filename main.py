from models import DenseNet121
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from DatasetGenerator import DatasetGenerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn.utils.prune as p
from sklearn.metrics import multilabel_confusion_matrix
from AverageMeter import AverageMeter
import matplotlib.pyplot as plt
import numpy as np
import time
from random import randrange, randint
from os import path
from torchmetrics import AUROC
from collections import OrderedDict

CLASS_COUNT = 14
IS_TRAINED = False
PRUNE_AMOUNT = 0.3
TOL = 1e-4
ITER = 10
EARLY_STOP = 5
PRUNE_STOP = 5
BATCH_SIZE = 8
BATCH_MARKER = 107
MAX_EPOCH = 10
# ---- Parameters related to image transforms: size of the down-scaled image, cropped image
IMG_TRANS_RESIZE = 320
IMG_TRANS_CROP = 299

SAVE_PATH = './saved_models/'
IMG_DIR = "./database"
TRAIN_FILE = "./dataset/train_1.txt"
VAL_FILE = "./dataset/val_1.txt"
TEST_FILE = "./dataset/test_1.txt"
CURR_DATE = time.strftime("%d%m%Y")
CURR_TIME = time.strftime("%H%M%S")
MODEL_NAME = f"chexnet_prune_{CURR_DATE}_{CURR_TIME}"

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
               'Hernia']


def getTime(finish, start=0):
    total = round(finish - start)

    secs = total % 60
    total -= secs
    total = round(total / 60)

    mins = total % 60
    total -= mins

    hours = round(total / 60) if total > 60 else 0

    return hours, mins, secs


def getDataLoader(batchSize, transformSequence, file):
    data_set = DatasetGenerator(imgDir=IMG_DIR, transform=transformSequence, datasetFile=file)
    data_loader = DataLoader(dataset=data_set, batch_size=batchSize, shuffle=True, num_workers=0,
                             pin_memory=True)

    return data_loader


def getTrainValDataLoaders(transCrop, batchSize):
    transforms = getTrainTransforms(transCrop)
    trainDataLoader = getDataLoader(batchSize, transforms, TRAIN_FILE)
    valDataLoader = getDataLoader(batchSize, transforms, VAL_FILE)

    return trainDataLoader, valDataLoader


def getTestDataLoader(trans_resize, trans_crop, batch_size):
    transforms = getTestTransforms(trans_resize, trans_crop)

    return getDataLoader(batch_size, transforms, TEST_FILE)


def getTrainTransforms(trans_crop):
    normalize = transforms.Normalize([0.52, 0.52, 0.52], [0.23, 0.23, 0.23])

    transform_list = [
        transforms.RandomResizedCrop(trans_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    transform_sequence = transforms.Compose(transform_list)

    return transform_sequence


def getTestTransforms(trans_resize, trans_crop):
    normalize = transforms.Normalize([0.52, 0.52, 0.52], [0.23, 0.23, 0.23])

    def toTensor(crops):
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])

    def normal(crops):
        return torch.stack([normalize(crop) for crop in crops])

    transform_list = [
        transforms.Resize(trans_resize),
        transforms.TenCrop(trans_crop),
        transforms.Lambda(toTensor),
        transforms.Lambda(normal)
    ]
    transform_sequence = transforms.Compose(transform_list)

    return transform_sequence


def prune(is_global=False):
    best_prune_acc = 0
    prune_count = 0

    start_time = time.time()

    model = DenseNet121(classCount=CLASS_COUNT, isTrained=IS_TRAINED)
    model = nn.DataParallel(model).cuda()

    modules_d1 = [
        model.module.densenet121.features.denseblock1.denselayer1.conv2,
        model.module.densenet121.features.denseblock1.denselayer2.conv2,
        model.module.densenet121.features.denseblock1.denselayer3.conv2,
        model.module.densenet121.features.denseblock1.denselayer4.conv2,
        model.module.densenet121.features.denseblock1.denselayer5.conv2,
        model.module.densenet121.features.denseblock1.denselayer6.conv2,
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
    ]

    train_data_loader, val_data_loader = getTrainValDataLoaders(transCrop=IMG_TRANS_CROP, batchSize=BATCH_SIZE)
    test_data_loader = getTestDataLoader(trans_resize=IMG_TRANS_RESIZE, trans_crop=IMG_TRANS_CROP,
                                         batch_size=BATCH_SIZE)

    # Pruning cycle
    for k in range(0, ITER):
        print(f"============= Pruning cycle #{k + 1} =============")

        if prune_count > PRUNE_STOP:
            print(f'Early Stop finish pruning in iteration {prune_count} with avg val acc: {best_acc}')
            break

        def reset_params(mod):
            print("Resetting model parameters...")
            for layer in mod.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        reset_params(model.module.densenet121)

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, mode='min')

        # -------------------- SETTINGS: LOSS
        loss = nn.BCELoss(reduction='sum')

        train_loss = []
        train_acc = []
        val_loss = []
        val_accuracies = []
        epochs_no_improve = 0
        val_acc = 0
        best_acc = 0

        for epoch in range(1, MAX_EPOCH + 1):
            print(f"Running epoch {epoch}")
            if epochs_no_improve > EARLY_STOP:
                print(f'Early Stop finish training in epoch {epoch} with avg val acc: {val_acc}')
                break
            t_loss, t_acc = getModelLossInfo(model, train_data_loader, optimizer, loss, True)

            # Clean up residual memory before evaluating
            torch.cuda.empty_cache()

            print(f"Start validation process")
            v_loss, v_acc = getModelLossInfo(model, val_data_loader, None, loss, False)
            val_acc = v_acc.avg
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            scheduler.step(t_loss.avg)

            if not is_best:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0

            print(f"Statistics: \n Train Loss: {t_loss.avg} \n Train Acc: {t_acc.avg} \n Val Loss: {v_loss.avg} \n "
                  f"Val Acc: {v_acc.avg}")

            train_loss.append(t_loss.avg)
            train_acc.append(t_acc.avg)
            val_loss.append(v_loss.avg)
            val_accuracies.append(v_acc.avg)

            del t_acc, t_loss, v_acc, v_loss

        plot_training_stats(train_loss, train_acc, val_loss, val_accuracies)

        print("Testing Model")

        test_loss, test_acc = test(model, test_data_loader)

        test_mean = np.array(test_acc).mean()

        is_best_prune = test_mean > best_prune_acc
        best_prune_acc = max(best_prune_acc, test_mean)

        # Count how many times we've been getting worse results
        if not is_best_prune:
            prune_count += 1
        else:
            prune_count = 0

        if is_global:
            #Do something
            prune_modules = modules_d1 + modules_d3
            prune_tuples = []

            for module in prune_modules:
                prune_tuples.append((module, "weight"))
                prune_tuples.append((module, "bias"))

            p.global_unstructured(parameters=prune_tuples, pruning_method=p.L1Unstructured, amount=PRUNE_AMOUNT)

            for module, param in prune_tuples:
                p.remove(module, param)
        else:
            # Coinflip to determine which module to take
            if randint(0, 1) == 1:
                use_list = modules_d1
            else:
                use_list = modules_d3

            # Select a random layer from the list to be used
            rand_idx = randrange(len(use_list) - 1)

            # Unstructured pruning
            p.l1_unstructured(use_list[rand_idx], "weight", amount=PRUNE_AMOUNT)
            p.l1_unstructured(use_list[rand_idx], "bias", amount=PRUNE_AMOUNT)
            p.remove(use_list[rand_idx], "weight")

        # Clean up residual memory before starting over
        torch.cuda.empty_cache()

    print("============= Testing Pruned Model =============")
    test_targets, test_logits = test(model, test_data_loader)
    test_predictions = (test_logits >= 0.5) * 1

    print(multilabel_confusion_matrix(y_true=test_targets, y_pred=test_predictions))

    finish_time = time.time()

    hours, mins, secs = getTime(finish_time, start_time)

    print(f"Finished pruning in {hours:02d}:{mins:02d}:{secs:02d}")

    torch.save({'state_dict': model.state_dict(), 'best_loss': np.array(train_loss).mean(),
                'optimizer': optimizer.state_dict()}, path.join(SAVE_PATH, f'{MODEL_NAME}.pth.tar'))


def accuracy(output, target):
    y_pred_tag = torch.round(output)

    correct_results_sum = (y_pred_tag == target).sum().float()
    acc = correct_results_sum / target.shape[0]
    acc = torch.round(acc * 100)

    return acc


def getModelLossInfo(model, data_loader, optimizer, loss, is_train):
    losses = AverageMeter()
    top_acc = AverageMeter()

    if is_train:
        model.train()
    else:
        model.eval()

    for batchId, (input, target) in enumerate(data_loader):
        # Add a marker for batches run, as a rough estimate of where the model is looking at a given time
        if batchId % BATCH_MARKER == 0:
            print(f"Processing batch {batchId}")
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(input)
        loss_val = loss(output, target)

        acc = accuracy(output, target)  # , topk=(1, 1))
        losses.update(loss_val.item(), input.size(0))
        top_acc.update(acc.item(), input.size(0))

        if is_train:
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        # Explicit cleanup of variables
        del output

    return losses, top_acc


def test(model, data_loader):
    model.eval()
    targets = torch.autograd.Variable().cuda()
    predictions = torch.autograd.Variable().cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            if i % BATCH_MARKER == 0:
                print(f"Testing batch {i}")
            input, target = input.cuda(), target.cuda()
            targets = torch.cat((targets, target), 0)
            bs, n_crops, c, h, w = input.size()
            out = model(input.view(-1, c, h, w).cuda())

            outMean = out.view(bs, n_crops, -1).mean(1)

            predictions = torch.cat((predictions, outMean.data), 0)

    computeAUROC(targets, predictions)
    return targets.cpu().numpy(), predictions.cpu().numpy()


def computeAUROC(target, prediction):
    """
    Get the AUROC for all classes, and calculate individual AUROC.
    Individual AUROC is calculated as a 1 vs all evaluation, where it checks
    for each class when it was correctly evaulated and when it was not.
    """
    # aurocs = np.array(list(map(roc_auc_score, target, prediction[:CLASS_COUNT, :])))
    target = target.int()
    macro_auroc_evaluator = AUROC(num_classes=CLASS_COUNT, average='macro', pos_label=1)
    weighted_auroc_evaluator = AUROC(num_classes=CLASS_COUNT, average='weighted', pos_label=1)

    macro_auroc = macro_auroc_evaluator(prediction, target)
    weighted_auroc = weighted_auroc_evaluator(prediction, target)

    del macro_auroc_evaluator, weighted_auroc_evaluator

    print(f"MACRO AUROC = {macro_auroc}")
    print(f"WEIGHTED AUROC = {weighted_auroc}")
    print(f"Individual AUROCs: ")
    class_aurocs = OrderedDict()

    for i in range(CLASS_COUNT):
        auroc_evaluator = AUROC(num_classes=None, average='macro', pos_label=1)
        class_aurocs[CLASS_NAMES[i]] = auroc_evaluator(prediction[:, i], target[:, i])
        del auroc_evaluator

    for name, value in class_aurocs.items():
        print(f"CLASS: {name}, AUROC: {value}")


def plot_training_stats(t_loss, t_acc, v_loss, v_acc):
    curr_date = time.strftime("%d%m%Y")
    curr_time = time.strftime("%H%M%S")
    plt.figure()
    plt.plot(t_loss)
    plt.plot([acc / 100 for acc in t_acc])
    plt.plot(v_loss)
    plt.plot([acc / 100 for acc in v_acc])
    plt.title(f'DenseNet modified model statistics')
    plt.ylabel('loss - acc')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Train acc', 'Val loss', 'Val acc'], loc='upper left')
    plt.savefig(path.join(SAVE_PATH, f'{MODEL_NAME}_{curr_date}-{curr_time}_plot.png'))
    # plt.show()


if __name__ == '__main__':
    prune(is_global=True)
