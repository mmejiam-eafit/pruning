from models import DenseNet121
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from DatasetGenerator import DatasetGenerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn.utils.prune as p
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_auc_score
from AverageMeter import AverageMeter
import matplotlib.pyplot as plt
import numpy as np

CLASS_COUNT = 14
IS_TRAINED = False
TOL = 1e-4
ITER = 20
EARLY_STOP = 10
PRUNE_STOP = ITER
best_prune_acc = 0
prune_count = 0

IMG_DIR = "./database"
TRAIN_FILE = "./dataset/train_2.txt"
VAL_FILE = "./dataset/val_2.txt"
TEST_FILE = "./dataset/test_2.txt"

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
               'Hernia']


def getDataLoader(batchSize, transformSequence, file):
    dataSet = DatasetGenerator(imgDir=IMG_DIR, transform=transformSequence, datasetFile=file)
    dataLoader = DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=True, num_workers=0,
                            pin_memory=True)

    return dataLoader


def getTrainValDataLoaders(transCrop, batchSize):
    transforms = getTrainTransforms(transCrop)
    trainDataLoader = getDataLoader(batchSize, transforms, TRAIN_FILE)
    valDataLoader = getDataLoader(batchSize, transforms, VAL_FILE)

    return trainDataLoader, valDataLoader


def getTestDataLoader(transResize, transCrop, batchSize):
    transforms = getTestTransforms(transResize, transCrop)

    return getDataLoader(batchSize, transforms, TEST_FILE)


def getTrainTransforms(transCrop):
    normalize = transforms.Normalize([0.52, 0.52, 0.52], [0.23, 0.23, 0.23])

    transformList = []

    transformList.append(transforms.RandomResizedCrop(transCrop))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence


def getTestTransforms(transResize, transCrop):
    normalize = transforms.Normalize([0.52, 0.52, 0.52], [0.23, 0.23, 0.23])

    def toTensor(crops):
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])

    def normal(crops):
        return torch.stack([normalize(crop) for crop in crops])

    transformList = []
    transformList.append(transforms.Resize(transResize))
    transformList.append(transforms.TenCrop(transCrop))
    transformList.append(transforms.Lambda(toTensor))
    transformList.append(transforms.Lambda(normal))
    transformSequence = transforms.Compose(transformList)

    return transformSequence


def prune():
    global prune_count
    global best_prune_acc

    model = DenseNet121(classCount=CLASS_COUNT, isTrained=IS_TRAINED)
    model = nn.DataParallel(model).cuda()
    batchSize = 200
    maxEpoch = 10

    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 320
    imgtransCrop = 299

    trainDataLoader, valDataLoader = getTrainValDataLoaders(transCrop=imgtransCrop, batchSize=batchSize)
    testDataLoader = getTestDataLoader(transResize=imgtransResize, transCrop=imgtransCrop, batchSize=batchSize)

    # Pruning cycle
    for k in range(0, ITER):

        if prune_count > PRUNE_STOP:
            print(f'Early Stop finish pruning in iteration {prune_count} with avg val acc: {best_acc}')
            break

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

        for epoch in range(1, maxEpoch + 1):
            if epochs_no_improve > EARLY_STOP:
                print(f'Early Stop finish training in epoch {epoch} with avg val acc: {val_acc}')
                break
            t_loss, t_acc = getModelLossInfo(model, trainDataLoader, optimizer, loss, True)
            v_loss, v_acc = getModelLossInfo(model, valDataLoader, None, loss, False)
            val_acc = v_acc.avg
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            scheduler.step(t_loss.avg)

            if not is_best:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
            train_loss.append(t_loss.avg)
            train_acc.append(t_acc.avg)
            val_loss.append(v_loss.avg)
            val_accuracies.append(v_acc.avg)

        plot_training_stats(train_loss, train_acc, val_loss, val_accuracies)

        test_loss, test_acc = test(model, testDataLoader)

        testMean = np.array(test_acc).mean()


        is_best_prune = testMean > best_prune_acc
        best_prune_acc = max(best_prune_acc, testMean)

        if not is_best_prune:
            prune_count += 1
        else:
            prune_count = 0

        # Unstructured pruning
        p.l1_unstructured(model.module.densenet121.features.denseblock1.denselayer1.conv2, "weight", amount=0.3)

        # print(model.module.densenet121.features.denseblock1.denselayer1.conv2.weight_mask.sum())

        # Structured pruning
        # model.module.densenet121.features.denseblock1.denselayer1.conv2 = p.ln_structured(
        #     model.module.densenet121.features.denseblock1.denselayer1.conv2,"weight", amount=0.3, n=1, dim=0)

    for i,name in model.module.densenet121.features.denseblock1.denselayer1.conv2.named_parameters():
        print(f"{i} = {name.size()}")

    test_targets, test_logits = test(model, testDataLoader)

    test_predictions = (test_logits >= 0.5) * 1

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampSTART = timestampDate + '-' + timestampTime

    print(multilabel_confusion_matrix(y_true=test_targets, y_pred=test_predictions))

    torch.save({'state_dict': model.state_dict(), 'best_loss': lossMIN,
                'optimizer': optimizer.state_dict()}, f'./saved_models/chexnet_prune_{timestampSTART}.pth.tar')


def accuracy(output, target):
    y_pred_tag = torch.round(output)

    correct_results_sum = (y_pred_tag == target).sum().float()
    acc = correct_results_sum / target.shape[0]
    acc = torch.round(acc * 100)

    return acc


def getModelLossInfo(model, dataLoader, optimizer, loss, isTrain):
    losses = AverageMeter()
    top_acc = AverageMeter()

    if isTrain:
        model.train()
    else:
        model.eval()

    for batchId, (input, target) in enumerate(dataLoader):
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(input)
        lossVal = loss(output, target)

        acc = accuracy(output, target)  # , topk=(1, 1))
        losses.update(lossVal.item(), input.size(0))
        top_acc.update(acc.item(), input.size(0))

        if isTrain:
            optimizer.zero_grad()
            lossVal.backward()
            optimizer.step()

    return losses, top_acc


def test(model, dataLoader):
    model.eval()
    targets = torch.autograd.Variable().cuda()
    predictions = torch.autograd.Variable().cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(dataLoader):
            input, target = input.cuda(), target.cuda()
            targets = torch.cat((targets, target), 0)
            bs, n_crops, c, h, w = input.size()
            out = model(input.view(-1, c, h, w).cuda())

            outMean = out.view(bs, n_crops, -1).mean(1)

            predictions = torch.cat((predictions, outMean.data), 0)
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    computeAUROC(targets, predictions)
    return targets, predictions


def computeAUROC(target, prediction):
    aurocs = np.array(list(map(roc_auc_score, target[:CLASS_COUNT, :], prediction[:CLASS_COUNT, :])))
    aurocMean = aurocs.mean()

    print(f"AUROC mean = {aurocMean}")
    # print(f"Individual AUROCs: ")
    # print_func = lambda className, auroc: print(f"CLASS: {className}, AUROC: {auroc}")
    # map(print_func, CLASS_NAMES, aurocs)


def plot_training_stats(t_loss, t_acc, v_loss, v_acc):
    plt.figure()
    plt.plot(t_loss)
    plt.plot([acc / 100 for acc in t_acc])
    plt.plot(v_loss)
    plt.plot([acc / 100 for acc in v_acc])
    plt.title(f'DenseNet modified model statistics')
    plt.ylabel('loss - acc')
    plt.xlabel('epoch')
    plt.legend(['T loss', 'T acc', 'V loss', 'V_acc'], loc='upper left')
    # plt.savefig(os.path.join(SAVE_PATH, f'{MODEL_NAME}_plot.png'))
    plt.show()


if __name__ == '__main__':
    prune()
