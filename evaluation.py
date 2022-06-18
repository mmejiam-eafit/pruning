#  -*- coding: utf-8 -*-
#  Copyright (c) 1, 2022.
#  Author: This Code has been developed by Miguel Mejía (mmejiam@eafit.edu.co)
#  Repository: pruning (git clone git@github.com:mmejiam-eafit/pruning.git)
#  Last Modified: 1/11/22, 10:51 AM by Miguel Mejía
from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

from AverageMeter import AverageMeter


class _ModelLossEvaluator(ABC):
    def __init__(self, loss: _Loss):
        """

        :param loss:
        """
        self._losses = None
        self._top_acc = None
        self._loss = loss
        self.reset_params()

    @property
    def losses(self) -> AverageMeter:
        """

        :return:
        """
        return self._losses

    @property
    def top_acc(self) -> AverageMeter:
        """

        :return:
        """
        return self._top_acc

    def reset_params(self):
        """

        :return:
        """
        self._losses = AverageMeter()
        self._top_acc = AverageMeter()

    def update_params(self, values: List[Tuple[float, int]]):
        """

        :param values:
        :return:
        """
        self._losses.update(values[0][0], values[0][1])
        self._top_acc.update(values[1][0], values[1][1])

    @abstractmethod
    def run(self, model: nn.Module, data_loader: DataLoader, **kwargs) -> Union[
        Tuple[AverageMeter, AverageMeter], Tuple[Tensor, Tensor]]:
        """

        :param model:
        :param data_loader:
        :param kwargs:
        :return:
        """
        raise NotImplementedError("Method needs to be overridden and ran in child classes.")

    def get_accuracy(self, target: Tensor, predicted: Tensor) -> Tensor:
        y_pred_tag = torch.round(predicted)

        correct_results_sum = (y_pred_tag == target).sum().float()
        acc = correct_results_sum / target.shape[0]
        acc = torch.round(acc * 100)

        return acc


class TestEvaluator(_ModelLossEvaluator):
    """

    """

    def __init__(self, metric: Metric):
        self._metric = metric

    @property
    def metric(self) -> Metric:
        return self._metric

    @metric.setter
    def metric(self, new_metric: Metric):
        self._metric = new_metric

    def run(self, model: nn.Module, data_loader: DataLoader, **kwargs) -> Tuple[Tensor, Tensor]:
        model.eval()
        targets = None
        predictions = None

        with torch.no_grad():
            for i, (data_input, target) in enumerate(data_loader):
                # Add a marker for batches run, as a rough estimate of where the model is looking at a given time
                if kwargs.get('batch_marker', None) is not None:
                    if batchId % kwargs.get('batch_marker', None) == 0:
                        print(f"Testing batch {i}")
                data_input, target = data_input.cuda(), target.cuda()

                if targets is None:
                    targets = torch.empty(size=target.size(), requires_grad=False).cuda()

                targets = torch.cat((targets, target), 0)
                batch_size, num_crops, num_channels, height, width = data_input.size()
                out = model(data_input.view(-1, num_channels, height, width).cuda())
                out_mean = out.view(batch_size, num_crops, -1).mean(1)

                if predictions is None:
                    predictions = torch.empty(size=out_mean.size(), requires_grad=False).cuda()

                predictions = torch.cat((predictions, out_mean.data), 0)

        return targets, predictions

    def get_evaluation(self, target: Any, output: Tensor) -> Any:
        return self._metric(output, target)


class ValEvaluator(_ModelLossEvaluator):
    """

    """

    def run(self, model: nn.Module, data_loader: DataLoader, **kwargs) -> Tuple[AverageMeter, AverageMeter]:
        """

        :param model:
        :param data_loader:
        :param kwargs:
        :return:
        """
        self.reset_params()

        model.eval()

        for batchId, (input, target) in enumerate(data_loader):
            # Add a marker for batches run, as a rough estimate of where the model is looking at a given time
            if kwargs.get('batch_marker', None) is not None:
                if batchId % kwargs.get('batch_marker', None) == 0:
                    print(f"Processing batch {batchId}")

            input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(input)
            loss_val = self._loss(output, target)
            acc = self.get_accuracy(target, output)

            curr_loss_values = (loss_val.item(), input.size(0))
            curr_acc_values = (acc.item(), input.size(0))

            self.update_params([curr_loss_values, curr_acc_values])

            # Explicit cleanup of variables
            del output

        return self._losses, self._top_acc


class TrainEvaluator(_ModelLossEvaluator):
    """

    """

    def run(self, model: nn.Module, data_loader: DataLoader, **kwargs) -> Tuple[AverageMeter, AverageMeter]:
        """

        :param model:
        :param data_loader:
        :param kwargs:
        :return:
        """
        assert 'optimizer' in kwargs.keys(), "No optimizer defined"
        assert isinstance(kwargs['optimizer'], Optimizer), "Optimizer option should be an instance of Optimizer"

        optimizer = kwargs['optimizer']
        model.train()

        for batchId, (data_input, target) in enumerate(data_loader):
            # Add a marker for batches run, as a rough estimate of where the model is looking at a given time
            if kwargs.get('batch_marker', None) is not None:
                if batchId % kwargs.get('batch_marker', None) == 0:
                    print(f"Processing batch {batchId}")

            data_input, target = data_input.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(data_input)
            loss_val = self._loss(output, target)
            acc = self.get_accuracy(target, output)

            curr_loss_values = (loss_val.item(), data_input.size(0))
            curr_acc_values = (acc.item(), data_input.size(0))

            self.update_params([curr_loss_values, curr_acc_values])

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Explicit cleanup of variables
            del output

        return self._losses, self._top_acc
