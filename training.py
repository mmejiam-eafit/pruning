#  -*- coding: utf-8 -*-
#  Copyright (c) 1, 2022.
#  Author: This Code has been developed by Miguel Mejía (mmejiam@eafit.edu.co)
#  Repository: pruning (git clone git@github.com:mmejiam-eafit/pruning.git)
#  Last Modified: 1/11/22, 10:51 AM by Miguel Mejía
from collections import defaultdict
from os import path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.nn.utils.prune import global_unstructured, remove, l1_unstructured
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from evaluation import _ModelLossEvaluator


class ModelTrainer(object):

    @property
    def model(self):
        return self._model

    @property
    def accuracies(self):
        return self._accuracies

    @property
    def losses(self):
        return self._losses

    @model.setter
    def model(self, new_model):
        self._model = new_model

    def __init__(self, image_dir: str, model: nn.Module, model_name: str, train_dl: DataLoader, val_dl: DataLoader,
                 train_evaluator: _ModelLossEvaluator, val_evaluator: _ModelLossEvaluator):
        """

        :param image_dir:
        :param model:
        :param model_name:
        :param train_dl:
        :param val_dl:
        :param train_evaluator:
        :param val_evaluator:
        """
        self._image_dir = image_dir
        self._model = model
        self._model_name = model_name
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._train_evaluator = train_evaluator
        self._val_evaluator = val_evaluator
        self._accuracies = defaultdict(list)
        self._losses = defaultdict(list)

    def train(self, **kwargs) -> nn.Module:
        assert 'num_epochs' in kwargs.keys(), "No number of epochs defined"
        assert 'early_stop' in kwargs.keys(), "No early stop defined"
        assert 'optimizer' in kwargs.keys(), "No optimizer defined"
        assert 'loss' in kwargs.keys(), "No loss defined"
        assert isinstance(kwargs['num_epochs'], int), "Num Epochs must be an integer"
        assert isinstance(kwargs['early_stop'], int), "Early stop must be an integer"
        assert isinstance(kwargs['optimizer'], Optimizer), "Optimizer option should be an instance of Optimizer"
        assert isinstance(kwargs['loss'], _Loss), "Loss option should be an instance of _Loss"

        epochs_no_improve = 0
        best_acc = 0

        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for epoch in range(kwargs['num_epochs']):
            print(f"Running epoch {epoch + 1} of {kwargs['num_epochs']}")

            if epochs_no_improve > kwargs['early_stop']:
                print(f"Early Stop finish training in epoch {epoch} with avg val acc: {self._accuracies['val'][-1]}")
                break

            current_train_loss, current_train_acc = self._train_evaluator.run(model=self.model,
                                                                              data_loader=self._train_dl, **kwargs)
            # Clean up residual memory before evaluating
            torch.cuda.empty_cache()

            print(f"Start validation process")
            current_val_loss, current_val_acc = self._val_evaluator.run(model=self.model, data_loader=self._val_dl)
            is_best = current_val_loss.avg > best_acc
            best_acc = max(best_acc, current_val_loss.avg)

            if kwargs['scheduler'] is not None:
                kwargs['scheduler'].step(current_train_loss.avg)

            if not is_best:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0

            print(
                f"Statistics: \n Train Loss: {current_train_loss.avg} \n Train Acc: {current_train_acc.avg} \n Val Loss: {current_val_loss.avg} \n "
                f"Val Acc: {current_val_acc.avg}")

            self._losses['train'].append(current_train_loss.avg)
            self._losses['val'].append(current_val_loss.avg)
            self._accuracies['train'].append(current_train_acc.avg)
            self._accuracies['val'].append(current_val_acc.avg)

            del current_train_loss, current_train_acc, current_val_loss, current_val_acc

        return self._model

    def save_model(self, save_path: str, optimizer: Optimizer):
        """

        :param optimizer:
        :param save_path:
        :return:
        """
        torch.save({'state_dict': self._model.state_dict(), 'best_loss': np.array(self._losses['val']).mean(),
                    'optimizer': optimizer.state_dict()}, path.join(save_path, self._model_name))


class BaseModelPruner(ModelTrainer):

    def __init__(self, image_dir: str, model: nn.Module, model_name: str, train_dl: DataLoader, val_dl: DataLoader,
                 train_evaluator: _ModelLossEvaluator, val_evaluator: _ModelLossEvaluator):
        super(BaseModelPruner).__init__(image_dir, model, model_name, train_dl, val_dl, train_evaluator, val_evaluator)
        self._keep_mask = False

    @property
    def keep_mask(self):
        """

        :return:
        """
        return self._keep_mask

    @keep_mask.setter
    def keep_mask(self, keep_mask: bool):
        """

        :param keep_mask:
        :return:
        """
        self._keep_mask = keep_mask

    def remove_pruning_params(self, module: Module, name: str):
        """

        :param module:
        :param name:
        :return:
        """
        if self.keep_mask:
            return
        remove(module, name)

    # TODO add forward hooks for pruned model


class ModelPrunerSingle(BaseModelPruner):
    def train(self, **kwargs) -> nn.Module:
        """

        :param kwargs:
        :return:
        """
        assert 'prune_percent' in kwargs.keys(), "No prune percentage defined"
        # assert 'prune_step' in kwargs.keys(), "No prune step defined"
        # assert 'prune_limit' in kwargs.keys(), "No prune limit defined"
        assert 'prune_layers' in kwargs.keys(), "No layers defined to be pruned"

        assert isinstance(kwargs.get('prune_percent', None), int) and kwargs[
            'prune_percent'] <= 100, "Prune % must be an integer "
        # Slowly raise pruning of layers by prune_step
        # assert isinstance(kwargs.get('prune_step', None), int) and kwargs.get("prune_step", None) < 100, "Prune step must be an integer"
        # Slowly raise pruning of layers up to prune_limit
        # assert isinstance(kwargs.get('prune_limit', None), int) and 100 >= kwargs['prune_limit'] > kwargs[
        #     'prune_percent'], "Prune Limit must be an integer above Prune %"
        assert isinstance(kwargs.get('prune_layers', None), list) \
               and all(isinstance(layer, nn.Module) for layer in kwargs.get('prune_layers', None)), "Must pass a list" \
                                                                                                    " of layers to be" \
                                                                                                    " pruned."
        model = super(ModelPrunerSingle, self).train(**kwargs)

        prune_tuples = []

        for module in kwargs.get("prune_layers"):
            prune_tuples.append((module, "weight"))
            if module.bias is not None:
                prune_tuples.append((module, "bias"))

        print(f"Start global pruning on {len(kwargs.get('prune_layers', None))} layers")
        global_unstructured(parameters=prune_tuples, pruning_method=l1_unstructured(), amount=kwargs['prune_percent'])

        for module, param in prune_tuples:
            self.remove_pruning_params(module, param)
        print(f"Finished global pruning on {len(kwargs.get('prune_layers', None))} layers," \
              f" removed {kwargs.get('prune_percent', None)}% of weights")
        return model


class ModelPrunerIncremental(BaseModelPruner):
    def train(self, **kwargs) -> nn.Module:
        """

        :param kwargs:
        :return:
        """
        assert 'prune_percent' in kwargs.keys(), "No prune percentage defined"
        assert 'prune_step' in kwargs.keys(), "No prune step defined"
        assert 'prune_limit' in kwargs.keys(), "No prune limit defined"
        assert 'prune_layers' in kwargs.keys(), "No layers defined to be pruned"
        assert 'prune_early_stop' in kwargs.keys(), "No early stop for pruning defined"

        assert isinstance(kwargs.get('prune_percent', None), int) and kwargs[
            'prune_percent'] <= 100, "Prune % must be an integer "
        # Slowly raise pruning of layers by prune_step
        assert isinstance(kwargs.get('prune_step', None), int) and kwargs.get("prune_step",
                                                                              None) < 100, "Prune step must be an integer"
        # Slowly raise pruning of layers up to prune_limit
        assert isinstance(kwargs.get('prune_limit', None), int) and 100 >= kwargs['prune_limit'] > kwargs[
            'prune_percent'], "Prune Limit must be an integer above Prune %"
        assert isinstance(kwargs.get('prune_layers', None), list) \
               and all(isinstance(layer, nn.Module) for layer in kwargs.get('prune_layers', None)), "Must pass a list" \
                                                                                                    " of layers to be" \
                                                                                                    " pruned."
        pruning_no_improve = 0
        best_acc = 0

        prune_tuples = []

        for module in kwargs.get("prune_layers"):
            prune_tuples.append((module, "weight"))
            if module.bias is not None:
                prune_tuples.append((module, "bias"))

        print(
            f"Start global pruning on {len(kwargs.get('prune_layers', None))} layers, starting on {kwargs.get('prune_percent', None)}% up to {kwargs.get('prune_limit', None)}")

        for prune_percent in range(kwargs.get('prune_percent', None), kwargs.get('prune_limit', None),
                                   kwargs.get('prune_step', None)):

            if pruning_no_improve > kwargs.get('prune_early_stop', None):
                print(f"Early Stop finish pruning at {prune_percent}% with avg val acc: {best_acc}")
                break

            model = super(ModelPrunerIncremental, self).train(**kwargs)

            print(
                f" ======== Start global pruning on {len(kwargs.get('prune_layers', None))} layers with {prune_percent}% pruning ========")
            global_unstructured(parameters=prune_tuples, pruning_method=l1_unstructured, amount=prune_percent)

            print(f"Start pruning validation process")
            current_val_loss, current_val_acc = self._val_evaluator.run(model=self.model, data_loader=self._val_dl)
            is_best = current_val_loss.avg > best_acc
            best_acc = max(best_acc, current_val_loss.avg)

            if not is_best:
                pruning_no_improve += 1
            else:
                pruning_no_improve = 0

        for module, param in prune_tuples:
            self.remove_pruning_params(module, param)
        print(f"Finished global pruning on {len(kwargs.get('prune_layers', None))} layers," \
              f" removed {kwargs.get('prune_percent', None)}% to {kwargs.get('prune_limit', None)}% of weights iteratively")
        return model


class ModelPrunerIterative(BaseModelPruner):
    def train(self, **kwargs) -> nn.Module:
        """

        :param kwargs:
        :return:
        """
        assert 'prune_percent' in kwargs.keys(), "No prune percentage defined"
        assert 'prune_step' in kwargs.keys(), "No prune step defined"
        assert 'prune_limit' in kwargs.keys(), "No prune limit defined"
        assert 'prune_layers' in kwargs.keys(), "No layers defined to be pruned"
        assert 'prune_early_stop' in kwargs.keys(), "No early stop for pruning defined"
        assert 'num_pruning' in kwargs.keys(), "No number of times for pruning defined"

        assert isinstance(kwargs.get('prune_percent', None), int) and kwargs[
            'prune_percent'] <= 100, "Prune % must be an integer "
        # Slowly raise pruning of layers by prune_step
        assert isinstance(kwargs.get('prune_step', None), int) and kwargs.get("prune_step",
                                                                              None) < 100, "Prune step must be an integer"
        # Slowly raise pruning of layers up to prune_limit
        assert isinstance(kwargs.get('prune_limit', None), int) and 100 >= kwargs['prune_limit'] > kwargs[
            'prune_percent'], "Prune Limit must be an integer above Prune %"
        assert isinstance(kwargs.get('prune_layers', None), list) \
               and all(isinstance(layer, nn.Module) for layer in kwargs.get('prune_layers', None)), "Must pass a list" \
                                                                                                    " of layers to be" \
                                                                                                    " pruned."
        pruning_no_improve = 0
        best_acc = 0

        prune_tuples = []

        for module in kwargs.get("prune_layers"):
            prune_tuples.append((module, "weight"))
            if module.bias is not None:
                prune_tuples.append((module, "bias"))

        print(
            f"Start global pruning on {len(kwargs.get('prune_layers', None))} layers, pruning {kwargs.get('prune_percent', None)}% up to {kwargs.get('num_pruning', None)} times")

        for pruning_epoch in range(kwargs.get('num_pruning', None)):

            if pruning_no_improve > kwargs.get('prune_early_stop', None):
                print(f"Early Stop finish pruning at {prune_percent}% with avg val acc: {best_acc}")
                break

            model = super(ModelPrunerIterative, self).train(**kwargs)

            print(
                f" ======== Start global pruning on {len(kwargs.get('prune_layers', None))} layers with {kwargs['prune_percent']}% pruning ========")
            global_unstructured(parameters=prune_tuples, pruning_method=l1_unstructured, amount=kwargs['prune_percent'])

            print(f"Start pruning validation process")
            current_val_loss, current_val_acc = self._val_evaluator.run(model=self.model, data_loader=self._val_dl)
            is_best = current_val_loss.avg > best_acc
            best_acc = max(best_acc, current_val_loss.avg)

            if not is_best:
                pruning_no_improve += 1
            else:
                pruning_no_improve = 0

        for module, param in prune_tuples:
            self.remove_pruning_params(module, param)
            # TODO: Add pruning evaluation for metrics and graphs
        print(f"Finished global pruning on {len(kwargs.get('prune_layers', None))} layers,"
              f" removed {kwargs.get('prune_percent', None)}% to {kwargs.get('prune_limit', None)}% of weights iteratively")
        return model


class TransferLearningTrainer(BaseModelPruner):
    pass
