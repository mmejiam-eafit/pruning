#  -*- coding: utf-8 -*-
#  Copyright (c) 1, 2022.
#  Author: This Code has been developed by Miguel Mejía (mmejiam@eafit.edu.co)
#  Repository: pruning (git clone git@github.com:mmejiam-eafit/pruning.git)
#  Last Modified: 1/11/22, 10:51 AM by Miguel Mejía
from abc import abstractmethod, ABC
from os import path, makedirs
from typing import List, Union

import matplotlib.pyplot as plt
from numpy import ndarray
from torch import Tensor


class BasePlotter(ABC):
    def __init__(self, save_path: str):
        self._save_path = save_path

    @abstractmethod
    def plot(self, data: Union[ndarray, Tensor, List], **kwargs):
        raise NotImplementedError("Method needs to be implemented on child class")


class StatsPlotter(BasePlotter):

    def plot(self, data: List, **kwargs):
        if kwargs.get('x', None) is None:
            x = None
        else:
            x = kwargs['x']

        if any(isinstance(el, list) for el in data):
            for i, el in enumerate(data):
                x = list(range(len(el))) if x is None else x
                plt.plot(x, el, label=kwargs['label'][i] if kwargs['label'] is not None else '')
        else:
            x = list(range(len(data))) if x is None else x
            plt.plot(x, data, label=kwargs['label'] if kwargs['label'] is not None else '')

        if kwargs.get('title', None) is not None:
            plt.title(kwargs['title'])

        if kwargs.get('ylabel', None) is not None:
            plt.ylabel(kwargs['ylabel'])

        if kwargs.get('xlabel', None) is not None:
            plt.xlabel(kwargs['xlabel'])

        if kwargs.get('legend', None) is not None and kwargs['legend']:
            plt.legend(loc='upper left')

        if kwargs.get('save_graph', None) is not None and kwargs['save_graph']:
            name = kwargs.get('plot_name', None)
            plot_name = name if name is not None else ''
            if not path.exists(self._save_path):
                makedirs(self._save_path, exist_ok=True)
            plt.savefig(path.join(self._save_path, f"{kwargs.get('model_name', None)}{f'_{plot_name}'}_plot.png"))

        plt.show()

# TODO Add class to get heatmaps from neural network
