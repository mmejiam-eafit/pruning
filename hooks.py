# TODO Add function to register forward hooks for layer activation

"""
Links of Interest:
    https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
    https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    https://medium.com/analytics-vidhya/pytorch-hooks-5909c7636fb
    https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict
from uuid import uuid4

import numpy as np
import os
from torch import Tensor
from torch.nn import Module
from torch import sum as t_sum
from torch.utils.hooks import RemovableHandle
from torchvision.utils import save_image, make_grid
from datetime import datetime

def save_convolution_filters(module: Module, input: Tensor, output: Tensor):
    weights = module.weight.detach().cpu()

    n, c, w, h = weights.shape

    tensor = weights.view(n * c, -1, w, h) if c == 3 else weights[:, c, :, :].unsqueeze(dim=1)
    weight_grid = make_grid(tensor, nrow=8, normalize=True, scale_each=True)
    save_image(weight_grid, fp=f"./weights_module_{int(time())}", format="png")


def get_activation_mask(module: Module, input: Tensor, output: Tensor):
    # my_in = input.detach().cpu()

    for named_buffer, buffer_tensor in module.named_buffers():
        if named_buffer == "weight_mask":
            mask = buffer_tensor.detach().cpu()
            weights = module.weight.detach().cpu()

            weight_grid = make_grid(weights, nrow=8, normalize=True, scale_each=True)
            save_image(weight_grid, fp=f"./weights_module_{int(time())}", format="png")
            if mask.min().item() == 0:
                mask_grid = make_grid(mask, nrow=8, normalize=True, scale_each=True)

                save_image(mask_grid, fp=f"./mask_module_{int(time())}", format="png")


def get_propagated_info_generator_function(module_str: str, image_dir: str, images_per_batch: int) -> Callable:
    # TODO Add string to determine what pruning method we're using, if any
    def get_propagated_info(module: Module, t_input: Tensor, t_output: Tensor):
        my_in = t_input[0].clone().detach().cpu()
        my_out = t_output.clone().detach().cpu()

        def save_tensor_as_image(my_tensor: Tensor, image_name: str, batch_size: int):
            for tensor in [t for index, t in enumerate(my_tensor) if index % int(batch_size / images_per_batch) == 0]:
                grid = make_grid(tensor.unsqueeze(dim=1), nrow=16, normalize=True, scale_each=True)
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                save_image(grid, fp=os.path.join(image_dir, f"{image_name}.png"), format="png")

        uid = uuid4().hex
        today = datetime.today().strftime("%Y%M%d_%H%m%S")

        save_tensor_as_image(my_in, image_name=f"{module_str}_{uid}_input_{today}", batch_size=my_in.size()[0])
        save_tensor_as_image(my_out, image_name=f"{module_str}_{uid}_output_{today}", batch_size=my_out.size()[0])

    return get_propagated_info


def register_hook(module: Module, fn: Callable) -> RemovableHandle:
    return module.register_forward_hook(fn)


class BaseHook(ABC):

    def __init__(self, module_name: str, module: Module):
        self.module_name = module_name
        self.module = module

    def create_hook(self, module: Module):
        return self.hook

    @abstractmethod
    def hook(self, module: Module, input: Tensor, output: Tensor):
        pass


class HookManager:
    _hooks: Dict[str, RemovableHandle]

    def __init__(self):
        self._hooks = defaultdict(RemovableHandle)

    def addHook(self, module: Module, hook_name: str, fn: Callable) -> bool:
        if hook_name in self._hooks.keys():
            print(f"Hook handle {hook_name} already exists")
            return False

        self._hooks[hook_name] = module.register_forward_hook(fn)
        return True

    def removeHook(self, hook_name: str) -> bool:
        if hook_name not in self._hooks.keys():
            print(f"Hook handle {hook_name} does not exist")
            return False
        self._hooks[hook_name].remove()
        return True

    def removeAllHooks(self):
        for name, hook in self._hooks.items():
            hook.remove()
            del self._hooks[name]
