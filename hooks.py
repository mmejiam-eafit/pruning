# TODO Add function to register forward hooks for layer activation

"""
Links of Interest:
    https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
    https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    https://medium.com/analytics-vidhya/pytorch-hooks-5909c7636fb
    https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
"""
from collections import defaultdict
from typing import Callable, Dict

from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


def get_activation_mask(module: Module, input: Tensor, output: Tensor):
    # my_in = input.detach().cpu()

    for named_buffer, buffer_tensor in module.named_buffers():
        if named_buffer == "weight_mask":
            mask = buffer_tensor.detach().cpu().numpy()
            print(mask.shape)


def get_propagated_info(module: Module, input: Tensor, output: Tensor):
    my_out = output.cpu().detach().numpy
    print(my_out.shape)


def register_hook(module: Module, fn: Callable) -> RemovableHandle:
    return module.register_forward_hook(fn)


class HookManager(object):
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
