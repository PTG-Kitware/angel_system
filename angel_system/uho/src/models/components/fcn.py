from collections import OrderedDict
from multiprocessing import Pool
from typing import Dict, Tuple

import torch
from torch import nn
from torchvision.models import resnext50_32x4d


class UnifiedFCNModule(nn.Module):
    """Class implements fully convolutional network for extracting spatial
    features from the video frames."""

    def __init__(self, net: str):
        super(UnifiedFCNModule, self).__init__()

        self.output_layers = [8]  # 8 -> Avg. pool layer
        self.selected_out = OrderedDict()
        self.net = self._select_network(net)
        # Freeze network weights
        for param in self.net.parameters():
            param.requires_grad = False

        self.fhooks = []
        for i, l in enumerate(list(self.net._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(
                    getattr(self.net, l).register_forward_hook(self.forward_hook(l))
                )

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook

    def _select_network(self, net_opt: str) -> nn.Module:
        net: nn.Module = None
        if net_opt == "resnext":
            net = resnext50_32x4d(pretrained=True)
        else:
            print("NN model not found. Change the feature extractor network.")

        return net

    def forward(self, x):
        out = self.net(x)
        x = self.selected_out["avgpool"].reshape(-1, 2048)

        return x
