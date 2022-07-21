"""

TODO:
* Update documentation

"""

import pdb
from collections import OrderedDict
from typing import Dict

import torch
from torch import nn
from torchvision.models import resnext50_32x4d  # ,convnext_tiny

# from torchvision.models.feature_extraction import create_feature_extractor


class SpatialFCNModule(nn.Module):
    """Class implements fully convolutional network for extracting spatial
    features from the video frames.

    Args: TBD
    """

    def __init__(self, net: str):
        super(SpatialFCNModule, self).__init__()
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

        # loss function
        self.lhand_loss = None
        self.rhand_loss = None
        self.obj_pose_loss = None
        self.conf_loss = None
        self.oclass_loss = None
        self.vclass_loss = None

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook

    def _select_network(self, net_opt: str) -> nn.Module:
        net: nn.Module = None
        if net_opt == "resnext":
            net = resnext50_32x4d(pretrained=True)
            self.output_layers = [8]  # 8 -> Avg. pool layer
        else:
            print("NN model not found. Change the feature extractor network.")

        return net

    def forward(self, x: torch.Tensor):
        # pdb.set_trace()
        bsize = x.shape[0]
        out = self.net(x)
        x = self.selected_out["avgpool"].reshape(bsize, -1)

        return x
