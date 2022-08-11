from collections import OrderedDict
from multiprocessing import Pool
from typing import Dict, Tuple

import torch
from torch import nn
from torchvision.models import resnext50_32x4d  # ,convnext_tiny

# from torchvision.models.feature_extraction import create_feature_extractor


class UnifiedFCNModule(nn.Module):
    """Class implements fully convolutional network for extracting spatial
    features from the video frames."""

    def __init__(self, net: str, num_cpts: int, obj_classes: int, verb_classes: int):
        super(UnifiedFCNModule, self).__init__()
        self.num_cpts = num_cpts
        self.obj_classes = obj_classes
        self.verb_classes = verb_classes

        self.output_layers = [8]  # 8 -> Avg. pool layer
        self.selected_out = OrderedDict()
        self.net = self._select_network(net)
        # Freeze network weights
        for param in self.net.parameters():
            param.requires_grad = False

        self.fhooks = []
        # 2048 -> The length of features out of last layer of ResNext
        self.fc1 = nn.Linear(2048, self.obj_classes + self.verb_classes)
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
        self.oclass_loss = nn.CrossEntropyLoss()
        self.vclass_loss = nn.CrossEntropyLoss()

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

    def forward(self, data: Dict):
        x = data["frm"]
        out = self.net(x)
        x = self.selected_out["avgpool"].reshape(-1, self.fc1.in_features)
        ov_preds = self.fc1(x)

        if "obj_label" in data.keys() and "verb" in data.keys():
            loss = dict(
                obj_loss=self.oclass_loss(ov_preds[:, : self.obj_classes], data["obj_label"]),
                verb_loss=self.vclass_loss(ov_preds[:, self.obj_classes :], data["verb"]),
            )
            ov_preds = dict(
                obj=torch.argmax(ov_preds[:, : self.obj_classes], dim=1),
                verb=torch.argmax(ov_preds[:, self.obj_classes :], dim=1),
            )
            return x, ov_preds, loss
        else:
            return x, {}, 0
