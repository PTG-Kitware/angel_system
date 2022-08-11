# """

# TODO:
# * Implement losses MSE and CE losses
# * Use hydra to set hyperparams: alpha and dth (distance threshold)
# * Update documentation

# """

# from multiprocessing import Pool
# from typing import Dict, Tuple

# import torch
# from torch import nn
# from torchvision.models import convnext_tiny
# from torchvision.models.feature_extraction import create_feature_extractor

# from utils.grid_utils import compute_corner_confidence, convert2grid

# def _process_batch(data: Tuple):
#     (
#         cp_pred_np,
#         obj_pred_np,
#         l_verb_pred,
#         r_verb_pred,
#         conf_pred_np,
#         l_hand,
#         r_hand,
#         obj_label,
#         obj_pose,
#         verb,
#     ) = data

#     # control points gt

#     # confidence scores for grid
#     alpha = 2.0
#     dth = [75, 7.5]  # For (x, y) (in px) and z (in cm) dims
#     conf = compute_corner_confidence(cp_pred_np, obj_pose, l_hand, r_hand, alpha, dth)

#     # confidence mask

#     return conf

# class UnifiedFCNModule(nn.Module):
#     """Class implements fully convolutional network for extracting spatial features from the video
#     frames.

#     Args: TBD
#     """

#     def __init__(self, net: str, num_cpts: int, obj_classes: int, verb_classes: int):
#         super(UnifiedFCNModule, self).__init__()
#         self.net = self._select_network(net)
#         self.net = create_feature_extractor(
#             self.net, return_nodes={"features.7.2.block.4": "feat_out"}
#         )

#         self.num_cpts = num_cpts
#         self.obj_classes = obj_classes
#         self.verb_classes = verb_classes

#         # 'out_channels' depicts the number of elements in the flattened grid
#         out_channels = 5 * 3 * (3 * self.num_cpts + 1 + self.obj_classes + self.verb_classes)
#         self.fc1 = nn.Linear(3072, out_channels)

#         # train
#         self.lhand_loss = None
#         self.rhand_loss = None
#         self.obj_pose_loss = None
#         self.conf_loss = None
#         self.oclass_loss = None
#         self.vclass_loss = None
#         self.pool = Pool(processes=10)

#     def _select_network(self, net_opt: str) -> nn.Module:
#         net: nn.Module = None
#         if net_opt == "convnext_tiny":
#             net = convnext_tiny(pretrained=True)
#         else:
#             print("NN model not found. Change the feature extractor network.")

#         return net

#     def forward(self, data: Dict):
#         x = data["frm"]
#         out = self.net(x)
#         x = out["feat_out"]
#         x = self.fc1(x)
#         x = x.permute(0, 3, 1, 2)

#         # extracting predictions
#         bsize, _, h, w = x.size()
#         x_reshaped = x.contiguous().view(
#             bsize, -1, 3, 3 * self.num_cpts + 1 + self.obj_classes + self.verb_classes
#         )

#         # vector indices (at position 2): 0 -> object, 1 -> l_hand, 2 -> r_hand
#         cp_pred = torch.sigmoid(x_reshaped[:, :, :, 0 : 3 * self.num_cpts])
#         cp_pred = convert2grid(cp_pred)
#         obj_pred = torch.sigmoid(
#             x_reshaped[:, :, 0, 3 * self.num_cpts + 1 : 3 * self.num_cpts + 1 + self.obj_classes]
#         )
#         l_verb_pred = torch.sigmoid(
#             x_reshaped[
#                 :,
#                 :,
#                 1,
#                 3 * self.num_cpts
#                 + 1
#                 + self.obj_classes : 3 * self.num_cpts
#                 + 1
#                 + self.obj_classes
#                 + self.verb_classes,
#             ]
#         )
#         r_verb_pred = torch.sigmoid(
#             x_reshaped[
#                 :,
#                 :,
#                 2,
#                 3 * self.num_cpts
#                 + 1
#                 + self.obj_classes : 3 * self.num_cpts
#                 + 1
#                 + self.obj_classes
#                 + self.verb_classes,
#             ]
#         )
#         conf_pred = x_reshaped[:, :, :, 3 * self.num_cpts].contiguous()

#         # for training
#         if self.training:
#             cp_pred_np = cp_pred.data.cpu().numpy()
#             obj_pred_np = obj_pred.data.cpu().numpy()
#             l_verb_pred_np = l_verb_pred.data.cpu().numpy()
#             r_verb_pred_np = r_verb_pred.data.cpu().numpy()
#             conf_pred_np = conf_pred.data.cpu().numpy()

#             _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = self._build_target(
#                 cp_pred_np,
#                 obj_pred_np,
#                 l_verb_pred_np,
#                 r_verb_pred_np,
#                 conf_pred_np,
#                 data["l_hand"],
#                 data["r_hand"],
#                 data["obj_label"],
#                 data["obj_pose"],
#                 data["verb"],
#             )

#             _boxes = net_utils.np_to_variable(_boxes)
#             _ious = net_utils.np_to_variable(_ious)
#             _classes = net_utils.np_to_variable(_classes)
#             box_mask = net_utils.np_to_variable(_box_mask, dtype=torch.FloatTensor)
#             iou_mask = net_utils.np_to_variable(_iou_mask, dtype=torch.FloatTensor)
#             class_mask = net_utils.np_to_variable(_class_mask, dtype=torch.FloatTensor)

#             num_boxes = sum((len(boxes) for boxes in gt_boxes))

#             # _boxes[:, :, :, 2:4] = torch.log(_boxes[:, :, :, 2:4])
#             box_mask = box_mask.expand_as(_boxes)

#             self.bbox_loss = (
#                 nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes
#             )  # noqa
#             self.iou_loss = (
#                 nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes
#             )  # noqa

#             class_mask = class_mask.expand_as(prob_pred)
#             self.cls_loss = (
#                 nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask)
#                 / num_boxes
#             )  # noqa

#         return bbox_pred, iou_pred, prob_pred

#     def _build_target(
#         self,
#         cp_pred_np,
#         obj_pred_np,
#         l_verb_pred,
#         r_verb_pred,
#         conf_pred_np,
#         l_hand,
#         r_hand,
#         obj_label,
#         obj_pose,
#         verb,
#     ):
#         """Convert the labels and predictions to grid coordinates/format."""

#         bsize = cp_pred_np.shape[0]

#         targets = self.pool.map(
#             _process_batch,
#             (
#                 (
#                     cp_pred_np[b],
#                     obj_pred_np[b],
#                     l_verb_pred[b],
#                     r_verb_pred[b],
#                     conf_pred_np[b],
#                     l_hand[b],
#                     r_hand[b],
#                     obj_label[b],
#                     obj_pose[b],
#                     verb[b],
#                 )
#                 for b in range(bsize)
#             ),
#         )

#         _cpts = np.stack(tuple((row[0] for row in targets)))
#         _ious = np.stack(tuple((row[1] for row in targets)))
#         _classes = np.stack(tuple((row[2] for row in targets)))
#         _box_mask = np.stack(tuple((row[3] for row in targets)))
#         _iou_mask = np.stack(tuple((row[4] for row in targets)))
#         _class_mask = np.stack(tuple((row[5] for row in targets)))

#         return _cpts, _ious, _classes, _box_mask, _iou_mask, _class_mask
