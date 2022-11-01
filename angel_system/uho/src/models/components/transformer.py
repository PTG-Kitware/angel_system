import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import constant, normal
import time
import math
from functools import reduce
import torchvision.models as models
import sklearn.utils.class_weight as class_weight
from collections import OrderedDict
import pdb

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class TemTRANSModule(nn.Module):
    def __init__(self, 
        act_classes,
        hidden,
        dropout=0.,
        depth=4,
        num_head=8):
        super().__init__()
        self._det_topk = 10

        # 2048 -> The length of features out of last layer of ResNext
        self.fc_x = nn.Linear(2048, hidden)
        # 2048 -> The length of features out of Faster R-CNN
        self.fc_d = nn.Linear(2048, hidden*2-12)
        self.fc_b = nn.Linear(4, 12)
        # 126 -> 63*2 (Each hand has a descriptor of length 63 compatible with H2O format)        
        self.fc_h = nn.Linear(126, hidden)

        context_length = 77
        det_classes = 1600
        self.frame_pos_embeddings = nn.Embedding(context_length, hidden*2)
        self.transformer = TemporalTransformer(width=hidden*2, layers=depth, heads=num_head)
        self.classifier_action = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, act_classes))
        self.classifier_detection = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, det_classes))
        # Loss functions
        self.loss_detection = nn.CrossEntropyLoss(label_smoothing=0.1)
        class_weight = torch.ones(act_classes)
        class_weight[0] = 2./float(act_classes)
        self.loss_action = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.1) 
        self.apply(self.init_weights)

    @property
    def det_topk(self):
        return self._det_topk

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, inputs, if_train):
        # load features       
        feat_x = inputs[0]["feats"] # RGB features
        lh, rh = inputs[0]["labels"]["l_hand"], inputs[0]["labels"]["r_hand"] # hand poses
        feat_d = inputs[0]["dets"] # detection features
        feat_b = inputs[0]["bbox"] # bounding boxes
        det_labels = inputs[0]["dcls"] # detections labels
        labels = inputs[0]["act"] # action labels
        idx = inputs[0]["idx"] # video clip index

        # simulation: select detections at arbitrary frames (training: 4~32; testing 4~8)
        max_sample_fr = feat_x.shape[1] if if_train else 8
        num_det_fr = np.random.randint(max_sample_fr-3)+4
        valid_fr = np.random.permutation(feat_x.shape[1])[:num_det_fr]
        valid_fr = np.sort(valid_fr)

        # sample detections
        topK = self._det_topk
        num_batch, num_det, num_dim = feat_d.shape
        feat_d = [feat_d[:,k*topK:(k+1)*topK,:] for k in valid_fr]
        feat_d = torch.stack(feat_d,1).reshape(num_batch, -1, num_dim)
        feat_b = [feat_b[:,k*topK:(k+1)*topK,:] for k in valid_fr]
        feat_b = torch.stack(feat_b,1).reshape(num_batch, -1, 4)
        det_labels = [det_labels[:,k*topK:(k+1)*topK] for k in valid_fr]
        det_labels = torch.stack(det_labels,1).reshape(num_batch, -1)

        # convert features
        feat_x = self.fc_x(feat_x)
        feat_h = self.fc_h(torch.cat([lh, rh], axis=-1).float())
        feat_x = torch.cat([feat_x, feat_h], axis=-1)
        num_batch, num_frame, num_dim = feat_x.shape

        feat_d = self.fc_d(feat_d)
        feat_b[:,:,[0,2]] = feat_b[:,:,[0,2]]/1280.
        feat_b[:,:,[1,3]] = feat_b[:,:,[1,3]]/720.
        feat_b = self.fc_b(feat_b.float())
        feat_d = torch.cat([feat_d, feat_b], axis=-1)

        enc_feat = torch.cat([feat_x, feat_d], axis=1)
        enc_feat = enc_feat.contiguous()

        # add positional encoding
        pos_ids = torch.arange(num_frame, dtype=torch.long, device=enc_feat.device)
        pos_ids_frame = pos_ids.unsqueeze(0).expand(num_batch, -1)
        pos_ids_det = pos_ids[valid_fr].repeat(topK,1).transpose(1,0).reshape(-1, feat_d.size(1)).expand(num_batch, -1)
        frame_pos_embed = self.frame_pos_embeddings(pos_ids_frame)
        det_pos_embed = self.frame_pos_embeddings(pos_ids_det)    
        pos_embeddings = torch.cat([frame_pos_embed, det_pos_embed], axis=1)
        trans_feat = enc_feat + pos_embeddings

        # calculate attentions
        trans_feat = trans_feat.permute(1, 0, 2)  # NLD -> LND
        trans_feat = self.transformer(trans_feat)
        trans_feat = trans_feat.permute(1, 0, 2)  # LND -> NLD
        trans_feat = trans_feat.type(enc_feat.dtype) + enc_feat

        # classification
        action_feat = trans_feat[:,:num_frame,:]
        detection_feat = trans_feat[:,num_frame:,:]
        action_out = self.classifier_action(action_feat.mean(dim=1, keepdim=False))

        action_pred = torch.argmax(action_out, dim=1)
        action_loss = self.loss_action(action_out, labels.long())
        detection_out = self.classifier_detection(detection_feat)
        detection_loss = self.loss_detection(detection_out.permute(0,2,1), det_labels.long())
        action_out = torch.softmax(action_out, dim=1)
        loss = action_loss + 0.025*detection_loss

        return action_out, action_pred, loss, idx

    def predict(self, inputs):
        topK = self._det_topk
        # load features       
        feat_x = inputs["feats"] # RGB features
        lh, rh = inputs["lhand"], inputs["rhand"] # hand poses
        feat_d = inputs["dets"] # detection features
        feat_b = inputs["bbox"] # bounding boxes

        # extract detections at valid frames
        num_batch, num_det, num_dim = feat_d.shape
        tmp = torch.sum(feat_b, 2)
        valid_fr = [k for k in range(feat_x.shape[1]) if torch.sum(tmp[:,k*topK:(k+1)*topK]) != 0]

        feat_d = [feat_d[:,k*topK:(k+1)*topK,:] for k in valid_fr]
        feat_d = torch.stack(feat_d,1).reshape(num_batch, -1, num_dim)
        feat_b = [feat_b[:,k*topK:(k+1)*topK,:] for k in valid_fr]
        feat_b = torch.stack(feat_b,1).reshape(num_batch, -1, 4)

        # convert features
        feat_x = self.fc_x(feat_x)
        feat_h = self.fc_h(torch.cat([lh, rh], axis=-1).float())
        feat_x = torch.cat([feat_x, feat_h], axis=-1)
        num_batch, num_frame, num_dim = feat_x.shape

        feat_d = self.fc_d(feat_d)
        feat_b[:,:,[0,2]] = feat_b[:,:,[0,2]]/1280.
        feat_b[:,:,[1,3]] = feat_b[:,:,[1,3]]/720.
        feat_b = self.fc_b(feat_b.float())
        feat_d = torch.cat([feat_d, feat_b], axis=-1)

        enc_feat = torch.cat([feat_x, feat_d], axis=1)
        enc_feat = enc_feat.contiguous()

        # add positional encoding
        pos_ids = torch.arange(num_frame, dtype=torch.long, device=enc_feat.device)
        pos_ids_frame = pos_ids.unsqueeze(0).expand(num_batch, -1)
        pos_ids_det = pos_ids[valid_fr].repeat(topK,1).transpose(1,0).reshape(-1, feat_d.size(1)).expand(num_batch, -1)
        frame_pos_embed = self.frame_pos_embeddings(pos_ids_frame)
        det_pos_embed = self.frame_pos_embeddings(pos_ids_det)    
        pos_embeddings = torch.cat([frame_pos_embed, det_pos_embed], axis=1)
        trans_feat = enc_feat + pos_embeddings

        # calculate attentions
        trans_feat = trans_feat.permute(1, 0, 2)  # NLD -> LND
        trans_feat = self.transformer(trans_feat)
        trans_feat = trans_feat.permute(1, 0, 2)  # LND -> NLD
        trans_feat = trans_feat.type(enc_feat.dtype) + enc_feat

        # classification
        action_feat = trans_feat[:,:num_frame,:]
        action_out = self.classifier_action(action_feat.mean(dim=1, keepdim=False))
        action_pred = torch.argmax(action_out, dim=1)
        action_out = torch.softmax(action_out, dim=1)

        return action_out, action_pred
