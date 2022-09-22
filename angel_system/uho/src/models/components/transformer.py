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
#from .vidswin import VideoSwinTransformerBackbone
import pdb


configs = {
    'video_swin_t_p4w7':
        dict(patch_size=(2, 4, 4),
             embed_dim=96,
             depths=[2, 2, 6, 2],
             num_heads=[3, 6, 12, 24],
             window_size=(8, 7, 7),
             mlp_ratio=4.,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             drop_path_rate=0.2,
             patch_norm=True,
             use_checkpoint=False
             ),
    'video_swin_s_p4w7':
        dict(patch_size=(2, 4, 4),
             embed_dim=96,
             depths=[2, 2, 18, 2],
             num_heads=[3, 6, 12, 24],
             window_size=(8, 7, 7),
             mlp_ratio=4.,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             drop_path_rate=0.2,
             patch_norm=True,
             use_checkpoint=False
             ),
    'video_swin_b_p4w7':
        dict(patch_size=(2, 4, 4),
             embed_dim=128,
             depths=[2, 2, 18, 2],
             num_heads=[4, 8, 16, 32],
             window_size=(8, 7, 7),
             mlp_ratio=4.,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             drop_path_rate=0.2,
             patch_norm=True,
             use_checkpoint=False
             )
}

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

        self.use_CLS = False
        self.use_BBOX = True
        self.cls_token = nn.Parameter(torch.zeros(1,1, hidden*2))
        # 2048 -> The length of features out of last layer of ResNext
        self.fc_x = nn.Linear(2048, hidden)
        if self.use_BBOX:
            # 2048 -> The length of features out of Faster R-CNN
            self.fc_d = nn.Linear(2048, hidden*2-12)
            self.fc_b = nn.Linear(4, 12)
        else:
            self.fc_d = nn.Linear(2048, hidden*2)

        # 126 -> 63*2 (Each hand has a descriptor of length 63 compatible with H2O format)
        self.fc_h = nn.Linear(126, hidden)
        '''layers = []
        n_mlp = 3
        for i in range(n_mlp):
            if i == 0:
                in_dim, out_dim = 126, hidden
            else:
                in_dim, out_dim = hidden, hidden
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(0.2, True))
        self.fc_h = nn.Sequential(*layers)'''
        #self.fc_v = nn.Linear(768, hidden*2)

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
        self.loss_action = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.1) #LabelSmoothingLoss(0.1, 37, ignore_index=-1)
        self.apply(self.init_weights)

        # video swin transformer
        #cfgs = configs['video_swin_t_p4w7']
        #self.extr_3d = VideoSwinTransformerBackbone(True, 'checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', False, **cfgs)

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

    def forward(self, inputs):
        topK = 5

        # load features
        feat_x = inputs["feats"] # RGB features
        lh, rh = inputs["labels"]["l_hand"], inputs["labels"]["r_hand"] # hand poses
        feat_d = inputs["dets"] # detections
        feat_b = inputs["bbox"] # bounding boxes

        feat_x = self.fc_x(feat_x)
        feat_h = self.fc_h(torch.cat([lh, rh], axis=-1).float())
        feat_x = torch.cat([feat_x, feat_h], axis=-1)
        num_batch, num_frame, num_dim = feat_x.shape

        if feat_d.ndim == 2:
            feat_d = feat_d.unsqueeze(1)
            feat_b = feat_b.unsqueeze(1)
        feat_d = self.fc_d(feat_d)

        if self.use_BBOX:
            feat_b[:,:,[0,2]] = feat_b[:,:,[0,2]]/1280.
            feat_b[:,:,[1,3]] = feat_b[:,:,[1,3]]/720.
            feat_b = self.fc_b(feat_b.float())
            feat_d = torch.cat([feat_d, feat_b], axis=-1)
        #print(feat_x.shape)
        if self.use_CLS:
            cls_tokens = self.cls_token.expand(num_batch, -1, -1)
            enc_feat = torch.cat([cls_tokens, feat_x, feat_d], axis=1)
        else:
            enc_feat = torch.cat([feat_x, feat_d], axis=1)
        enc_feat = enc_feat.contiguous()

        # add positional encoding
        if self.use_CLS:
            pos_ids = torch.arange(num_frame+1, dtype=torch.long, device=enc_feat.device)
            pos_ids_frame = pos_ids.unsqueeze(0).expand(num_batch, -1)
            pos_ids_det = pos_ids[1:].repeat(topK,1).transpose(1,0).reshape(-1, feat_d.size(1)).expand(num_batch, -1)
            frame_pos_embed = self.frame_pos_embeddings(pos_ids_frame)
            det_pos_embed = self.frame_pos_embeddings(pos_ids_det)
            #pos_embeddings = torch.cat([frame_pos_embed, frame_pos_embed[:,1:,:], det_pos_embed], axis=1)
            pos_embeddings = torch.cat([frame_pos_embed, det_pos_embed], axis=1)
        else:
            pos_ids = torch.arange(num_frame, dtype=torch.long, device=enc_feat.device)
            pos_ids_frame = pos_ids.unsqueeze(0).expand(num_batch, -1)
            pos_ids_det = pos_ids.repeat(topK,1).transpose(1,0).reshape(-1, feat_d.size(1)).expand(num_batch, -1)
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
        if self.use_CLS:
            action_feat = trans_feat[:,0,:]
            #hand_feat = trans_feat[:,num_frame+1:2*num_frame+1,:]
            #detection_feat = trans_feat[:,2*num_frame+1:,:]
            detection_feat = trans_feat[:,num_frame+1:,:]
            action_out = self.classifier_action(action_feat)
        else:
            action_feat = trans_feat[:,:num_frame,:]
            #hand_feat = trans_feat[:,num_frame:2*num_frame,:]
            detection_feat = trans_feat[:,num_frame:,:]#feat_b.size(2)]
            action_out = self.classifier_action(action_feat.mean(dim=1, keepdim=False))

        detection_out = self.classifier_detection(detection_feat)
        action_pred = torch.argmax(action_out, dim=1)
        action_out = torch.softmax(action_out, dim=1)

        return action_out, action_pred

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")
