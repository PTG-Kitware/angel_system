import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import constant, normal
import time
import math
from functools import reduce
import torchvision.models as models
from collections import OrderedDict
from .vidswin import VideoSwinTransformerBackbone
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


def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

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
        num_head=4):
        super().__init__()

        # 2048 -> The length of features out of last layer of ResNext
        self.fc_x = nn.Linear(2048, hidden)
        self.fc_d = nn.Linear(2048, hidden*2)
        # 126 -> 63*2 (Each hand has a descriptor of length 63 compatible with H2O format)        
        self.fc_h = nn.Linear(126, hidden)
        self.fc_v = nn.Linear(768, hidden*2)
        context_length = 77
        det_classes = 1600
        self.frame_position_embeddings = nn.Embedding(context_length, hidden*2)
        self.transformer = TemporalTransformer(width=hidden*2, layers=depth, heads=num_head)
        self.classifier_action = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, act_classes))
        self.classifier_detection = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, det_classes))
        # Loss functions
        self.loss = nn.CrossEntropyLoss()
        self.loss_func = LabelSmoothingLoss(0.1, 37, ignore_index=-1)
        self.apply(self.init_weights)

        # video swin transformer
        cfgs = configs['video_swin_t_p4w7']
        self.extr_3d = VideoSwinTransformerBackbone(True, 'checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', False, **cfgs)

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
        # load features
        clip = inputs[0]["frms"]
        out_3d = self.extr_3d(clip)
        
        '''feat_x = inputs[0]["feats"]
        print(feat_x.shape)
        pdb.set_trace()

        feat_d = inputs[0]["dets"]
        feat_x = self.fc_x(feat_x)
        lh, rh = inputs[0]["labels"]["l_hand"], inputs[0]["labels"]["r_hand"]
        feat_h = self.fc_h(torch.cat([lh, rh], axis=-1).float())
        if feat_d.ndim == 2: feat_d = feat_d.unsqueeze(1)
        feat_d = self.fc_d(feat_d)
        feat_x = torch.cat([feat_x, feat_h], axis=-1)
        num_batch, num_frame, num_dim = feat_x.shape

        enc_feat = torch.cat([feat_x, feat_d], axis=1)
        enc_feat = enc_feat.contiguous()
        # add positive encoding
        position_ids = torch.arange(num_frame, dtype=torch.long, device=enc_feat.device)
        position_ids_frame = position_ids.unsqueeze(0).expand(feat_x.size(0), -1)
        position_ids_det = position_ids.repeat(4,1).transpose(1,0).reshape(-1, feat_d.size(1)).expand(feat_d.size(0), -1)

        frame_position_embeddings = self.frame_position_embeddings(position_ids_frame)
        det_position_embeddings = self.frame_position_embeddings(position_ids_det)
        position_embeddings = torch.cat([frame_position_embeddings, det_position_embeddings], axis=1)
        trans_feat = enc_feat + position_embeddings

        # calculate attentions
        trans_feat = trans_feat.permute(1, 0, 2)  # NLD -> LND
        trans_feat = self.transformer(trans_feat)
        trans_feat = trans_feat.permute(1, 0, 2)  # LND -> NLD
        trans_feat = trans_feat.type(enc_feat.dtype) + enc_feat

        app_feat = trans_feat[:,:num_frame,:]
        det_feat = trans_feat[:,num_frame:,:]'''
        app_feat = self.fc_v(out_3d)
        # apply the classifier to each output feature vector (independently)
        action_out = self.classifier_action(app_feat.mean(dim=1, keepdim=False))
        #detection_out = self.classifier_detection(det_feat)
        action_pred = torch.argmax(action_out, dim=1)
        loss = 0.1*self.loss_func(action_out, inputs[0]["act"].long())/4.
        #action_loss = 0.1*self.loss_func(action_out, inputs[0]["act"].long())/4.
        #detection_loss = 0.1*self.loss(detection_out.permute(0,2,1), inputs[0]["dcls"].long())/4.
        #loss = action_loss + detection_loss

        return action_out, action_pred, loss


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
        # one_hot[self.ignore_index] = 0
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
