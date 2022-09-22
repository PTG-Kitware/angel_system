import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import constant, normal
import time
import math
from functools import reduce
import torchvision.models as models
import pdb


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, x):
        bt, n, c = x.size() 
        key = self.key_embedding(x)
        query = self.query_embedding(x)
        value = self.value_embedding(x)
        att, _ = self.attention(query, key, value)
        output = self.output_linear(att)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.linears = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.linears(x)
        return x


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, num_head=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.norm1(x)
        x = x + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)
        
        return x

class transformer(nn.Module):
    def __init__(
        self,
        act_classes,
        hidden,
        dropout=0.,
        depth=4,
        num_head=4
    ):
        """
        act_classes: number of classes
        hidden: number of hidden units
        dropout: dropout probability
        depth: number of transformer layers
        """
        super(transformer, self).__init__()

        self.position_embeddings = PositionEncoding(n_filters=hidden*2, max_len=500)

        # 2048 -> The length of features out of last layer of ResNext
        self.fc_x = nn.Linear(2048, hidden)
        self.fc_d = nn.Linear(2048, hidden*2)
        # 126 -> 63*2 (Each hand has a descriptor of length 63 compatible with H2O format)        
        self.fc_h = nn.Linear(126, hidden)
        
        blocks = []
        for _ in range(depth):
            blocks.append(TransformerBlock(hidden=hidden*2, num_head=num_head, dropout=dropout))
        self.transformer = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2, act_classes))
        # Loss functions
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # load features
        feat_x = inputs[0]["feats"]
        feat_d = inputs[0]["dets"]
        batch_size = feat_x.shape[1]
        feat_x = self.fc_x(feat_x)
        lh, rh = inputs[0]["labels"]["l_hand"], inputs[0]["labels"]["r_hand"]
        feat_h = self.fc_h(torch.cat([lh, rh], axis=-1).float())
        if feat_d.ndim == 2: feat_d = feat_d.unsqueeze(1)
        feat_d = self.fc_d(feat_d)
        feat_x = torch.cat([feat_x, feat_h], axis=-1)
        enc_feat = feat_x#torch.cat([feat_x, feat_d], axis=1)
        enc_feat = enc_feat.permute(1,0,2).contiguous()
        # add positive encoding
        enc_feat = self.position_embeddings(enc_feat)
        # calculate attentions
        num_batch, num_frame, num_dim = feat_x.shape
        trans_feat = self.transformer(enc_feat)
        enc_feat = enc_feat + trans_feat
        app_feat = enc_feat[:,:num_frame,:]
        det_feat = enc_feat[:,num_frame:,:]
        # apply the classifier to each output feature vector (independently)
        y = self.classifier(app_feat.reshape(-1, num_dim)).reshape(num_batch, num_frame, -1)
        out = torch.mean(y, dim=1)
        pred = torch.argmax(out, dim=1)

        loss = self.loss(out, inputs[0]["act"].long())/4.
        #pdb.set_trace()
        return out, pred, loss
