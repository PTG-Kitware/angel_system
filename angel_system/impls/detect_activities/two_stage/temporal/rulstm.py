import pdb

from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class OpenLSTM(nn.Module):
    """"An LSTM implementation that returns the intermediate hidden and cell states.
    The original implementation of PyTorch only returns the last cell vector.
    For RULSTM, we want all cell vectors computed at intermediate steps"""
    def __init__(self, feat_in, feat_out, num_layers=1, dropout=0):
        """
            feat_in: input feature size
            feat_out: output feature size
            num_layers: number of layers
            dropout: dropout probability
        """
        super(OpenLSTM, self).__init__()

        # simply create an LSTM with the given parameters
        self.lstm = nn.LSTM(feat_in, feat_out, num_layers=num_layers, dropout=dropout)

    def forward(self, seq):
        # manually iterate over each input to save the individual cell vectors
        last_cell=None
        last_hid=None
        hid = []
        cell = []
        for i in range(seq.shape[0]):
            el = seq[i,...].unsqueeze(0)
            if last_cell is not None:
                _, (last_hid, last_cell) = self.lstm(el, (last_hid,last_cell))
            else:
                _, (last_hid, last_cell) = self.lstm(el)
            hid.append(last_hid)
            cell.append(last_cell)

        return torch.stack(hid, 0),  torch.stack(cell, 0)

class RULSTM(nn.Module):
    def __init__(self, act_classes, hidden, dropout=0.8, depth=1, 
            sequence_completion=False, return_context=False):
        """
            act_classes: number of classes
            hidden: number of hidden units
            dropout: dropout probability
            depth: number of LSTM layers
            sequence_completion: if the network should be arranged for sequence completion pre-training
            return_context: whether to return the Rolling LSTM hidden and cell state (useful for MATT) during forward
        """
        super(RULSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden=hidden
        self.fc1 = nn.Linear(2048, hidden)
        self.fc_h = nn.Linear(126, hidden)
        self.rolling_lstm = OpenLSTM(2*hidden, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.unrolling_lstm = nn.LSTM(2*hidden, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, act_classes))
        self.sequence_completion = sequence_completion

        # Loss functions
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # permute the inputs for compatibility with the LSTM
        x = inputs
 
        # Enable for passing hand pose information
        # lh, rh = inputs[0]["labels"]["l_hand"], inputs[0]["labels"]["r_hand"]
        # h = self.fc_h(torch.cat([lh, rh], axis=-1).float())
        x = self.fc1(x)
        x = torch.cat([x, x], axis=-1)  # Placeholder for hand pose modality
        # x = torch.cat([x, h], axis=-1)  # Concatenate with h for hand pose modality

        # pass the frames through the rolling LSTM
        # and get the hidden (x) and cell (c) states at each time-step
        h, c = self.rolling_lstm(self.dropout(x))
        h = h.contiguous() # batchsize x timesteps x hidden
        c = c.contiguous() # batchsize x timesteps x hidden

        # accumulate the predictions in a list
        predictions = [] # accumulate the predictions in a list
        
        # for each time-step
        for t in range(h.shape[0]):
            # get the hidden and cell states at current time-step
            hid = h[t,...]
            cel = c[t,...]

            if self.sequence_completion:
                # take current + future inputs (looks into the future)
                ins = x[t:,...]
            else:
                # replicate the current input for the correct number of times (time-steps remaining to the beginning of the action)
                ins = x[t,...].unsqueeze(0).expand(x.shape[0]-t+1,x.shape[1],x.shape[2]).to(x.device)
            
            # initialize the LSTM and iterate over the inputs
            h_t, (_,_) = self.unrolling_lstm(self.dropout(ins), (hid.contiguous(), cel.contiguous()))
            # get last hidden state
            h_n = h_t[-1,...]

            # append the last hidden state to the list
            predictions.append(h_n)
        
        # obtain the final prediction tensor by concatenating along dimension 1
        x = torch.stack(predictions,1)

        # apply the classifier to each output feature vector (independently)
        y = self.classifier(x.view(-1,x.size(2))).view(x.size(0), x.size(1), -1)
        # pdb.set_trace()
        out = y[:, -1, :]

        return out