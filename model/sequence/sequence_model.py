import torch
from torch import nn


class SequentialModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_dim = args.n_fglobal
        self.hidden_size = args.sequence_hidden_size
        self.sequence_layer = args.sequence_layer
        self.output_size = args.class_num
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_size, num_layers=self.sequence_layer)
        self.classifer = nn.Linear(self.hidden_size * self.sequence_layer, self.output_size)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(args.drop_out)

    def forward(self, sequence, true_batch_size):
        output, (h_n, c_n) = self.lstm(sequence)
        h_n = h_n.permute(1, 0, 2).contiguous().view(true_batch_size, -1)
        out = self.relu(h_n)
        out = self.classifer(out)
        out = torch.softmax(out, dim=-1)
        return out

    def extract_feature(self, sequence, true_batch_size):
        output, (h_n, c_n) = self.lstm(sequence)
        h_n = h_n.permute(1, 0, 2).contiguous().view(true_batch_size, -1)
        out = self.relu(h_n)
        out = self.drop_out(out)
        # print(f"the true batch size is {true_batch_size}")
        # print(f"the shape of global output is {out.shape}")
        return out

