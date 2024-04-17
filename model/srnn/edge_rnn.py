import torch
import torch.nn as nn


class SpatialEdgeRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # the required sizes in the forward
        self.edge_feature_size = args.edge_feature_size
        self.edge_hidden_size_spatial = args.edge_hidden_size_spatial

        self.BatchNorm = nn.BatchNorm1d(args.edge_feature_size)

        # the rnn cell
        self.edge_rnn_cell = nn.LSTMCell(self.edge_feature_size, self.edge_hidden_size_spatial)

        self.relu = nn.ReLU()

    def forward(self, true_batch_size, edge_feature, h0, c0):
        # if true_batch_size > 1:
        #     edge_feature = self.BatchNorm(edge_feature)
        if h0 is None and c0 is None:
            h1, c1 = self.edge_rnn_cell(edge_feature)
        else:
            # edge_feature = self.BatchNorm(edge_feature)
            h1, c1 = self.edge_rnn_cell(edge_feature, (h0, c0))
        h1 = self.relu(h1)
        return h1, c1


class TemporalEdgeRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # the required sizes in the forward
        self.edge_feature_size = args.edge_feature_size
        self.edge_hidden_size_temporal = args.edge_hidden_size_temporal

        self.BatchNorm = nn.BatchNorm1d(args.edge_feature_size)

        # the rnn cell
        self.edge_rnn_cell = nn.LSTMCell(self.edge_feature_size, self.edge_hidden_size_temporal)

        self.relu = nn.ReLU()

    def forward(self, true_batch_size, edge_feature, h0, c0):
        # if true_batch_size > 1:
        #     edge_feature = self.BatchNorm(edge_feature)
        if h0 is None and c0 is None:

            h1, c1 = self.edge_rnn_cell(edge_feature)
        else:
            # edge_feature = self.BatchNorm(edge_feature)
            h1, c1 = self.edge_rnn_cell(edge_feature, (h0, c0))
        h1 = self.relu(h1)
        return h1, c1


