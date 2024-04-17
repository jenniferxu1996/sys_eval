import torch
import torch.nn as nn

"""
This file includes all the node RNN in the architecture
"""


class SegmentNodeRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        # the required sizes in the forward
        self.args = args

        self.node_feature_size = args.current_node_feature_size
        # the edge encode size is the size of features of the sum of the temporal edge hidden state and the weighted sum of the spatial edge hidden state
        self.edge_encode_size = args.edge_weighted_hidden_size_spatial + args.edge_hidden_size_temporal
        self.node_embedding_size = args.node_embedding_size
        self.edge_embedding_size = args.edge_embedding_size
        self.node_hidden_size = args.node_hidden_size
        self.output_size = args.node_output_size

        self.BatchNorm = nn.BatchNorm1d(args.current_node_feature_size)
        self.BatchNorm_2 = nn.BatchNorm1d(args.node_embedding_size)

        # the embedding layers for 1) nodes features and 2) edges hidden state
        self.node_linear = nn.Linear(self.node_feature_size, self.node_embedding_size)
        self.edge_linear = nn.Linear(self.edge_encode_size, self.edge_embedding_size)

        # the node RNN cell
        self.node_rnn_cell = nn.LSTMCell(self.node_embedding_size + self.edge_embedding_size, self.node_hidden_size)

        # the output linear layer
        self.output_linear = nn.Linear(self.node_hidden_size, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, true_batch_size, node_feature, h_temporal, h_spatial, h0, c0):
        # encode the node feature
        if true_batch_size > 1:
            node_feature = self.BatchNorm(node_feature)
        node_embed = self.node_linear(node_feature)
        # if true_batch_size > 1:
        #     node_embed = self.BatchNorm_2(node_embed)
        node_embed = self.relu(node_embed)


        # encode the edge feature
        edge_feature = torch.cat((h_temporal, h_spatial), -1)
        edge_embed = self.edge_linear(edge_feature)
        edge_embed = self.relu(edge_embed)

        # concat the node embedding and the edge embedding
        concat_embed = torch.cat((node_embed, edge_embed), -1)

        # operation on a rnn cell
        if h0 is None and c0 is None:
            h1, c1 = self.node_rnn_cell(concat_embed)
        else:
            h1, c1 = self.node_rnn_cell(concat_embed, (h0, c0))
        # output the hidden state
        h1 = self.relu(h1)
        output = self.output_linear(h1)
        output = self.relu(output)
        return output, h1, c1



