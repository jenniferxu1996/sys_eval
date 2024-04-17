import torch
import torch.nn as nn


class StGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature_size = args.n_fapp + args.n_fpos

        self.weight_matrix = torch.randn(size=(self.feature_size, self.feature_size), requires_grad=True).to(torch.device("cuda"))
        self.relu = nn.ReLU()

    def forward(self, node_list, adj_matrix):
        weighted_feature = torch.matmul(adj_matrix, node_list)
        st_node_h = self.relu(torch.matmul(weighted_feature, self.weight_matrix))
        return st_node_h

