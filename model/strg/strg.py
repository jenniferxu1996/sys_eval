import torch
from model.strg.sim_gcn import SimGCN
from model.strg.st_gcn import StGCN
import torch.nn as nn


class StrG(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.feature_size = args.n_fapp + args.n_fpos
        self.graph_embedding_size = args.graph_embedding_size
        self.output_size = args.output_size

        self.sim_gcn = SimGCN(args)
        self.st_fw_gcn = StGCN(args)
        self.st_bw_gcn = StGCN(args)
        self.classifier = nn.Linear(self.graph_embedding_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, true_batch_size, node_list, edge_index, fw_adj_matrix, bw_adj_matrix):
        sim_node_h = self.sim_gcn(node_list, edge_index, true_batch_size)

        fw_st_node_h = self.st_fw_gcn(node_list, fw_adj_matrix)
        if bw_adj_matrix is not None:
            bw_st_node_h = self.st_bw_gcn(node_list, bw_adj_matrix)
            st_node_h = torch.add(fw_st_node_h, bw_st_node_h)
        else:
            st_node_h = fw_st_node_h

        sum_node_h = torch.add(sim_node_h, st_node_h)

        out = torch.mean(sum_node_h, 1)
        out = self.classifier(out)
        out = torch.softmax(out, dim=-1)
        return out

    def extract_feature(self, true_batch_size, node_list, edge_index, fw_adj_matrix, bw_adj_matrix):
        sim_node_h = self.sim_gcn(node_list, edge_index, true_batch_size)

        fw_st_node_h = self.st_fw_gcn(node_list, fw_adj_matrix)
        if bw_adj_matrix is not None:
            bw_st_node_h = self.st_bw_gcn(node_list, bw_adj_matrix)
            st_node_h = torch.add(fw_st_node_h, bw_st_node_h)
        else:
            st_node_h = fw_st_node_h

        sum_node_h = torch.add(sim_node_h, st_node_h)
        out = torch.mean(sum_node_h, 1)
        return out



