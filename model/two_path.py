from model.sequence.sequence_model import SequentialModel
from model.strg.strg import StrG
from model.srnn.srnn import SRNN
from model.beedog.beedog import Beedog
from model.vivit_mlp.vivit_mlp import ViVitMLP
import torch.nn as nn
import torch


class TwoPath(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.basic_graph = args.basic_graph
        self.global_path = args.global_path
        if args.global_path == 'sequence':
            self.hidden_size = args.sequence_hidden_size
        else:
            self.hidden_size = args.mlp_nodes
        self.sequence_layer = args.sequence_layer

        if self.basic_graph == 'strg':
            self.strg = StrG(args)
            self.graph_output_size = args.graph_embedding_size
        elif self.basic_graph == 'srnn':
            self.srnn = SRNN(args)
            self.graph_output_size = args.node_output_size
        elif self.basic_graph == 'beedog' or 'spatial' or 'object':
            self.beedog = Beedog(args)
            self.graph_output_size = args.g_sequence_hidden_size * args.g_sequence_layer
        if self.global_path == 'resnet2d' or self.global_path == 'sequence':
            self.sequence_model = SequentialModel(args)
        elif self.global_path == 'vivit':
            self.vivit_mlp = ViVitMLP(args)

        self.output_size = args.class_num
        if args.global_path == 'sequence':
            self.classifier = nn.Linear(self.graph_output_size + self.hidden_size * self.sequence_layer, self.output_size)
        else:
            self.classifier = nn.Linear(self.graph_output_size + self.hidden_size, self.output_size)

    def forward(self, args, true_batch_size, node_list=None, edge_index=None, fw_adj_matrix=None, bw_adj_matrix=None,
                node_num=None, node_features=None, spatial_edge_features=None, temporal_edge_features=None, global_input=None,
                graph_list=None, adjacent_mappings=None):
        graph_feature = None
        global_feature = None

        if args.inference:
            if self.basic_graph == 'beedog':
                graph_feature, edge_index, weight0_list, weight1_list = self.beedog.extract_feature(args, graph_list, adjacent_mappings)
        else:
            if self.basic_graph == 'strg':
                graph_feature = self.strg.extract_feature(true_batch_size, node_list, edge_index, fw_adj_matrix, bw_adj_matrix)
            elif self.basic_graph == 'srnn':
                graph_feature = self.srnn.extract_feature(node_num, true_batch_size, node_features, spatial_edge_features, temporal_edge_features)
            elif self.basic_graph == 'beedog':
                graph_feature = self.beedog.extract_feature(args, graph_list, adjacent_mappings)

        if self.global_path == 'resnet2d' or self.global_path == 'sequence':
            global_feature = self.sequence_model.extract_feature(global_input, true_batch_size)
        elif self.global_path == 'vivit':
            global_feature = self.vivit_mlp.extract_features(global_input)


        # print(f"the shape of global feature is {global_feature.shape}, the feature of graph feature is {graph_feature.shape}")
        out = torch.cat((graph_feature, global_feature), dim=1)
        out = self.classifier(out)
        out = torch.softmax(out, dim=-1)

        if args.inference:
            return out, edge_index, weight0_list, weight1_list
        else:
            return out

