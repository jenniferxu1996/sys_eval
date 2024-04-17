import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch_geometric.nn import GATConv, GCNConv, BatchNorm
from torch_geometric.data import Batch, Data


class Beedog(nn.Module):
    def __init__(self, args):
        super().__init__()
        # The dimensionality of the input feature
        if args.model_name == 'beedog':
            self.n_feature_a = 2 * args.n_fapp
            self.n_feature_p = 2 * args.n_fpos
        else:
            self.n_feature_a = args.n_fapp
            self.n_feature_p = args.n_fpos
        self.embedding_size = args.graph_embedding_size
        self.layers = args.graph_layers
        self.hidden_size = args.g_sequence_hidden_size
        self.sequence_layer = args.g_sequence_layer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_size = args.class_num
        self.has_attention = args.has_attention
        self.drop_out = nn.Dropout(args.drop_out)
        # The definition of the model_infer and parameters
        self.norm1 = BatchNorm(self.n_feature_a + self.n_feature_p)
        if self.has_attention:
            self.conv1 = GATConv(self.n_feature_a + self.n_feature_p, self.n_feature_a + self.n_feature_p)
            self.conv2 = GATConv(self.n_feature_a + self.n_feature_p, self.embedding_size)
        else:
            self.conv1 = GCNConv(self.n_feature_a + self.n_feature_p, self.n_feature_a + self.n_feature_p)
            self.conv2 = GCNConv(self.n_feature_a + self.n_feature_p, self.embedding_size)

        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                            num_layers=self.sequence_layer)
        self.classifier = nn.Linear(self.hidden_size * self.sequence_layer, self.output_size)
        # self.inference = args.inference

    def adjacent_mappings_to_edge_index(self, args, adjacent_mappings: List[List[int]]):
        edge_index = []
        for vertex_i in range(0, len(adjacent_mappings)):
            for neighbour in adjacent_mappings[vertex_i]:
                # edge = [vertex_i, neighbour]
                if args.inverse:
                    edge = [vertex_i, neighbour]
                else:
                    edge = [neighbour, vertex_i]
                edge_index.append(edge)
        edge_index = [list(i) for i in zip(*edge_index)]
        return torch.LongTensor(edge_index).to(self.device)

    def to_pyg_batch(self, batch_data, edge_index):
        pyg_batch = []
        for batch in batch_data:
            pyg_batch.append(Data(batch, edge_index=edge_index))
        return Batch.from_data_list(pyg_batch)

    def to(self, device, *args, **kwargs):
        self.device = device
        return super(Beedog, self).to(device, *args, **kwargs)

    def forward(self, args, node_features: torch.Tensor, adjacent_mappings: List[List[int]]):
        """
        :param node_features: size = (number_of_stc, batch_size, obj_pair_num*2, n_features)
        :param adjacent_mappings: a two-dimensional list, (obj_pair_num*2, n_neighbours)
               for example, if the current node is node 0, and the adjacent_mappings[0] = [1, 2, 4]
        :return:
        """
        batch_size = node_features.shape[1]
        sequence = []
        weight0_list = []
        weight1_list = []
        edge_index = self.adjacent_mappings_to_edge_index(args, adjacent_mappings)
        for st_c in node_features:
            # transform a mini-batch of st_c into the format of pyg batch
            pyg_batch = self.to_pyg_batch(st_c, edge_index)
            # hidden_state = self.norm1(pyg_batch.x)
            if args.inference:
                hidden_state, weight0 = self.conv1(pyg_batch.x, pyg_batch.edge_index, return_attention_weights=True)
                hidden_state = F.relu(hidden_state)
                hidden_state, weight1 = self.conv2(hidden_state, pyg_batch.edge_index, return_attention_weights=True)
                hidden_state = F.relu(hidden_state)
                weight0_list.append([i[0] for i in weight0[1].tolist()])
                weight1_list.append([i[0] for i in weight1[1].tolist()])
            else:
                hidden_state = F.relu(self.conv1(pyg_batch.x, pyg_batch.edge_index))
                hidden_state = F.relu(self.conv2(hidden_state, pyg_batch.edge_index))
            # resume the format of pyg batch to original format
            hidden_state = hidden_state.view(batch_size, -1, self.conv2.out_channels)

            hidden_state_output = torch.sum(hidden_state, 1)
            sequence.append(hidden_state_output)


        output, (h_n, c_n) = self.lstm(torch.stack(sequence))
        # switch the batch_size to the first dimension
        h_n = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        out = F.relu(h_n)
        out = self.classifier(out)
        out = torch.softmax(out, dim=1)
        if args.inference:
            edge_index = weight0[0]
            return out, edge_index, weight0_list, weight1_list
        else:
            return out

    def extract_feature(self, args, node_features: torch.Tensor, adjacent_mappings: List[List[int]]):
        """
        :param node_features: size = (number_of_stc, batch_size, obj_pair_num*2, n_features)
        :param adjacent_mappings: a two-dimensional list, (obj_pair_num*2, n_neighbours)
               for example, if the current node is node 0, and the adjacent_mappings[0] = [1, 2, 4]
        :return:
        """
        batch_size = node_features.shape[1]
        sequence = []
        weight0_list = []
        weight1_list = []
        edge_index = self.adjacent_mappings_to_edge_index(args, adjacent_mappings)
        for st_c in node_features:
            # transform a mini-batch of st_c into the format of pyg batch
            pyg_batch = self.to_pyg_batch(st_c, edge_index)
            # hidden_state = self.norm1(pyg_batch.x)
            if args.inference:
                hidden_state, weight0 = self.conv1(pyg_batch.x, pyg_batch.edge_index, return_attention_weights=True)
                hidden_state = F.relu(hidden_state)
                hidden_state, weight1 = self.conv2(hidden_state, pyg_batch.edge_index, return_attention_weights=True)
                hidden_state = F.relu(hidden_state)
                weight0_list.append([i[0] for i in weight0[1].tolist()])
                weight1_list.append([i[0] for i in weight1[1].tolist()])
            else:
                hidden_state = F.relu(self.conv1(pyg_batch.x, pyg_batch.edge_index))
                hidden_state = F.relu(self.conv2(hidden_state, pyg_batch.edge_index))
                # resume the format of pyg batch to original format
            hidden_state = hidden_state.view(batch_size, -1, self.conv2.out_channels)

            hidden_state_output = torch.sum(hidden_state, 1)
            sequence.append(hidden_state_output)

        output, (h_n, c_n) = self.lstm(torch.stack(sequence))
        # switch the batch_size to the first dimension
        h_n = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        out = F.relu(h_n)
        out = self.drop_out(out)
        if args.inference:
            edge_index = weight0[0]
            return out, edge_index, weight0_list, weight1_list
        else:
            return out

