import torch
import torch.nn as nn
from model.srnn.node_rnn import SegmentNodeRNN
from model.srnn.edge_rnn import SpatialEdgeRNN, TemporalEdgeRNN


class SRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.attention = args.attention
        self.node_num = args.node_num

        self.segment_node_hidden_size = args.node_hidden_size
        self.spatial_edge_hidden_size = args.edge_hidden_size_spatial
        self.temporal_edge_hidden_size = args.edge_hidden_size_temporal
        self.node_output_size = args.node_output_size

        self.SegmentNodeRNN = SegmentNodeRNN(args)
        self.SpatialEdgeRNN = SpatialEdgeRNN(args)
        self.TemporalEdgeRNN = TemporalEdgeRNN(args)
        self.AttLayer = ATTLayer(args)
        self.BatchNorm_1 = nn.BatchNorm1d(self.node_output_size)
        self.node_linear = nn.Linear(self.node_output_size, self.node_output_size)
        self.classifier = nn.Linear(self.node_output_size, args.output_size)
        self.relu = nn.ReLU()

    def forward(self, node_num, true_batch_size, node_features, spatial_edge_features, temporal_edge_features):
        sequence_length = len(spatial_edge_features[0])
        node_output_list = []

        # Iterate for one node
        for node_i in range(node_num):
            # print(f"The current node is node {node_i}")
            edge_hidden_state_t = None
            edge_cell_state_t = None
            edge_hidden_state_s = None
            edge_cell_state_s = None
            node_hidden_state = None
            node_cell_state = None
            node_output = None

            # Iterate all time step for one node
            for current_frame in range(sequence_length):
                # print(f"The current frame is frame {current_frame}")
                # edge_hidden_state_s_list = []  # The list to store all the spatial edge hidden state for a node in one frame
                if current_frame != 0:
                    # Forward pass for the current spatial edge features
                    # For one node, iterate all edges connecting other nodes in current frame
                    edge_feature_s_list = []
                    for node_j in range(node_num):
                        if node_i != node_j:
                            for k in range(true_batch_size):
                                if k == 0:
                                    current_edge_feature_s = torch.unsqueeze(spatial_edge_features[k][current_frame][node_i * (node_num - 1) + node_j], 0)
                                else:
                                    current_edge_feature_s_one_sample = torch.unsqueeze(spatial_edge_features[k][current_frame][node_i * (node_num - 1) + node_j], 0)
                                    current_edge_feature_s = torch.cat((current_edge_feature_s, current_edge_feature_s_one_sample), dim=0)
                            edge_feature_s_list.append(current_edge_feature_s)

                    # Forward pass for the current temporal edge
                    for k in range(true_batch_size):
                        if k == 0:
                            current_edge_feature_t = torch.unsqueeze(temporal_edge_features[k][current_frame - 1][node_i], 0)
                        else:
                            current_edge_feature_t_one_sample = torch.unsqueeze(temporal_edge_features[k][current_frame - 1][node_i], 0)
                            current_edge_feature_t = torch.cat((current_edge_feature_t, current_edge_feature_t_one_sample), dim=0)
                    edge_hidden_state_t, edge_cell_state_t = self.TemporalEdgeRNN(true_batch_size, current_edge_feature_t, edge_hidden_state_t, edge_cell_state_t)
                    # print(f" the current edge hidden state is {edge_hidden_state_t}, and the size is {edge_hidden_state_t.shape}")

                    if self.attention is True:
                        # Use current temporal edge feature to attend over all the current spatial edge features
                        # so that get the weighted sum of the spatial edge features
                        edge_feature_s_list = self.AttLayer(true_batch_size, current_edge_feature_t, edge_feature_s_list)
                    current_edge_feature_all = torch.sum(torch.stack(edge_feature_s_list), 0)
                    edge_hidden_state_s, edge_cell_state_s = self.SpatialEdgeRNN(true_batch_size,current_edge_feature_all, edge_hidden_state_s, edge_cell_state_s)

                    # Forward pass for the current node
                    for k in range(true_batch_size):
                        if k == 0:
                            current_node_feature = torch.unsqueeze(torch.Tensor(node_features[k][(current_frame, node_i)]), 0).to(torch.device("cuda"))
                        else:
                            current_node_feature_one_sample = torch.unsqueeze(torch.Tensor(node_features[k][(current_frame, node_i)]), 0).to(torch.device("cuda"))
                            current_node_feature = torch.cat((current_node_feature, current_node_feature_one_sample), dim=0)
                    node_output, node_hidden_state, node_cell_state = self.SegmentNodeRNN(true_batch_size, current_node_feature, edge_hidden_state_t, edge_hidden_state_s, node_hidden_state, node_cell_state)

            if node_output is not None:
                node_output_list.append(node_output)

        # Sum the output of all nodes up to make the final prediction
        node_output_all = torch.sum(torch.stack(node_output_list), dim=0)

        output = node_output_all
        if true_batch_size > 1:
            output = self.BatchNorm_1(node_output_all)
        output = self.classifier(output)
        # output = self.relu(output)
        output = torch.softmax(output, dim=-1)
        # print(f"the output of the whole model is {output}")

        return output

    def extract_feature(self, node_num, true_batch_size, node_features, spatial_edge_features, temporal_edge_features):
        sequence_length = len(spatial_edge_features[0])
        node_output_list = []

        # Iterate for one node
        for node_i in range(node_num):
            # print(f"The current node is node {node_i}")
            edge_hidden_state_t = None
            edge_cell_state_t = None
            edge_hidden_state_s = None
            edge_cell_state_s = None
            node_hidden_state = None
            node_cell_state = None
            node_output = None

            # Iterate all time step for one node
            for current_frame in range(sequence_length):
                # print(f"The current frame is frame {current_frame}")
                # edge_hidden_state_s_list = []  # The list to store all the spatial edge hidden state for a node in one frame
                if current_frame != 0:
                    # Forward pass for the current spatial edge features
                    # For one node, iterate all edges connecting other nodes in current frame
                    edge_feature_s_list = []
                    for node_j in range(node_num):
                        if node_i != node_j:
                            for k in range(true_batch_size):
                                if k == 0:
                                    current_edge_feature_s = torch.unsqueeze(spatial_edge_features[k][current_frame][node_i * (node_num - 1) + node_j], 0)
                                else:
                                    current_edge_feature_s_one_sample = torch.unsqueeze(spatial_edge_features[k][current_frame][node_i * (node_num - 1) + node_j], 0)
                                    current_edge_feature_s = torch.cat((current_edge_feature_s, current_edge_feature_s_one_sample), dim=0)
                            edge_feature_s_list.append(current_edge_feature_s)

                    # Forward pass for the current temporal edge
                    for k in range(true_batch_size):
                        if k == 0:
                            current_edge_feature_t = torch.unsqueeze(temporal_edge_features[k][current_frame - 1][node_i], 0)
                        else:
                            current_edge_feature_t_one_sample = torch.unsqueeze(temporal_edge_features[k][current_frame - 1][node_i], 0)
                            current_edge_feature_t = torch.cat((current_edge_feature_t, current_edge_feature_t_one_sample), dim=0)
                    edge_hidden_state_t, edge_cell_state_t = self.TemporalEdgeRNN(true_batch_size, current_edge_feature_t, edge_hidden_state_t, edge_cell_state_t)
                    # print(f" the current edge hidden state is {edge_hidden_state_t}, and the size is {edge_hidden_state_t.shape}")

                    if self.attention is True:
                        # Use current temporal edge feature to attend over all the current spatial edge features
                        # so that get the weighted sum of the spatial edge features
                        edge_feature_s_list = self.AttLayer(true_batch_size, current_edge_feature_t, edge_feature_s_list)
                    current_edge_feature_all = torch.sum(torch.stack(edge_feature_s_list), 0)
                    edge_hidden_state_s, edge_cell_state_s = self.SpatialEdgeRNN(true_batch_size,current_edge_feature_all, edge_hidden_state_s, edge_cell_state_s)

                    # Forward pass for the current node
                    for k in range(true_batch_size):
                        if k == 0:
                            current_node_feature = torch.unsqueeze(torch.Tensor(node_features[k][(current_frame, node_i)]), 0).to(torch.device("cuda"))
                        else:
                            current_node_feature_one_sample = torch.unsqueeze(torch.Tensor(node_features[k][(current_frame, node_i)]), 0).to(torch.device("cuda"))
                            current_node_feature = torch.cat((current_node_feature, current_node_feature_one_sample), dim=0)
                    node_output, node_hidden_state, node_cell_state = self.SegmentNodeRNN(true_batch_size, current_node_feature, edge_hidden_state_t, edge_hidden_state_s, node_hidden_state, node_cell_state)

            if node_output is not None:
                node_output_list.append(node_output)

        # Sum the output of all nodes up to make the final prediction
        node_output_all = torch.sum(torch.stack(node_output_list), dim=0)

        return node_output_all




class ATTLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.edge_feature_size = args.edge_feature_size
        self.linear = nn.Linear(self.edge_feature_size, self.edge_feature_size)

    def forward(self, true_batch_size, temporal_edge_feature, edge_feature_s_list):
        e_j_sum = torch.zeros((true_batch_size, 1)).to(torch.device("cuda"))
        e_ij_list = torch.zeros((len(edge_feature_s_list), true_batch_size, 1)).to(torch.device("cuda"))
        # print(e_j_sum)
        for count, each_edge in enumerate(edge_feature_s_list):
            m_i = self.linear(temporal_edge_feature)
            m_j = self.linear(each_edge)
            e_ij = torch.unsqueeze(torch.stack([torch.sum(torch.mul(m_i[h], m_j[h])) for h in range(m_i.shape[0])]), 1)
            # print(e_ij)
            e_ij_list[count] = e_ij
            for b in range(true_batch_size):
                e_j_sum[b] += e_ij[b]
        # print(e_j_sum)
        # print(e_ij_list)
        weighted_list = [torch.div(k, e_j_sum)for k in e_ij_list]
        # print(weighted_list)
        for index, each_edge in enumerate(edge_feature_s_list):
            edge_feature_s_list[index] = torch.mul(weighted_list[index], each_edge)
            # print(weighted_list[index])
            # print(edge_feature_s_list[index])
        return edge_feature_s_list




