import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, BatchNorm, LayerNorm
from torch_geometric.data import Batch, Data


class SimGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.feature_size = args.n_fapp + args.n_fpos
        self.embedding_size = args.graph_embedding_size

        self.gconv1 = GCNConv(self.feature_size, self.embedding_size)
        self.gconv2 = GCNConv(self.embedding_size, self.embedding_size)
        self.gconv3 = GCNConv(self.embedding_size, self.embedding_size)

        self.layers = args.graph_layers

        # self.norm1 = LayerNorm(self.embedding_size)
        # self.norm2 = LayerNorm(self.embedding_size)
        self.norm1 = BatchNorm(self.embedding_size)
        self.norm2 = BatchNorm(self.embedding_size)


    def to_pyg_batch(self, batch_data, edge_index):
        pyg_batch = []
        for batch in batch_data:
            # print(Data(x=batch, edge_index=edge_index))
            pyg_batch.append(Data(x=batch, edge_index=edge_index))
        return Batch.from_data_list(pyg_batch)

    # def to(self, device, *args, **kwargs):
    #     self.device = device
    #     return super(SimGCN, self).to(device, *args, **kwargs)

    def forward(self, node_list, edge_index, true_batch_size):
        pyg_batch = self.to_pyg_batch(node_list, edge_index)
        # print(pyg_batch)
        for i in range(self.layers):
            if i == 0:
                sim_node_h = self.gconv1(pyg_batch.x, pyg_batch.edge_index).relu()
                sim_node_h = self.norm1(sim_node_h)
            elif i == 1:
                sim_node_h = self.gconv2(sim_node_h, pyg_batch.edge_index).relu()
                sim_node_h = self.norm2(sim_node_h)
            else:
                sim_node_h = self.gconv3(sim_node_h, pyg_batch.edge_index).relu()
        sim_node_h = sim_node_h.view(true_batch_size, -1, self.gconv2.out_channels)
        # print(sim_node_h.shape)
        return sim_node_h









