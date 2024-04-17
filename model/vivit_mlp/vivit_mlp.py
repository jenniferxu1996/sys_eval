import torch
from torch import nn


class ViVitMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = nn.Linear(args.vivit_embed_dims, args.mlp_nodes)
        self.relu = nn.ReLU()

    def extract_features(self, vivit_x):
        out = self.relu(self.mlp(vivit_x))
        return out
