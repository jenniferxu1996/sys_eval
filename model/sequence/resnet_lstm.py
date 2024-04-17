import torch
import torch.nn as nn
import torchvision.models as models


class ResNetSequence(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True) # load the pretrained ResNet50 model
        # for param in self.resnet.parameters(): # freeze the ResNet50 parameters
        #     param.requires_grad = False
        self.hidden_size = args.sequence_hidden_size
        self.sequence_layer = args.sequence_layer
        self.resnet.fc = nn.Identity() # remove the last fully connected layer
        self.lstm = nn.LSTM(input_size=2048, hidden_size=self.hidden_size, num_layers=self.sequence_layer, batch_first=True) # add a LSTM layer
        self.linear = nn.Linear(self.hidden_size * self.sequence_layer, args.num_class)  # add a linear layer for classification
        self.BatchNorm = nn.BatchNorm1d(self.sequence_layer * self.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size() # x is a tensor of shape (batch_size, timesteps, 3, 224, 224)
        x = x.view(batch_size * timesteps, C, H, W) # reshape x to feed into ResNet50
        x = self.resnet(x) # get the feature vector of shape (batch_size * timesteps, 2048)
        x = x.view(batch_size, timesteps, -1) # reshape x to feed into LSTM
        x, (h, c) = self.lstm(x) # get the output and hidden state of LSTM
        h = h.permute(1, 0, 2).contiguous().view(-1,  self.sequence_layer * self.hidden_size)
        # print(f'the shape of h is {h.shape}')
        # x = self.linear(x[:, -1, :]) # get the final output of shape (batch_size, 10)
        if batch_size > 1:
            h = self.relu(self.BatchNorm(h))
        out = self.linear(h)
        # print(f'the x for softmax is {out.shape}')
        # out = torch.softmax(out, dim=-1)
        return out
