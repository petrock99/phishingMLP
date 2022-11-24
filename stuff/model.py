import torch.nn as nn

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class BinaryMLPModel(nn.Module):
    def __init__(self, n_inputs, n_hiddens_list, use_gpu):
        assert isinstance(n_hiddens_list, list), "n_hiddens_list must be a list or tuple"

        super(BinaryMLPModel, self).__init__()

        # Build a list of hidden linear layers and corresponding batchnorms
        layers = []
        batchnorms = []
        in_features = n_inputs
        out_features = 0
        for out_features in n_hiddens_list:
            layers.append(nn.Linear(in_features, out_features))
            batchnorms.append(nn.BatchNorm1d(out_features))
            in_features = out_features

        # Set up the output layer, which only has one node to make this model
        # a binary classifier
        n_output = 1
        layers.append(nn.Linear(out_features, n_output))

        self.n_hiddens_list = n_hiddens_list
        self.layers = nn.ModuleList(layers)
        self.batchnorms = nn.ModuleList(batchnorms)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

        # Use the GPU if it supports CUDA, otherwise use the CPU
        self.to('cuda' if use_gpu else 'cpu')

    def forward(self, inputs):
        x = inputs
        assert len(self.layers) == len(self.batchnorms) + 1, "Expected last layer to be the output later"
        for layer, batchnorm in zip(self.layers[:-1], self.batchnorms):
            x = self.relu(layer(x))
            x = batchnorm(x)
        x = self.dropout(x)
        return self.sigmoid(self.layers[-1](x))

    def __str__(self):
        return f'{super().__str__()}\nUsing: {self.device}'
