import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout)
        self.downsample = None
        if input_size != output_size:
            self.downsample = nn.Conv1d(input_size, output_size, 1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # Pad or trim the res tensor to match the size of out
        if out.size(-1) != res.size(-1):
            pad_amount = out.size(-1) - res.size(-1)
            res = nn.functional.pad(res, (0, pad_amount))
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else hidden_units
            out_channels = hidden_units
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                      dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                      dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x, hidden=None):
        x = x.permute(0, 2, 1)  # Reshape for Conv1D
        x = self.network(x)
        x = x.permute(0, 2, 1)  # Reshape back
        return x, hidden

