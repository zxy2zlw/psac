import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Position_linear(nn.Module):
    """
    A class representing a Positional Linear layer.
    """

    def __init__(self, window_size=3, filter_num=128, feature=128, seq_len=41):
        super(Position_linear, self).__init__()
        self.filter_num = filter_num
        self.feature = feature
        self.window_size = window_size
        self.seq_len = seq_len
        self.pad_len = int(self.window_size / 2)

        # Create a list to store linear layers for each position
        self.dense_layer_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.window_size * self.feature, filter_num),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            for _ in range(self.seq_len)
        ])

    def forward(self, inputs):
        x = torch.transpose(inputs, 1, 2)
        x = F.pad(x, (self.pad_len, self.pad_len), "constant", 0)
        x = torch.transpose(x, 1, 2)

        x_temp = [
            self.dense_layer_net[i - self.pad_len](
                x[:, i - self.pad_len: i + self.pad_len + 1, :].reshape(-1, self.window_size * self.feature)
            )
            for i in range(self.pad_len, self.seq_len + self.pad_len)
        ]

        x = torch.stack(x_temp)
        x = torch.transpose(x, 0, 1)
        return x


def squash(input, eps=1e-21):
    """
    Applies the squash function to the input vector.
    """
    n = torch.norm(input, dim=-1, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (input / (n + eps))


class PrimaryCapsLayer(nn.Module):
    """
    A class representing a Primary Capsule Layer in a Capsule Network.
    """

    def __init__(self, in_channels, kernel_size, num_capsules, padding, stride=1):
        super(PrimaryCapsLayer, self).__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=num_capsules,
            padding=padding,
        )

    def forward(self, input):
        output = self.depthwise_conv(input)
        return squash(output)


class RoutingLayer(nn.Module):
    """
    A class representing a Routing Layer in a Capsule Network.
    """

    def __init__(self, num_capsules, in_channel, feature):
        super(RoutingLayer, self).__init__()
        self.W = torch.randn(num_capsules, in_channel, feature, device=device)
        self.b = nn.Parameter(torch.zeros(num_capsules, in_channel, 1, device=device))

    def forward(self, inputs):
        # Transform input capsules using the weight matrix W
        u = torch.einsum('...ji,kji->...kji', inputs, self.W)

        # Calculate the coupling coefficients c using dynamic routing
        c = torch.einsum("...ji,...ji->...j", u, u)[..., None]
        c = c / torch.sqrt(torch.tensor([128.0], device=device))
        c = torch.softmax(c, dim=-1)
        c = c + self.b
        s = torch.sum(u * c, dim=-2)
        return squash(s)


class FC(nn.Module):
    def __init__(self, feature, dropout):
        super().__init__()
        self.LastDense = nn.Sequential(
            nn.Linear(feature, 64, bias=False),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(2).expand(-1, -1, x.size(1), -1) + x.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        x, _ = torch.max(x, dim=1)
        result = self.LastDense(x)
        return result


class Deepnet(nn.Module):
    """
    A class representing a deep neural network (Deepnet) with a Capsule Network architecture.
    """

    def __init__(self, feature, dropout, filter_num, seq_len):
        super().__init__()
        self.embeddings = nn.Embedding(4, feature)
        self.position_linear_3 = Position_linear(window_size=3, filter_num=filter_num, feature=feature, seq_len=seq_len)
        self.position_linear_5 = Position_linear(window_size=5, filter_num=filter_num, feature=feature, seq_len=seq_len)
        self.position_linear_7 = Position_linear(window_size=7, filter_num=filter_num, feature=feature, seq_len=seq_len)
        self.PrimaryCapsLayer = PrimaryCapsLayer(in_channels=64, kernel_size=3, num_capsules=2, padding=1, stride=1)
        self.RoutingLayer = RoutingLayer(num_capsules=2, in_channel=64, feature=feature)
        self.conv1 = nn.Conv1d(in_channels=41, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)
        self.Fc = FC(feature=feature, dropout=dropout)

    def forward(self, x):
        x = self.embeddings(x)
        x3 = self.position_linear_3(x)
        x5 = self.position_linear_5(x3)
        x7 = self.position_linear_7(x5)
        x = self.conv1(x7)
        x = self.bn(x)
        x = self.relu(x)
        x = self.PrimaryCapsLayer(x)
        x = self.RoutingLayer(x)
        x = self.Fc(x)
        result, _ = torch.max(x, dim=1)
        return result
