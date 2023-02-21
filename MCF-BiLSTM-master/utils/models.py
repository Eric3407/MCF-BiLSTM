import torch
import torch.nn as nn
import torch.nn.functional as F





class MCF_BiLSTM(nn.Module):
    def __init__(self):
        super(MCF_BiLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=13, stride=1, padding=6, bias=False),
            nn.BatchNorm1d(1), nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=33, stride=1, padding=16, bias=False),
            nn.BatchNorm1d(1), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=53, stride=1, padding=26, bias=False),
            nn.BatchNorm1d(1), nn.ReLU())

        self.lstm1 = nn.LSTM(input_size=3, hidden_size=64,
                             num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm2= nn.LSTM(input_size=64*2, hidden_size=256,
                             num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)

        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(256 * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 =nn.Sequential(nn.Linear(512, 128))
        self.fc = nn.Sequential(nn.Linear(128, 3))

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x = torch.cat((x_1, x_2, x_3), dim=1)

        x = x.permute(0, 2, 1)
        H, _ = self.lstm1(x)
        H, _ = self.lstm2(H)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)

        return out
#
