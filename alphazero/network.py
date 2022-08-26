import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

import renju
import config


class ResBlock(nn.Module):
    def __init__(self, filter):
        super().__init__()
        self.conv1 = nn.Conv2d(filter, filter, kernel_size=3, padding=[1, 1])
        self.conv2 = nn.Conv2d(filter, filter, kernel_size=3, padding=[1, 1])
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x = x1 + x
        x = self.relu2(x)
        return x


class AlphaZeroImageBlock(nn.Module):
    def __init__(self, n_blocks=10, filter=256):
        super().__init__()

        self.conv1 = nn.Conv2d(renju.INPUT_CH, filter, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter)
        self.relu = nn.LeakyReLU()
        self.resblocks = nn.ModuleList([ResBlock(filter) for _ in range(n_blocks)])

    def forward(self, x):
        # print(x.dtype)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class AlphaZeroNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # image block
        self.resnet = AlphaZeroImageBlock(config.n_res_block)

        # policy net
        self.p_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_relu = nn.LeakyReLU()
        self.p_dense = nn.Linear(renju.ACTION_SPACE, renju.ACTION_SPACE)

        # value net
        self.v_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_relu = nn.LeakyReLU()
        self.v_dense = nn.Linear(renju.BOARD_SIZE ** 2, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, state, action_mask):
        """
        state: torch.Tensor [batch, c, h, w]
        action_mask: torch.Tensor [batch, c, h, w]
        """
        # x = torch.from_numpy(state.astype(np.float32)).clone()
        mask = torch.flatten(action_mask, 1).to(torch.float32)
        x = state.to(torch.float32)

        x = self.resnet(x)

        p = self.p_conv(x)
        p = self.p_bn(p)
        p = self.p_relu(p)
        p = torch.flatten(p, start_dim=1)
        p = F.softmax(self.p_dense(p), dim=1) + 0.0001

        v = self.v_conv(x)
        v = self.v_bn(v)
        v = self.v_relu(v)
        v = torch.flatten(v, start_dim=1)
        v = self.v_dense(v)
        v = self.value(v)

        return p * mask, v

    def predict(self, state, action_mask):
        """
        Parameters
        ------
        state: torch.Tensor([2, SIZE, SIZE])
        action_mask: torch.Tensor([2, SIZE, SIZE])

        Returns
        -----
        policy: torch.Tensor([2 * SIZE * SIZE])
        value: torch.Tensor([1])
        """
        state = torch.unsqueeze(torch.from_numpy(state), 0)
        action_mask = torch.unsqueeze(torch.from_numpy(action_mask), 0)
        p, v = self.forward(state, action_mask)
        return torch.squeeze(p, 0), torch.squeeze(v, 0)
