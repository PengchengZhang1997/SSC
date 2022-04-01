import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TPS_SpatialTransformerNetwork(nn.Module):

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        
        if torch.__version__ > "1.2.0":
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        self.localization_fc2.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        return batch_C_prime


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C

    def _build_inv_delta_C(self, F, C):
        hat_C = np.zeros((F, F), dtype=float)
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        P = np.stack(
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))
        C_tile = np.expand_dims(C, axis=0)
        P_diff = P_tile - C_tile
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat

    def build_P_prime(self, batch_C_prime):
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
            batch_size, 3, 2).float().to(device)), dim=1)
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)
        return batch_P_prime
