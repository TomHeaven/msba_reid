import torch.nn as nn

from .attention import CAM_Module

__all__ = ['ShallowCAM']

class ShallowCAM(nn.Module):

    def __init__(self, feature_dim: int):

        super().__init__()
        self.input_feature_dim = feature_dim
        self._cam_module = CAM_Module(self.input_feature_dim)


    def forward(self, x):
        x = self._cam_module(x)
        return x
