from typing import TypeAlias
import torch
import torch.nn as nn

Tensor: TypeAlias = torch.FloatTensor


class DQN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        n_actions: int,
    ):
        super().__init__()  # type: ignore

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape: tuple[int, int, int]):
        with torch.no_grad():
            out = self.conv(torch.zeros(1, *shape))
        return out.numel()

    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv(x).view(x.shape[0], -1)
        return self.fc(conv_out)
