import torch
import numpy as np
import architectures.pinn1d as pinn1d

class HardConstrainedTimeResNet(torch.nn.Module):
    def __init__(self,
                 number_of_blocks: int,
                 block_width: int,
                 number_of_outputs: int,
                 initial_position: float,
                 initial_speed: float,
                 time_bubble_sigma: float=0.3,
                 device='cuda'):
        super().__init__()
        self.z_predictor = pinn1d.ResidualNet(
            number_of_inputs=1,
            number_of_blocks=number_of_blocks,
            block_width=block_width,
            number_of_outputs=number_of_outputs
        ).to(device)
        self.u0 = initial_position
        self.v0 = initial_speed
        self.time_bubble_sigma = time_bubble_sigma

    def forward(self, t_tsr):  # t_tsr.shape = (B, 1)
        bubble_t_tsr = self.time_bubble(t_tsr)  # (B, 1)
        z_tsr = self.z_predictor(t_tsr)  # (B, 1)
        initial_interpolation_tsr = self.u0 + self.v0 * t_tsr  # (B, 1)
        return initial_interpolation_tsr + bubble_t_tsr * z_tsr  # (B, 1)

    def time_bubble(self, t_tsr):  # t_tsr.shape = (B, 1)
        g = torch.exp(-(torch.pow(t_tsr, 2))/(2 * self.time_bubble_sigma**2))  # (B, 1)
        return 1 - g  # (B, 1)

if __name__ == '__main__':
    device = 'cuda'


    net = HardConstrainedTimeResNet(
        number_of_blocks=2,
        block_width=32,
        number_of_outputs=1,
        initial_position=0.1,
        initial_speed=1.0,
        time_bubble_sigma=0.3,
        device=device
    )

    input_tsr = torch.randn(8, 1).to(device)
    output_tsr = net(input_tsr)