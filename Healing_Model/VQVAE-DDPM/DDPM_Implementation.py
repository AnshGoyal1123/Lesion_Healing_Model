import torch
import torch.nn as nn
import numpy as np

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet3D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Downsampling path
        for feature in features:
            self.downs.append(self._block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(self._block(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Check and adjust if needed, to handle potential shape mismatch
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

class DDPM(nn.Module):
    def __init__(self, betas, num_timesteps, model, device):
        super(DDPM, self).__init__()
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.num_timesteps = num_timesteps
        self.model = model
        self.device = device

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device).float()
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]), dim=0).float()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device).float()
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev).to(device).float()  # Ensure this is initialized
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / 
                                   (1.0 - self.alphas_cumprod)).to(device).float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Reshape the time tensor to be broadcastable with x_start
        t = t.view(-1, 1, 1, 1, 1)
        
        return (self.sqrt_alphas_cumprod[t] * x_start + 
                self.sqrt_one_minus_alphas_cumprod[t] * noise).float()

    def p_mean_variance(self, x, t, clip_denoised=True):
        model_output = self.model(x).float()
        posterior_mean = (
            self.sqrt_alphas_cumprod_prev[t] * x - self.betas[t] * model_output
        ) / torch.sqrt(1.0 - self.alphas_cumprod_prev[t])
        posterior_variance = self.posterior_variance[t]
        return posterior_mean, posterior_variance

    def p_sample(self, x, t):
        posterior_mean, posterior_variance = self.p_mean_variance(x, t)
        noise = torch.randn_like(x) if t > 0 else 0
        return posterior_mean + torch.sqrt(posterior_variance) * noise

    def sample(self, shape):
        x = torch.randn(shape, device=self.device).float()
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t)
        return x
