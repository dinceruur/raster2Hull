import numpy as np
import torch
import torch.nn as nn

class MeanShapeTemplate(nn.Module):
    def __init__(self, mean_shape_path, num_points=1024, jitter_factor=0.01):
        super().__init__()

        # Load the Mean Shape (N, 3)
        mean_points = np.load(mean_shape_path)

        # Resample if point counts don't match
        if mean_points.shape[0] != num_points:
            idx = np.random.choice(
                mean_points.shape[0], num_points, replace=True)
            mean_points = mean_points[idx]

        # Register as a fixed buffer
        tensor = torch.from_numpy(mean_points).float()
        self.register_buffer("template", tensor)

        # Calculates the standard deviation along X, Y, and Z (Shape: (3,))
        std_per_axis = tensor.std(dim=0)
        
        # Multiply by the chosen factor (e.g., 1% of the std) and register as buffer
        self.register_buffer("jitter_scale", std_per_axis * jitter_factor)

    def forward(self, batch_size, apply_jitter=False):
        # (N, 3) -> (B, N, 3)
        out = self.template.unsqueeze(0).expand(batch_size, -1, -1)

        if apply_jitter:
            # Generate standard normal noise (B, N, 3)
            noise = torch.randn_like(out)
            
            # Scale the noise using the axis-aware buffer
            out = out + (noise * self.jitter_scale)

        return out