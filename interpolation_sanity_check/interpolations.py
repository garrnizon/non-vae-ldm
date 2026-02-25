import torch

def interpolate_linear(z: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Linear interpolation between two latent vectors.
    z: (2, seq_len, z_channels)
    returns: (steps, seq_len, z_channels)
    """
    z0, z1 = z[0], z[1]
    ts = torch.linspace(0, 1, steps, device=z.device, dtype=z.dtype)
    return (1 - ts[:, None, None]) * z0[None] + ts[:, None, None] * z1[None]

def interpolate_slerp(z: torch.Tensor, steps: int = 10, dot_thr: float = 0.9995) -> torch.Tensor:
    z0 = z[0].flatten()   # (D,)
    z1 = z[1].flatten()   # (D,)
    ts = torch.linspace(0, 1, steps, device=z.device, dtype=z.dtype)  # (steps,)
    
    dot = (torch.nn.functional.normalize(z0, dim=0) *
           torch.nn.functional.normalize(z1, dim=0)).sum().clamp(-1, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # if torch.abs(dot) > dot_thr:
    #     s0 = (1 - ts).unsqueeze(1)   # (steps, 1) - broadcasts over D
    #     s1 = ts.unsqueeze(1)
    #     print('WO ', end='')
    # else:
    s0 = (torch.sin(theta * (1 - ts)) / sin_theta).unsqueeze(1)  # (steps, 1)
    s1 = (torch.sin(theta * ts) / sin_theta).unsqueeze(1)
    
    # s0 * z0 + s1 * z1 with broadcasting: (steps, 1) * (D,) -> (steps, D)
    result = s0 * z0 + s1 * z1
    return result.reshape([steps] + list(z.shape[1:]))
