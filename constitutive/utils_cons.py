import torch


def rotation_matrix(theta: torch.tensor, phi: torch.tensor)-> torch.tensor:
    r"""
    从原始坐标系，绕轴[nx, ny, nz]旋转\theta
    https://www.cnblogs.com/uestc-mm/p/15697073.html
    
    """
    nx, ny, nz = torch.sin(phi), - torch.cos(phi), 0.

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    R = torch.tensor([
        [nx**2 * (1-cos_theta)+cos_theta, nx*ny*(1-cos_theta), ny*sin_theta],
        [nx*ny*(1-cos_theta), ny**2 * (1-cos_theta)+cos_theta, -nx * sin_theta],
        [-ny*sin_theta, nx*sin_theta, cos_theta]
    ], dtype=torch.float32)
    return R