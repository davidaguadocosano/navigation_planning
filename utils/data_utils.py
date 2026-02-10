import torch
import numpy as np
from typing import Tuple


def actions2numpy(actions, nodes, num_dirs=4, time_step=2e-2):
    actions = actions.squeeze(dim=0).detach().cpu().numpy()
    path = []
    for i, action in enumerate(actions):
        if i == 0:
            position = nodes[action[0], :2]
        else:
            angle = action[1] * 2 * np.pi / num_dirs
            position = position + np.array([time_step * np.cos(angle), time_step * np.sin(angle)])
        path.append([action[0], *position])
    return np.array(path)


def batch2numpy(batch):
    for k, v in batch.items():
        batch[k] = v[0].cpu().detach().numpy()
    return batch


def add_flag(vector, flag: int = 0):
    return torch.cat((vector, torch.zeros_like(vector)[..., 0, None] + flag), dim=-1)


def add_prize(vector, reward_type: str = 'const', value: int = 0):
    if reward_type == 'const':
        return torch.cat((vector, torch.zeros_like(vector)[..., 0, None] + value), dim=-1)


def generate_obstacles(min_obs: int = 5, max_obs: int = 10, r_max: float = 0.12, r_min: float = 0.02) -> torch.Tensor:  # r_max: float = 0.2, r_min: float = 0.05
    """
    Generate obstacles.

    Args:
        num_obs (int): Maximum number of obstacles.
        r_max (float): Maximum radius.
        r_min (float): Minimum radius.

    Returns:
        torch.Tensor: Obstacles.
    """
    if max_obs <= 0:
        return None

    # Number of obstacles
    num_obs = torch.randint(low=min_obs, high=max_obs + 1, size=[1])[0]

    # Generate random obstacles (circles)
    radius = torch.rand(num_obs) * (r_max - r_min) + r_min
    center = torch.rand((num_obs, 2))
    obstacles = torch.cat((center, radius[..., None]), dim=-1)

    # Pad with (-1, -1, 0) where num_obstacles < num_obs
    obstacles = torch.nn.functional.pad(
        input=obstacles,
        pad=(0, 0, 0, max_obs - obstacles.shape[0]),
        mode='constant',
        value=-1
    )
    obstacles[..., 2][obstacles[..., 2] == -1] = 0
    return obstacles


def generate_nodes(
    num_nodes: int,
    num_depots: int = 1,
    obs: torch.Tensor | None = None,
    image_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate nodes.

    Args:
        num_nodes (int): Number of nodes.
        num_depots (int): Number of depots.
        obs (torch.Tensor): Obstacles.
        image_size (int): Binary map shape (assume squared image).

    Returns:
        tuple: nodes, initial depot, and end depot.
    """
    node_size = 0.02
    num_nodes = num_nodes + num_depots

    # Meshgrid
    num_obs = obs.shape[0]
    x, y = torch.meshgrid(torch.linspace(0, 1, image_size), torch.linspace(0, 1, image_size), indexing="ij")
    xy = torch.stack((x, y), dim=-1).expand([num_obs, image_size, image_size, 2])

    # No obstacles
    if obs is None:
        points = torch.FloatTensor(num_nodes, 2).uniform_(0, 1)
        obs_mask = torch.ones(image_size, image_size)

    # Obstacles
    else:
        obs_center = obs[..., :2]
        obs_radius = obs[..., 2]

        # Calculate distance squared from each point of the meshgrid to each obstacle center
        distances = (xy - obs_center[..., None, None, :]).norm(2, dim=-1)

        # Create the masks by comparing distances with squared radius
        obs_mask = (distances > obs_radius[..., None, None] + node_size).all(dim=0).T

        # Generate non-colliding points
        non_colliding_points = torch.nonzero(obs_mask)
        non_colliding_points = torch.stack((non_colliding_points[..., 1], non_colliding_points[..., 0]), dim=-1)
        points = non_colliding_points[torch.randperm(non_colliding_points.shape[0])[:num_nodes]] / float(image_size - 1)
        obs_mask = ~obs_mask
    
    # Separate regions and depots
    depot_ini = points[0] if num_depots > 0 else None
    depot_end = points[1] if num_depots == 2 else points[0]
    nodes = points[num_depots:]
    
    # Generate binary map
    binary_map = torch.zeros(image_size, image_size, 2 + num_depots)
    binary_map[..., 0] = ((xy[0, None] - nodes[..., None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0).permute(1, 0)
    binary_map[..., 1] = obs_mask
    binary_map[binary_map > 1] = 1
    if num_depots > 0:
        binary_map[..., 2] = ((xy[0, None] - depot_ini[None, None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0)
    if num_depots == 2:
        binary_map[..., 3] = ((xy[0, None] - depot_end[None, None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0)
    return nodes, depot_ini, depot_end, binary_map.permute(2, 0, 1)
