import torch
from typing import Any
from torch.utils.data import Dataset

from utils import generate_nodes, generate_obstacles, add_flag, add_prize


class OPGenerator(Dataset):
    
    def __init__(self, num_samples: int, num_nodes: int, image_size: int = 64) -> None:
        super(OPGenerator).__init__()
        
        # Number of samples of the dataset
        self.num_samples = num_samples
        
        # Number of nodes of the graph
        self.num_nodes = num_nodes
        
        # Binary map size
        self.image_size = image_size
        
        # Time limit
        self.max_length = (num_nodes + 60) / 40
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, index) -> Any:
        
        # Obstacles
        obstacles = generate_obstacles()
        
        # Define nodes
        nodes, depot_ini, depot_end, binary_map = generate_nodes(
            num_nodes=self.num_nodes, num_depots=0, obs=obstacles, image_size=self.image_size
        )
        
        # Set flag: 0=nodes, 1=obstacles, depot_ini=2, depot_end=3
        nodes = add_flag(nodes, flag=0)
        obstacles = add_flag(obstacles, flag=1)
        depot_ini = add_flag(depot_ini, flag=2)
        depot_end = add_flag(depot_end, flag=3)
        
        # Add prize for visiting nodes
        nodes = add_prize(nodes, value=1)
        depot_ini = add_prize(depot_ini, value=0)
        depot_end = add_prize(depot_end, value=10)
        
        # Combine depots and nodes
        nodes = torch.cat((depot_ini.unsqueeze(dim=1), depot_end.unsqueeze(dim=1), nodes), dim=1)
        
        # Return dictionary with data
        output = {
            'nodes': nodes,
            'obstacles': obstacles,
            'binary_map': binary_map,
            'max_length': torch.tensor(self.max_length),
        }
        return output
