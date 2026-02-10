import torch
from typing import Any
from torch.utils.data import Dataset

from utils import generate_nodes, generate_obstacles, add_flag


class TSPGenerator(Dataset):
    
    def __init__(self, num_samples: int, num_nodes: int, num_obs: int, image_size: int = 64) -> None:
        super(TSPGenerator).__init__()
        
        # Number of samples of the dataset
        self.num_samples = num_samples
        
        # Number of nodes of the graph
        self.num_nodes = num_nodes
        
        # Number of obstacles of the graph
        self.num_obs = num_obs
        
        # Binary map size
        self.image_size = image_size
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, index) -> Any:
        
        # Obstacles
        obstacles = generate_obstacles(max_obs=self.num_obs)
        
        # Define nodes
        nodes, _, _, binary_map = generate_nodes(
            num_nodes=self.num_nodes, num_depots=0, obs=obstacles, image_size=self.image_size
        )
        
        # Set flag: 0=nodes, 1=obstacles
        nodes = add_flag(nodes, flag=0)
        obstacles = add_flag(obstacles, flag=1)
        
        # Return dictionary with data
        output = {
            'nodes': nodes,
            'obstacles': obstacles,
            'binary_map': binary_map
        }
        return output
