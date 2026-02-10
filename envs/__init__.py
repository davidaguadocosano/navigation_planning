from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .op import OPGenerator, OPState
from .tsp import TSPGenerator, TSPState, print_tsp_results


def envs():
    return {
        'tsp': TSPState,
        'op': OPState,
    }


def scenarios():
    return ['contrastive', 'image', 'graph']


def get_state(env: str):
    return envs()[env]


def get_generator(
    env: str,
    num_samples: int,
    num_nodes: int,
    num_obs: int,
    image_size: int,
    batch_size: int,
    use_cuda: bool = False,
    use_distributed: int = False,
    num_workers: int = 0):
    
    # Get class
    generator = {
        'tsp': TSPGenerator,
        'op': OPGenerator,
    }[env]
    
    # Create dataset
    dataset = generator(
        num_samples=num_samples,
        num_nodes=num_nodes,
        num_obs=num_obs,
        image_size=image_size,
    )
    
    # Create DistributedSampler (if distributed training)
    sampler = DistributedSampler(dataset) if use_cuda and use_distributed else None
    
    # Create and return Dataloader
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler
    )


def special_args(env: str):
    
    # Get dimension of nodes in the graph
    if env == 'tsp':
        node_dim1 = 3  # x, y, class (0: node, 1: obstacle)
    elif env == 'op':
        node_dim1 = 4  # x, y, reward, class
    else:
        assert True, "Environment not listed!"
    
    # Get dimension of obstacles in the graph
    node_dim2 = 4  # x, y, radious, class
    
    # Get number of image channels
    if env == 'tsp':
        num_channels = 2  # nodes, obstacles
    elif env == 'op':
        num_channels = 4  # nodes, obstacles, depot_ini, depot_end
    else:
        assert True, "Environment not listed!"
    
    return node_dim1, node_dim2, num_channels


def print_results(env: str):
    return {
        'tsp': print_tsp_results,
    }[env]
    
