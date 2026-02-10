import torch
import numpy as np


class TSPAutoRegressive(torch.nn.Module):
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_dirs: int = 4,
        time_step: float = 2e-2,
        max_iters: int = 300,
        *args, **kwargs) -> None:
        super(TSPAutoRegressive, self).__init__()
        
        # State embedding
        self.state_embedding = torch.nn.Linear(2, hidden_dim)
        
        # Node prediction layers
        self.node_policy = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        
        # AR parameters
        self.num_dirs = num_dirs
        self.time_step = time_step
        self.max_iters = max_iters
        
        # Direction prediction layers
        num_maps, conv_dim, patch_size = 4, 4, 16
        self.path_prediction = torch.nn.Sequential(
            torch.nn.Conv2d(num_maps * 2, conv_dim, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(conv_dim, affine=True),
            torch.nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(conv_dim, affine=True),
            torch.nn.Flatten(),
            torch.nn.Linear(conv_dim * patch_size * patch_size // 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim, affine=True),
            torch.nn.Linear(hidden_dim, num_dirs),
        )
        
        # Decode type (train mode or eval mode)
        self.train_mode = False
        self.return_actions = True
        
    def state_args(self):
        return self.args
    
    def set_train(self, train=False, return_actions=True, *args, **kwargs):
        self.train_mode = train
        self.return_actions = return_actions
        
    def forward(self, embeddings, state):
        rewards, log_probs, actions = 0, 0, ()
        
        # Update some state parameters
        state =  state._replace(num_dirs=self.num_dirs, time_step=self.time_step, max_iters=self.max_iters)
        
        # Create obstacle map for direction prediction
        obs_data = create_obs_map(state.obstacles)
        
        # Iterate until terminal state
        while not state.finished().all().item():
            
            # Predict next action
            action, log_prob = self.step(embeddings, state, obs_data)
            
            # Update state
            state = state.update(action)
            
            # Save reward, log_prob, and action
            rewards = rewards + state.reward
            log_probs = log_probs + log_prob
            actions = actions + (action, )
            
        # Check success
        success = state.check_success()
        
        if self.return_actions:
            return rewards, log_probs, torch.stack(actions, dim=1), success
        return rewards, log_probs
        
    def step(self, embeddings, state, obs_data=None):
        
        # Get agent position
        position = state.position.unsqueeze(dim=1).clone()
        
        # Get state embedding
        state_embedding = self.state_embedding(position)
        
        # Predict next node policy
        node_policy = self.predict_node_policy(state_embedding, embeddings, state.get_mask_nodes())
        
        # Get next node from policy
        next_node, log_prob_node = self.select_from_discrete_policy(node_policy)
        
        # Predict next direction
        dir_policy = self.predict_dir_policy(state.position, next_node, state, obs_data, state.get_mask_dirs())
        
        # Get next direction from policy
        next_dir, log_prob_dir = self.select_from_discrete_policy(dir_policy)
        
        # Combine log probabilities
        log_prob = (log_prob_node + log_prob_dir) / 2
        
        # Return selections
        return torch.stack((next_node, next_dir), dim=1), log_prob
    
    def predict_node_policy(self, state_embedding, embeddings, mask=None):
        
        # Predict policy
        policy = self.node_policy(
            query=state_embedding, key=embeddings, value=embeddings, key_padding_mask=mask
        )[1].squeeze(dim=1)
        policy[mask] = -np.inf
        return policy
    
    def predict_dir_policy(self, position, next_node, state, obs_data=None, mask=None):
        
        # Get next selected goal
        goal = state.get_node_coords(next_node)

        # Get local maps
        maps = create_local_maps(position, *obs_data, goal)

        # Apply prediction layers
        policy = self.path_prediction(maps)

        # Ban prohibited actions
        policy[mask] = -np.inf
        return policy
    
    def select_from_discrete_policy(self, policy):
        
        # Normalize logits with softmax
        policy = torch.log_softmax(policy, dim=-1)
        
        # Exploration vs exploitation: Multinomial sampling (during train)
        if self.train_mode:
            action = policy.exp().multinomial(num_samples=1).squeeze(dim=1)
        
        # Exploitation: ArgMax (during eval)
        else:
            action = policy.exp().argmax(dim=1)
        
        # Get log probabilities of chosen actions
        log_prob = policy.gather(1, action.unsqueeze(-1)).squeeze(1)
        return action, log_prob
    
    def select_from_continuous_policy(self, policy):
        mean = policy[:, 0]
        std = policy[:, 1]
        
        # Exploration vs exploitation: Multinomial sampling (during train)
        if self.train_mode:
            action = mean + std * torch.distributions.normal.Normal(loc=0, scale=1).sample(std.shape).to(mean.device)
        
        # Exploitation: ArgMax (during eval)
        else:
            action = mean
            
        # Get log probabilities
        log_prob = torch.distributions.normal.Normal(loc=0, scale=1).log_prob(action)
        return action.tanh(), log_prob
        

def create_obs_map(obs: torch.Tensor, patch_size: int = 16, map_size: int = 64):
    """
    Create a map of the scenario representing the obstacles as bidimensional gaussian distributions.

    Args:
        obs (torch.Tensor): The tensor representing the obstacles.
        patch_size (int): The size of the patches.
        map_size (int): The size of the map.

    Returns:
        tuple: The obstacle map and grid.
    """

    batch_size, num_obs, _ = obs.shape
    device = obs.device
    padding = patch_size // 2  # To ensure that patches do not exceed image boundaries

    # Define meshgrid
    x = torch.linspace(0, map_size, map_size).to(device)
    y = torch.linspace(0, map_size, map_size).to(device)
    x, y = torch.meshgrid(x, y, indexing="ij")
    x = x.expand(batch_size, map_size, map_size)
    y = y.expand(batch_size, map_size, map_size)

    # Calculate global map
    z = torch.zeros(batch_size, map_size, map_size).to(device)
    for i in range(num_obs):
        x0 = obs[:, i, 0].view(-1, 1, 1) * map_size
        y0 = obs[:, i, 1].view(-1, 1, 1) * map_size
        s = obs[:, i, 2].view(-1, 1, 1) * map_size + 0.01
        g = 1 / (2 * torch.pi * s * s) * torch.exp(
            -(torch.div((x - x0) ** 2, (2 * s ** 2)) + torch.div((y - y0) ** 2, (2 * s ** 2)))
        )  # https://stackoverflow.com/questions/69024270/how-to-create-a-normal-2d-distribution-in-pytorch
        max_g = g.view(batch_size, -1).max(dim=-1).values
        w = torch.where((max_g > 0).view(-1, 1, 1), g / max_g.view(-1, 1, 1), g)
        z += w
    z = z.permute(0, 2, 1)
    z = torch.nn.functional.pad(z, (padding, padding, padding, padding), mode='constant', value=0)

    # Create meshgrid of coordinates for each batch element
    grid_range = torch.arange(0, patch_size).to(device)
    grid_x, grid_y = torch.meshgrid(grid_range, grid_range, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(batch_size, patch_size, patch_size, 2)
    return z, grid


def create_local_maps(
        pos: torch.Tensor,
        obs_map: torch.Tensor,
        obs_grid: torch.Tensor,
        goal: torch.Tensor,
        patch_size: int = 16,
        map_size: int = 64) -> torch.Tensor:
    """
    Create local maps.

    Args:
        pos (torch.Tensor): The position tensor.
        obs_map (torch.Tensor): The obstacle map.
        obs_grid (torch.Tensor): The obstacle grid.
        goal (torch.Tensor): The goal tensor.
        patch_size (int): The patch size.
        map_size (int): The size of the map.

    Returns:
        torch.Tensor: The local maps.
    """
    # log_dirs = int(np.log2(num_dirs))

    # Get agent position in obstacle map
    map_x = torch.floor(pos[..., 0] * map_size).type(torch.long)
    map_x[map_x < 0] = 0
    map_x[map_x >= map_size] = map_size - 1
    map_y = torch.floor(pos[..., 1] * map_size).type(torch.long)
    map_y[map_y < 0] = 0
    map_y[map_y >= map_size] = map_size - 1

    # Get obstacles patches
    offset_x = map_x[:, None, None] + obs_grid[..., 0]
    offset_y = map_y[:, None, None] + obs_grid[..., 1]
    patches = obs_map[torch.arange(obs_map.size(0))[:, None, None], offset_y, offset_x].permute(0, 2, 1)
    north = patches[:, patch_size // 2:, :].permute(0, 2, 1)
    south = patches[:, :patch_size // 2, :].permute(0, 2, 1)
    west = patches[:, :, :patch_size // 2]
    east = patches[:, :, patch_size // 2:]
    obs_patches = torch.stack((east, north, west, south), dim=1)
    # obs_patches, subpatch = [], patch_size // log_dirs
    # for i in range(log_dirs):
    #     for j in range(log_dirs):
    #         obs_patches.append(patches[:, i * subpatch:(i + 1) * subpatch, j * subpatch:(j + 1) * subpatch])
    # obs_patches = torch.stack(obs_patches, dim=1)

    # Get goal patches (project the position of the goal into the local map)
    goal_x = torch.floor(goal[..., 0] * map_size).type(torch.long)
    condition = goal_x > map_x + patch_size // 2
    goal_x[condition] = map_x[condition] + patch_size // 2
    condition = goal_x < map_x - patch_size // 2
    goal_x[condition] = map_x[condition] - patch_size // 2
    goal_x = goal_x - map_x + patch_size // 2
    goal_y = torch.floor(goal[..., 1] * map_size).type(torch.long)
    condition = goal_y > map_y + patch_size // 2
    goal_y[condition] = map_y[condition] + patch_size // 2
    condition = goal_y < map_y - patch_size // 2
    goal_y[condition] = map_y[condition] - patch_size // 2
    goal_y = goal_y - map_y + patch_size // 2
    gx = -(obs_grid[..., 0] - goal_x.view(-1, 1, 1)) ** 2 / (2 * (patch_size // 8) ** 2)
    gy = -(obs_grid[..., 1] - goal_y.view(-1, 1, 1)) ** 2 / (2 * (patch_size // 8) ** 2)
    patches = 1 / (2 * torch.pi * (patch_size // 8) ** 2) * torch.exp(gx + gy)

    # Combine goal patches
    north = patches[:, patch_size // 2:, :].permute(0, 2, 1)
    south = patches[:, :patch_size // 2, :].permute(0, 2, 1)
    west = patches[:, :, :patch_size // 2]
    east = patches[:, :, patch_size // 2:]
    goal_patches = torch.stack((east, north, west, south), dim=1)
    # goal_patches, subpatch = [], patch_size // log_dirs
    # for i in range(log_dirs):
    #     for j in range(log_dirs):
    #         goal_patches.append(patches[:, i * subpatch:(i + 1) * subpatch, j * subpatch:(j + 1) * subpatch])
    # goal_patches = torch.stack(goal_patches, dim=1)

    # Return obstacle patches and goal patches
    return torch.concat((obs_patches, goal_patches), dim=1)
