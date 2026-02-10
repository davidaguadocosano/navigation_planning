import torch
from typing import NamedTuple, Any


class OPState(NamedTuple):
    """Navigation Orienteering Problem (NOP) State class. Contains non-static info about the episode"""

    # Scenario
    nodes: torch.Tensor         # Visitable regions/nodes
    obstacles: torch.Tensor     # Obstacles
    max_length: torch.Tensor    # Max length

    # State
    position: torch.Tensor      # Current agent position
    length: torch.Tensor        # Time/distance of the travel
    bumped: torch.Tensor        # Indicates if agent has bumped into an obstacle
    visited: torch.Tensor       # Keeps track of nodes that have been visited
    is_traveling: torch.Tensor  # Indicates if agent has reached a region or is traveling
    prev_node: torch.Tensor     # Previous node selected to visit
    prev_dir: torch.Tensor      # Previous direction selected to visit
    
    # Training
    reward: torch.Tensor        # Last collected reward
    done: bool                  # Terminal state for every element of the batch
    
    # Misc
    i: torch.Tensor             # Count iterations
    max_iters: int              # Max iterations
    time_step: float            # Duration/length of a step
    num_dirs: int               # Number of directions to follow

    def __getitem__(self, key):
        """
        Get item(s) from the state using indexing.

        Args:
            key: Index or slice to retrieve.

        Returns:
            NopState: New state containing selected elements.
        """
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            nodes=self.nodes[key],
            obstacles=self.obstacles[key],
            position=self.position[key],
            length=self.length[key],
            max_length=self.max_length[key],
            visited=self.visited[key],
            prev_node=self.prev_node[key],
            prev_dir=self.prev_dir[key],
            is_traveling=self.is_traveling[key],
            reward=self.reward[key],
        )

    @staticmethod
    def initialize(data: dict | torch.Tensor, time_step: float = 2e-2, num_dirs: int = 4, max_iters: int = 200) -> Any:
        """
        Initialize the state.

        Args:
            data (dict): Input batch.
            time_step (float): Time step.

        Returns:
            NopState: Initialized state.
        """

        # Device
        device = data[list(data.keys())[0]].device
        batch_size, num_nodes, _ = data['nodes'].shape

        # Nodes
        nodes = data['nodes'][..., :2]
        
        # Obstacles
        obstacles = data['obstacles'][..., :3]
        bumped = torch.zeros(batch_size, dtype=torch.int64, device=device) if obstacles is not None else None
        
        # Max length
        max_length = data['max_length']

        # Position of the agent
        position = torch.zeros_like(nodes[:, 0])

        # Traveled length
        length = torch.zeros(batch_size, device=device)
        
        # Mask of visited nodes
        visited = torch.zeros(size=(batch_size, num_nodes), dtype=torch.bool, device=device)

        # Previously visited node
        prev_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        visited = torch.zeros(size=(batch_size, num_nodes), dtype=torch.bool, device=device)

        # Previously chosen direction
        prev_dir = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Traveling info
        is_traveling = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Reward and done
        reward = torch.zeros(1, device=device)
        done = False

        # Create State
        return OPState(
            i=torch.tensor(0, device=device),
            max_iters=max_iters,
            time_step=time_step,
            num_dirs=num_dirs,
            nodes=nodes,
            obstacles=obstacles,
            bumped=bumped,
            position=position,
            length=length,
            max_length=max_length,
            visited=visited,
            prev_node=prev_node,
            prev_dir=prev_dir,
            is_traveling=is_traveling,
            reward=reward,
            done=done,
        )
        
    def update(self, action):
        
        # Get actions
        next_node = action[:, 0].type(torch.int64)
        next_dir = action[:, 1]
        
        if self.i == 0:
            return self._replace(
                i=self.i + 1,
                prev_node=next_node,
                position=self.get_next_coords(next_node),
                visited=self.visited.scatter(dim=-1, index=next_node[..., None], value=1),
            )
        
        # Check non-terminal states
        nf = ~self.finished()
        
        # Update next position
        polar = torch.polar(
            torch.zeros_like(self.position[:, 0]) + self.time_step, next_dir * 2 * torch.pi / self.num_dirs
        )
        new_position = self.position.clone()
        new_position[nf] = self.position[nf] + torch.stack((polar.real, polar.imag), dim=-1)[nf]
        
        # Update length of route
        new_length = self.length + (new_position - self.position).norm(p=2, dim=-1)

        # Distance to next node
        dist2next = self.get_dist2next(next_node, new_position)

        # Check whether the agent has just arrived to next node or it is still traveling
        is_traveling = self.is_traveling
        is_traveling[dist2next > self.time_step] = True  # Traveling
        is_traveling[dist2next <= self.time_step] = False  # Not traveling
        is_traveling[~nf] = False  # Once on terminal state, do not travel

        # Mask visited regions
        visited = self.visited.clone()
        visited = visited.scatter(dim=-1, index=next_node[..., None], value=1)

        # Reward
        reward = torch.zeros(nf.shape[0], device=self.device)
        
        # Penalty: traveled length
        reward[nf] = 0.3 * dist2next[nf]
        
        # Reward: visiting next node
        condition = torch.logical_and(
            torch.logical_and(~is_traveling, nf),  # Agent is not traveling and has not finished yet
            dist2next <= self.time_step  # Next node is visited
        )
        reward[condition] += 60 * self.get_next_coords(next_node)[..., -1][condition] / (
                self.nodes.shape[1] - 2
        )[condition]
        
        # Reward: reaching end depot within time limit
        dist2end = (new_position - self.get_next_coords(torch.ones_like(next_node))).norm(p=2, dim=-1)
        condition = torch.logical_and(
            torch.logical_and(
                torch.eq(next_node, torch.ones_like(next_node)),  # Next node == end depot
                (dist2end <= self.time_step)  # End depot visited
            ),
            nf  # Not finished
        )
        reward[condition] = reward[condition] + 20
        
        # Penalty: not reaching end depot within time limit
        condition = torch.logical_and(
            torch.logical_and(
                (self.max_length - new_length < 0),  # Time limit is surpassed
                (dist2end > self.time_step)  # End depot not visited
            ),
            nf  # Not finished
        )
        reward[condition] = reward[condition] - 5
        
        # Penalty: bumping into obstacles
        bumped = self.bumped
        if self.obstacles is not None:
            bumped[
                torch.ge(
                    self.obstacles[..., 2],
                    self.get_dist2obs(position=new_position)
                ).any(dim=-1)
            ] = True  # bumped = 1 if agent is inside obstacle
            condition = torch.logical_and(
                bumped,  # Bumped into obstacle
                nf  # Not finished
            )
            reward[condition] = reward[condition] - 5
        
        # Update state
        return self._replace(
            i=self.i + 1,
            prev_node=next_node,
            prev_dir=next_dir,
            position=new_position,
            length=new_length,
            visited=visited,
            is_traveling=is_traveling,
            bumped=bumped,
            reward=-reward,   # Negative reward since torch always minimizes
            done=self.finished().all().item()
        )
        
        
    def finished(self):
        
        # Check conditions for finishing episode
        return torch.logical_and(
            self.i > 0,  # Not first step
            torch.logical_or(
                torch.logical_and(
                    torch.eq(self.prev_node, torch.ones_like(self.prev_node)),  # Going to end depot
                    (self.position - self.nodes[:, 1]).norm(p=2, dim=-1) <= self.time_step  # On end depot
                ),
                torch.logical_or(
                    self.bumped,  # No bumping
                    self.get_remaining_length() < 0  # Still on time
                )
            )
        )

    def get_remaining_length(self) -> torch.Tensor:
        """
        Get the remaining length at each moment.

        Returns:
            torch.Tensor: Remaining length for each element of the batch.
        """
        return self.max_length - self.length
    
    def get_next_coords(self, next_node: torch.Tensor) -> torch.Tensor:
        next_node = next_node.contiguous().view(-1, 1, 1).expand(self.nodes.shape[0], 1, self.nodes.shape[-1])
        next_coords = self.nodes.gather(index=next_node, dim=1).squeeze(dim=1)
        return next_coords
    
    def get_dist2next(self, next_node: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        next_coords = self.get_next_coords(next_node)
        dist2next = (position - next_coords).norm(p=2, dim=-1)
        return dist2next
    
    def get_dist2obs(self, position: torch.Tensor = None) -> torch.Tensor:
        """
        Get the distance to obstacles.

        Args:
            position (torch.Tensor): Current position.

        Returns:
            torch.Tensor: Distance to obstacles for each element of the batch.
        """
        if self.obstacles is None:
            return None
        if position is None:
            position = self.position
        return (position[:, None] - self.obstacles[..., :2]).norm(p=2, dim=-1)

    def get_mask_dirs(self) -> torch.Tensor:
        """
        Get the mask for directions.

        Returns:
            torch.Tensor: Mask for directions.
        """

        # Initialize mask
        mask = torch.zeros((self.nodes.shape[0], self.num_dirs), dtype=torch.bool, device=self.nodes.device)

        # Ban directions that lead out of the map | TODO: adapt for more than 4 directions
        mask[self.position[..., 0] + self.time_step > 1, 0] = 1
        mask[self.position[..., 1] + self.time_step > 1, 1] = 1
        mask[self.position[..., 0] - self.time_step < 0, 2] = 1
        mask[self.position[..., 1] - self.time_step < 0, 3] = 1

        # No more restrictions are required during the first step, so return the mask
        if self.i < 1:
            return mask.bool()

        # Avoid performing the action opposite to that performed before
        banned_dirs = self.prev_dir[..., None] + self.num_dirs / 2
        banned_dirs[banned_dirs > self.num_dirs - 1] -= self.num_dirs
        mask = mask.scatter(-1, banned_dirs.long(), 1)
        return mask.bool()

    def get_mask_nodes(self) -> torch.Tensor:
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions, depends on already visited and remaining
        capacity. 0 = feasible, 1 = infeasible. Forbids to visit depot twice in a row, unless all nodes have been
        visited.

        Returns:
            torch.Tensor: Mask for nodes.
        """
        batch_ids = torch.arange(self.nodes.shape[0], dtype=torch.int64, device=self.nodes.device)

        # Define mask (with visited nodes)
        mask = self.visited.clone()

        # While traveling or when finished, do not change agent's mind
        condition = torch.logical_or(self.is_traveling, self.finished())
        mask[condition] = 1
        
        # Allow always to visit next node
        mask[batch_ids[condition], self.prev_node[condition]] = 0
        
        # Add obstacles as non-visitable nodes
        mask = torch.cat((mask, torch.ones_like(mask[:, :self.obstacles.shape[1]])), dim=1)
        return mask
