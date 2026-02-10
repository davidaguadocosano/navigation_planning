import numpy as np
import matplotlib.pyplot as plt


def plot_tsp(actions: np.ndarray, batch: dict, reward: int = 0) -> None:

    # Initialize plot
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])

    # Data
    nodes = batch['nodes']

    # Plot nodes
    plt.scatter(nodes[..., 0], nodes[..., 1], c='mediumpurple', s=180)
    if 'obstacles' in batch:
        for obs in batch['obstacles']:
            ax.add_patch(plt.Circle(obs[:2], obs[2], color='k'))

    # Plot regions numbers (indexes)
    for i in range(nodes.shape[0]):
        plt.text(nodes[i, 0], nodes[i, 1], str(i))

    # Draw arrows
    d = 0
    for i in range(1, len(actions)):
        
        # Update traveled distance
        d += np.linalg.norm(actions[i, 1:] - actions[i - 1, 1:])
        
        # Plot new position
        plt.plot([actions[i - 1, 1], actions[i, 1]], [actions[i - 1, 2], actions[i, 2]], c='g')
        
        # Check if finished
        dist2obs = np.linalg.norm(actions[i, 1:] - batch['obstacles'][:, :2], axis=-1)
        if np.any(dist2obs < batch['obstacles'][:, 2]):
            plt.scatter(*actions[i, 1:], marker='x', c='r', s=90)
            break
    
    # Title
    title = f"TSP Length: {d:.2f} | Reward: {-reward:.2f}"
    plt.title(title)
    print(title)
    
    # Show plot
    plt.show()

