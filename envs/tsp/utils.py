import numpy as np
from datetime import timedelta


def print_tsp_results(results: tuple) -> None:
    """
    Print TSP results.

    Args:
        results (tuple): Results tuple with reward, actions, success, duration, num_nodes, and parallelism.
    """

    # Get results info
    reward, actions, success, duration, num_nodes, parallelism = results

    # Get number of nodes visited
    visits = []
    for i, action in enumerate(actions):
        nodes = np.array(action)[:, 0]
        unique_nodes = len(np.unique(nodes))
        if success[i]:
            visits.append(unique_nodes)

    # Print reward
    print("\nREWARD")
    print(f"\tAverage reward: {-np.mean(reward):.4f} +- {2 * np.std(reward) / np.sqrt(len(reward)):.4f}")
    print(f"\tMax reward: {-np.min(reward):.4f} | Min reward: {-np.max(reward):.4f}")

    # Print success rate
    print("\nSUCCESS")
    print(f"\tFound success in {np.sum(success)}/{len(success)} scenarios")
    print(f"\tRate of success: {np.mean(success):.4f} +- {2 * np.std(success) / np.sqrt(len(success)):.4f}")

    # Print number of nodes visited
    node_rate = np.array(visits) / np.mean(num_nodes)  # Max length is fixed to allow visiting half of the nodes
    print("\nNUMBER OF NODES VISITED")
    print(f"\tAverage number of nodes visited: {np.mean(visits):.4f} +- {2 * np.std(visits) / np.sqrt(len(visits)):.4f}")
    print(f"\tMax number of nodes visited: {np.max(visits):.0f} | Min number of nodes visited: {np.min(visits):.0f}")
    print(f"\tRate of nodes visited: {np.mean(node_rate):.4f} +- {2 * np.std(node_rate) / np.sqrt(len(node_rate)):.4f}")

    # Print serial duration
    mean_time, ci_time = np.mean(duration), 2 * np.std(duration) / np.sqrt(len(duration))
    print("\nTIME")
    print(f"\tAverage serial duration: {mean_time} +- {ci_time} seconds")

    # Print parallel duration
    mean_time, ci_time = np.mean(duration) / parallelism, 2 * np.std(duration) / np.sqrt(len(duration)) / parallelism
    print(f"\tAverage parallel duration: {mean_time} +- {ci_time} seconds")

    # Print total time
    total_time = np.sum(duration) / parallelism
    print(f"\tCalculated total duration: {timedelta(seconds=int(total_time))} ({total_time} seconds)\n")
