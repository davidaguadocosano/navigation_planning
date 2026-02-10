import time

from utils import *
from nets import load_model, models
from envs import get_generator, envs, scenarios, print_results


GRAPH_ENCODERS = list(models()['encoders']['graph_encoders'].keys())
IMAGE_ENCODERS = list(models()['encoders']['image_encoders'].keys())
DECODERS = models()['decoders']
ENVS = list(envs().keys())
SCENARIOS = scenarios()


def get_options(args: list = None) -> argparse.Namespace:
    """
    Parse command line arguments to configure testing options for evaluating the neural network model.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="Random seed to use")

    # Model
    parser.add_argument('--load_path', type=str, default='', help="Path to load model from")
    parser.add_argument('--decoder', type=str, default='tsp_decoder', help=f"Decoder: {', '.join(DECODERS)}. You may indicate a path to load instead")
    parser.add_argument('--num_dirs', type=int, default=8, help=f"Number of directions the agent can turn")
    parser.add_argument('--graph_encoder', type=str, default='gtn', help=f"Graph encoder: {', '.join(GRAPH_ENCODERS)}. You may indicate a path to load instead")
    parser.add_argument('--image_encoder', type=str, default='vit', help=f"Image encoder: {', '.join(IMAGE_ENCODERS)}. You may indicate a path to load instead")
    parser.add_argument('--image_size', type=int, default=64, help=f"Binary map shape (assume squared image)")
    parser.add_argument('--batch_size', type=int, default=2048, help="Number of instances per batch during training")

    # Data
    parser.add_argument('--env', type=str, default='tsp', help=f"Training environment: {', '.join(ENVS)}")
    parser.add_argument('--scenario', type=str, default='contrastive', help=f"Type of scenario: {', '.join(SCENARIOS)}")
    parser.add_argument('--num_nodes', type=int, default=20, help="Number of visitable nodes")
    parser.add_argument('--num_obs', type=int, default=10, help="Number of avoidable obstacles (0 to not use obstacles)")
    parser.add_argument('--test_size', type=int, default=10000, help="Number of instances per epoch during training")
    
    # Device
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    parser.add_argument('--num_workers', type=int, default=12, help="Number of parallel workers loading data batches")
    opts = parser.parse_args(args)

    # Use CUDA or CPU
    opts.use_distributed = False
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda
    return opts


def main(opts):
    """
    Main function to evaluate the neural network model on given datasets.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """

    # Device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Load model
    model, *_ = load_model(opts, dataparallel=False)

    # Test dataloader
    test_dataloader = get_generator(
        env=opts.env,
        num_samples=opts.test_size,
        num_nodes=opts.num_nodes,
        num_obs=opts.num_obs,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        use_cuda=opts.use_cuda,
        use_distributed=opts.use_distributed,
        num_workers=opts.num_workers
    )

    # Get batches from dataset
    results = [[] for _ in range(5)]
    for batch in tqdm(test_dataloader, desc='Testing'):
        batch = move_to(batch, device=opts.device)

        # Run episode
        with torch.no_grad():
            start = time.time()
            rewards, _, actions, success = model(batch)
            duration = time.time() - start

        # Collect results
        results[0] = [*results[0], *rewards.tolist()]
        results[1] = [*results[1], *actions.tolist()]
        results[2] = [*results[2], *success.tolist()]
        results[3] = [*results[3], duration]
        results[4] = [*results[4], opts.num_nodes]

    # Add parallelism info to results
    parallelism = opts.batch_size
    results.append(parallelism)

    # Print results
    print_results(opts.env)(results)
    print('Finished')


if __name__ == "__main__":
    main(get_options())
