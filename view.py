from utils import *
from nets import load_model, models
from envs import get_generator, envs, scenarios


GRAPH_ENCODERS = list(models()['encoders']['graph_encoders'].keys())
IMAGE_ENCODERS = list(models()['encoders']['image_encoders'].keys())
DECODERS = models()['decoders']
ENVS = list(envs().keys())
SCENARIOS = scenarios()


def get_options(args: list = None) -> argparse.Namespace:
    """
    Parse command line arguments to configure training options for the neural network model.

    Args:
        args (list): List of command line arguments. If None, defaults to sys.argv.

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

    # Data
    parser.add_argument('--env', type=str, default='tsp', help=f"Training environment: {', '.join(ENVS)}")
    parser.add_argument('--scenario', type=str, default='contrastive', help=f"Type of scenario: {', '.join(SCENARIOS)}")
    parser.add_argument('--num_nodes', type=int, default=20, help="Number of visitable nodes")
    parser.add_argument('--num_obs', type=int, default=10, help="Number of avoidable obstacles (0 to not use obstacles)")
    
    # Device
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    opts = parser.parse_args(args)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    opts.use_cuda = opts.use_cuda if cuda_available else False
    opts.use_distributed = False
    print(f"[*] Device: {'CUDA' if opts.use_cuda else 'CPU'}{'' if cuda_available else ', since CUDA is not available'}")
    
    # Set seed for reproducibility
    set_seed(opts.seed)
    return opts
  

def main(opts):
    
    # Get device and set training parameters to fixed value
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Load model
    model, *_ = load_model(opts)
    
    # Load data
    batch = next(iter(get_generator(
        env=opts.env,
        num_samples=1,
        num_nodes=opts.num_nodes,
        num_obs=opts.num_obs,
        image_size=opts.image_size,
        batch_size=1,
        use_cuda=opts.use_cuda,
        use_distributed=opts.use_distributed,
        num_workers=0
    )))
    batch = move_to(var=batch, device=opts.device)
    
    # Make predictions
    reward, _, actions, _ = model(batch)
    reward = reward.sum().detach().cpu().numpy()
    
    # Batch tensors to numpy arrays
    batch = batch2numpy(batch)
    
    # Predictions to numpy arrays
    actions = actions2numpy(actions, batch['nodes'], num_dirs=opts.num_dirs)

    # Plot path
    for action in actions:
        print(f"Node: {int(action[0])} - Position: ({action[1]:.4f}, {action[2]:.4f}) ")
    plot_tsp(actions, batch, reward)
    print('Finished')
    
    
if __name__ == "__main__":
    main(get_options())
