import torch
import argparse
from typing import Any

from utils import get_inner_model
from nets import Critic
from .basic import NoBaseline
from .warmup import WarmupBaseline
from .critic import CriticBaseline
from .rollout import RolloutBaseline
from .exponential import ExponentialBaseline


def bls():
    return {
        None: NoBaseline,
        'rollout': RolloutBaseline,
        'critic': CriticBaseline,
        'exponential': ExponentialBaseline,
    }


def get_baseline(opts: argparse.Namespace, model: torch.nn.Module, loss_func: Any, checkpoint: dict = None, epoch: int = 0):
    """
    Load the specified baseline.

    Args:
        opts (argparse.Namespace): Options for loading the baseline.
        model (torch.nn.Module): The model.
        loss_func (str): The loss function.
        checkpoint (dict): Data to load.

    Returns:
        Baseline: The loaded baseline.
    """

    # Exponential baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.bl_exp)

    # Critic baseline (Actor-Critic)
    elif opts.baseline == 'critic':
        baseline = CriticBaseline(
            Critic(get_inner_model(model), opts.hidden_dim).to(opts.device)
        )

    # Rollout baseline
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, loss_func, opts, epoch)

    # No baseline
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    # Warmup baseline (for a few of the initial epochs)
    if opts.bl_warmup > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup, warmup_exp_beta=opts.bl_exp)

    # Load baseline from data, make sure script is called with same type of baseline
    if checkpoint is not None:
        if 'baseline' in checkpoint:
            baseline.load_state_dict(checkpoint['baseline'])
            baseline.epoch_callback(model, epoch)

    #dac
    # Load baseline state dict
    if checkpoint is not None and 'baseline' in checkpoint:
        try:
            baseline.load_state_dict(checkpoint['baseline'])
            print(f"[*] Baseline state loaded from {opts.load_path}")
        except (KeyError, TypeError):
            print("[!] Checkpoint doesn't contain a valid baseline for this scenario. Starting with a fresh baseline.")
    else:
        print("[*] No baseline found in checkpoint. Initializing fresh baseline.")
        
    return baseline
