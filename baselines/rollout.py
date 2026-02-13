import copy
import torch
import argparse
from typing import Any, Tuple
from scipy.stats import ttest_rel

from utils import get_inner_model, validate
from .basic import BasicBaseline
from envs import get_generator


class RolloutBaseline(BasicBaseline):

    def __init__(self, model: torch.nn.Module, loss_func:Any, opts: argparse.Namespace, epoch: int = 0) -> None:
        """
        Initialize the RolloutBaseline.

        Args:
            model (torch.nn.Module): Model.
            loss_func (Any): Loss function.
            opts: Options.
            epoch (int): Epoch number (default is 0).
        """
        super(BasicBaseline, self).__init__()
        self.opts = opts
        self.device = opts.device #dac
        self.loss_func = loss_func
        self._update_model(model, epoch)

    def _update_model(self, model: torch.nn.Module, epoch: int) -> None:
        """
        Update the baseline model.

        Args:
            model (torch.nn.Module): Model.
            epoch (int): Epoch number.
            env (Any): Environment.
        """

        # Get current epoch
        self.epoch = epoch

        # Copy model for baseline
        self.model = copy.deepcopy(model)

        # Generate new baseline data when updating model to prevent overfitting to the baseline dataset
        self.dataloader = get_generator(
            env=self.opts.env,
            num_samples=self.opts.val_size,
            num_nodes=self.opts.num_nodes,
            num_obs=self.opts.num_obs,
            image_size=self.opts.image_size,
            batch_size=self.opts.batch_size_val,
            use_cuda=self.opts.use_cuda,
            use_distributed=self.opts.use_distributed,
            num_workers=self.opts.num_workers
        )

        # Rollout and get reward
        self.reward_bl = validate(self.model, self.dataloader, self.loss_func, self.opts.device, text=f"Baseline").cpu().numpy()
        self.mean = self.reward_bl.mean()

    def eval(self, x: dict | torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor | float, torch.Tensor | float]:
        """
        Evaluate the baseline.

        Args:
            x (dict or torch.Tensor): Input batch.
            c (torch.Tensor): Cost (negative reward) found by the model.
            e (Any): Environment.

        Returns:
            tuple: Tuple containing the baseline value and the loss.
        """

        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)

        # There is no loss
        return v, 0

    def epoch_callback(self, model: torch.nn.Module, epoch: int) -> None:
        """
        Epoch callback. Challenges the current baseline with the model and replaces the baseline model if improved.

        Args:
            model (torch.Tensor): Model.
            epoch (int): Epoch number.
        """

        # Evaluate candidate model on evaluation dataset
        candidate_vals = validate(self.model, self.dataloader, self.loss_func, self.device, text=f"Baseline").cpu().numpy()
        candidate_mean = candidate_vals.mean()
        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch}, "
              f"mean {self.mean}, difference {candidate_mean - self.mean}")

        # Calc p-value
        if candidate_mean - self.mean < 0:
            t, p = ttest_rel(candidate_vals, self.reward_bl)
            assert t < 0, "T-statistic should be negative"
            p_val = p / 2  # one-sided
            print("p-value: {}".format(p_val))

            # Update baseline
            if p_val < self.opts.bl_alpha:
                print("Update baseline")
                self._update_model(model, epoch)

    def state_dict(self) -> dict:
        """
        Get the state dictionary.

        Returns:
            dict: State dictionary.
        """
        return {'model': self.model, 'epoch': self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state dictionary of the baseline.

        Args:
            state_dict: State dictionary to load.
        """
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'])
