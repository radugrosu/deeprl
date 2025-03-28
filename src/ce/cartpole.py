#!/usr/bin/env python3
from pathlib import Path
from numpy.typing import NDArray
import typer
from typing import TypeAlias
import gymnasium as gym
import numpy as np
from jaxtyping import Float32
import torch.optim as optim

from src.ce.ce_utils import (
    Net,
    SimpleStoppingCriterion,
    train_crossentropy,
)
from utils import ensure_dir_exists # type: ignore


ObsType: TypeAlias = Float32[NDArray[np.float32], "4"]
ActType: TypeAlias = np.int64


def main(
    hidden_size: int = 128,
    batch_size: int = 16,
    lr: float = 0.01,
    percentile: float = 70,
    target_reward: float = 200.0,
    epsiode_discount_factor: float = 1.0,
    max_iter: int = 100,
    log_level: str = "INFO",
    disable_logging: bool = False,
    tb_root: Path = typer.Option(
        default="runs", file_okay=False, callback=ensure_dir_exists
    ),
    video_log_root: Path = typer.Option(
        default="video", file_okay=False, callback=ensure_dir_exists
    ),
):
    env: gym.Env[ObsType, ActType] = gym.make(  # type: ignore
        "CartPole-v1", render_mode="rgb_array" if video_log_root else None
    )
    obs_size = int(env.observation_space.shape[0])  # type: ignore
    n_actions = int(env.action_space.n)  # type: ignore
    net = Net(obs_size, hidden_size, n_actions)
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
    stopping_criterion = SimpleStoppingCriterion(max_iter, target_reward)
    train_crossentropy(
        net,
        optimizer,
        env,
        stopping_criterion,
        f"h={hidden_size},p={percentile},b={batch_size},lr={lr}",
        batch_size,
        percentile,
        epsiode_discount_factor,
        log_level,
        disable_logging,
        tb_root,
    )


if __name__ == "__main__":
    typer.run(main)
