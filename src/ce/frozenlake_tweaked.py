#!/usr/bin/env python3
from pathlib import Path
from typing import TypeAlias
import gymnasium as gym
import numpy as np
from jaxtyping import Float32
from numpy.typing import NDArray
import torch.optim as optim
import typer

from src.ce.ce_utils import (
    Net,
    SimpleStoppingCriterion,
    train_crossentropy_elite_batches,
)

from utils import ensure_dir_exists  # type: ignore

ObsType: TypeAlias = np.int64
WrapperObsType: TypeAlias = Float32[NDArray[np.float32], "16"]
ActType: TypeAlias = np.int64


class DiscreteOneHotWrapper(gym.ObservationWrapper[WrapperObsType, ActType, ObsType]):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n.item(),)
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            0.0, 1.0, shape, dtype=np.float32
        )

    def observation(self, observation: ObsType) -> WrapperObsType:
        res: WrapperObsType = np.copy(self.observation_space.low)  # type: ignore
        res[observation] = 1.0
        return res


def main(
    hidden_size: int = 128,
    lr: float = 0.001,
    batch_size: int = 100,
    max_batch_size: int = 500,
    percentile: float = 30,
    target_reward: float = 0.8,
    epsiode_discount_factor: float = 0.9,
    max_iter: int = 6000,
    log_level: str = "INFO",
    disable_logging: bool = False,
    tb_root: Path = typer.Option(
        default="runs/frozenlake", file_okay=False, callback=ensure_dir_exists
    ),
    video_log_root: Path = typer.Option(
        default="video/frozenlake", file_okay=False, callback=ensure_dir_exists
    ),
    seed: int = 42,
    is_slippery: bool = False,
):
    import random

    random.seed(seed)
    wrapped: gym.Env[ObsType, ActType] = gym.make(  # type: ignore
        "FrozenLake-v1",
        render_mode="rgb_array" if video_log_root else None,
        is_slippery=is_slippery,
    )
    wrapped.reset(seed=seed)
    env: gym.Env[WrapperObsType, ActType] = DiscreteOneHotWrapper(wrapped)
    obs_size = int(env.observation_space.shape[0])  # type: ignore
    n_actions = int(env.action_space.n)  # type: ignore
    net = Net(obs_size, hidden_size, n_actions)
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
    stopping_criterion = SimpleStoppingCriterion(max_iter, target_reward)
    run = f"h={hidden_size},p={percentile},b={batch_size},mb={max_batch_size},disc={epsiode_discount_factor},lr={lr}"
    train_crossentropy_elite_batches(
        net,
        optimizer,
        env,
        stopping_criterion,
        run,
        batch_size,
        max_batch_size,
        percentile,
        epsiode_discount_factor,
        log_level,
        disable_logging,
        tb_root,
    )


if __name__ == "__main__":
    typer.run(main)
