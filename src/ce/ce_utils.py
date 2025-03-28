from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Protocol, TypeAlias

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from jaxtyping import Float32
from loguru import logger
from numpy.typing import NDArray
from torch.utils.tensorboard.writer import SummaryWriter

from src.utils import obs2tensor, save_videos, setup_logger, tensor2probs

ObsType: TypeAlias = Float32[NDArray[np.float32], "4"]
ActType: TypeAlias = np.int64


class EpisodeStep[ObsType, ActType](NamedTuple):
    observation: ObsType
    action: ActType


class Episode[ObsType, ActType](NamedTuple):
    reward: float
    steps: list[EpisodeStep[ObsType, ActType]]


class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super().__init__()  # type: ignore
        self.activation = nn.Softmax(dim=1)
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(
        self, x: Float32[torch.Tensor, "batch obs_size"]
    ) -> Float32[torch.Tensor, "batch n_actions"]:
        return self.net(x)

    def policy(self, state: ObsType) -> NDArray[ActType]:
        obs_t = obs2tensor(state)
        act_probs_t = self.activation(self(obs_t))
        act_probs = tensor2probs(act_probs_t)
        return act_probs


class StoppingCriterion(Protocol):
    max_iter: int
    target_reward: float

    def __call__(self, iter_no: int, mean_reward: float) -> bool: ...


class SimpleStoppingCriterion(NamedTuple):
    max_iter: int = 100
    target_reward: float = 200.0

    def __call__(self, iter_no: int, mean_reward: float) -> bool:
        if mean_reward > self.target_reward:
            logger.info("Solved! Reward >= {:.0f}", mean_reward)
            return True
        if iter_no > self.max_iter:
            logger.info(
                "Could not get reward of {} >= {:.0f} in {}",
                self.target_reward,
                mean_reward,
                self.max_iter,
            )
            return True
        return False


class FrozenLakeStoppingCriterion(NamedTuple):
    max_iter: int = 100
    target_reward: float = 0.8

    def __call__(self, iter_no: int, mean_reward: float) -> bool:
        if mean_reward > self.target_reward:
            logger.info("Solved! Reward >= {:.0f}", mean_reward)
            return True
        if iter_no > self.max_iter:
            logger.info(
                "Could not get reward of {} >= {:.0f} in {}",
                self.target_reward,
                mean_reward,
                self.max_iter,
            )
            return True
        return False


def filter_batch[ObsType, ActType](
    batch: list[Episode[ObsType, ActType]],
    percentile: float,
    discount_factor: float = 1.0,
):
    discounts = np.array([discount_factor ** len(e.steps) for e in batch])
    rewards = np.array([e.reward for e in batch])
    reward_bound = np.percentile(rewards * discounts, percentile).item()
    reward_mean = np.mean(rewards).item()

    train_obs: list[ObsType] = []
    train_act: list[ActType] = []
    for reward, steps in batch:
        if reward >= reward_bound:
            for step in steps:
                train_obs.append(step.observation)
                train_act.append(step.action)

    train_obs_t = torch.from_numpy(np.array(train_obs, dtype=np.float32))  # type: ignore
    train_act_t = torch.from_numpy(np.array(train_act, dtype=np.int64))  # type: ignore
    return train_obs_t, train_act_t, reward_bound, reward_mean


def train_crossentropy(
    policy: Net,
    optimizer: optim.Optimizer,
    env: gym.Env[ObsType, ActType],
    stopping_criterion: Callable[[int, float], bool],
    run_name: str,
    batch_size: int = 16,
    percentile: float = 70,
    episode_discount_factor: float = 1.0,
    log_level: str = "INFO",
    disable_logging: bool = False,
    tb_root: Path = Path("runs"),
    video_log_root: Path = Path("videos"),
):
    setup_logger(log_level, disable_logging)
    writer = SummaryWriter(log_dir=f"{tb_root}/{run_name}")

    pbar = tqdm.tqdm()
    objective = nn.CrossEntropyLoss()
    for iter_no, batch in enumerate(iterate_batches(env, policy, batch_size)):
        obs_t, act_t, reward_b, reward_m = filter_batch(
            batch, percentile, episode_discount_factor
        )
        optimizer.zero_grad()
        act_logits = policy(obs_t)
        loss_v = objective(act_logits, act_t)
        loss_v.backward()
        optimizer.step()  # type: ignore

        pbar.set_description(f"it {iter_no}")
        pbar.update(1)
        pbar.set_postfix(  # type: ignore
            loss=f"{loss_v.item():.3f}",
            rw_mean=f"{reward_m:.1f}",
            rw_bound=f"{reward_b:.1f}",
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)  # type: ignore
        writer.add_scalar("reward_bound", reward_b, iter_no)  # type: ignore
        writer.add_scalar("reward_mean", reward_m, iter_no)  # type: ignore

        if stopping_criterion(iter_no, reward_m):
            break

    writer.close()
    if video_log_root:
        save_videos(env, policy, f"{video_log_root}/{run_name}")


def iterate_batches(
    env: gym.Env[ObsType, ActType],
    net: Net,
    batch_size: int,
) -> Iterable[list[Episode[ObsType, ActType]]]:
    batch: list[Episode[ObsType, ActType]] = []
    episode_steps: list[EpisodeStep[ObsType, ActType]] = []
    episode_reward = 0.0
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_t = obs2tensor(obs)
        act_probs_t = sm(net(obs_t))
        act_probs = tensor2probs(act_probs_t)
        action: ActType = np.random.choice(range(len(act_probs)), p=act_probs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if terminated or truncated:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch_elite[ObsType, ActType](
    batch: list[Episode[ObsType, ActType]],
    percentile: float,
    discount_factor: float = 1.0,
):
    discounts = np.array([discount_factor ** len(e.steps) for e in batch])
    rewards = np.array([e.reward for e in batch])
    discounted_rewards = rewards * discounts
    reward_bound = np.percentile(discounted_rewards, percentile).item()
    reward_mean = np.mean(rewards).item()

    train_obs: list[ObsType] = []
    train_act: list[ActType] = []
    elite_batch: list[Episode[ObsType, ActType]] = []
    for disc_reward, example in zip(discounted_rewards, batch):
        if disc_reward > reward_bound:
            elite_batch.append(example)
            for step in example.steps:
                train_obs.append(step.observation)
                train_act.append(step.action)

    train_obs_t = torch.from_numpy(np.array(train_obs, dtype=np.float32))  # type: ignore
    train_act_t = torch.from_numpy(np.array(train_act, dtype=np.int64))  # type: ignore
    return elite_batch, train_obs_t, train_act_t, reward_bound, reward_mean


def train_crossentropy_elite_batches(
    policy: Net,
    optimizer: optim.Optimizer,
    env: gym.Env[ObsType, ActType],
    stopping_criterion: Callable[[int, float], bool],
    run_name: str,
    batch_size: int = 128,
    max_batch_size: int = 512,
    percentile: float = 30,
    episode_discount_factor: float = 0.9,
    log_level: str = "INFO",
    disable_logging: bool = False,
    tb_root: Path = Path("runs"),
    video_log_root: Path = Path("videos"),
):
    setup_logger(log_level, disable_logging)
    writer = SummaryWriter(log_dir=f"{tb_root}/{run_name}")

    pbar = tqdm.tqdm()
    objective = nn.CrossEntropyLoss()
    full_batch: list[Episode[ObsType, ActType]] = []
    for iter_no, batch in enumerate(iterate_batches(env, policy, batch_size)):
        full_batch, obs_t, act_t, reward_bound, reward_mean = filter_batch_elite(
            full_batch + batch, percentile, episode_discount_factor
        )
        if not full_batch:
            continue

        full_batch = full_batch[-max_batch_size:]

        optimizer.zero_grad()
        act_logits = policy(obs_t)
        loss_v = objective(act_logits, act_t)
        loss_v.backward()
        optimizer.step()  # type: ignore

        pbar.set_description(f"it {iter_no} | bsz {len(full_batch)}")
        pbar.update(1)
        pbar.set_postfix(  # type: ignore
            loss=f"{loss_v.item():.3f}",
            rw_mean=f"{reward_mean:.1f}",
            rw_bound=f"{reward_bound:.1f}",
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)  # type: ignore
        writer.add_scalar("reward_bound", reward_bound, iter_no)  # type: ignore
        writer.add_scalar("reward_mean", reward_mean, iter_no)  # type: ignore

        if stopping_criterion(iter_no, reward_mean):
            break

    writer.close()
    if video_log_root:
        save_videos(env, policy, f"{video_log_root}/{run_name}")
