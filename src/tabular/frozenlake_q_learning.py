import collections
import tqdm
import copy
import itertools
from pathlib import Path
from typing import Iterable, TypeAlias

import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatdim
from numpy.typing import NDArray
from torch.utils.tensorboard.writer import SummaryWriter
import typer
from loguru import logger

from utils import ensure_dir_exists, save_videos, setup_logger  # type: ignore


ObsType: TypeAlias = np.int64
ActType: TypeAlias = np.int64


class Agent:
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        gamma: float,
        alpha: float,
        eps: float = 0.9,
    ):
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.state, _ = self.env.reset()
        self.action = self.env.action_space.sample()  # for sarsa
        self.values: dict[ObsType, NDArray[np.float32]] = collections.defaultdict(
            lambda: np.zeros(len(list(self.get_available_actions())), dtype=np.float32)
        )

    def get_available_actions(self) -> Iterable[ActType]:
        return np.arange(flatdim(self.env.action_space))

    def get_available_states(self) -> Iterable[ObsType]:
        return np.arange(flatdim(self.env.observation_space))

    def select_action(self, state: ObsType) -> ActType:
        return np.argmax(self.values[state])

    def play_episode(self) -> float:
        total_reward = 0.0
        env = self.test_env
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
            state = new_state
        return total_reward

    def policy(self, state: ObsType) -> NDArray[np.float32]:
        return self.values[state]

    def sample_action_eps_greedy(self, state: ObsType) -> ActType:
        if np.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            return self.select_action(state)

    def sample_env(self) -> tuple[ObsType, ActType, float, ObsType]:
        old_state = self.state
        action = self.sample_action_eps_greedy(old_state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.state, _ = self.env.reset()
        else:
            self.state = next_state
        return old_state, action, float(reward), next_state

    def value_update(
        self, state: ObsType, action: ActType, reward: float, next_state: ObsType
    ) -> None:
        td_err = (
            reward
            + self.gamma * np.max(self.values[next_state])
            - self.values[state][action]
        )
        self.values[state][action] += self.alpha * td_err

    def q_learn(self):
        s, a, r, next_s = self.sample_env()
        self.value_update(s, a, r, next_s)

    def sarsa(self):
        state = self.state
        action = self.sample_action_eps_greedy(state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.state, _ = self.env.reset()
            self.action = self.env.action_space.sample()
            next_value = 0
        else:
            self.state = next_state
            self.action = self.sample_action_eps_greedy(next_state)
            next_value = self.values[next_state][self.action]
        td_err = float(reward) + self.gamma * next_value - self.values[state][action]
        self.values[state][action] += self.alpha * td_err


def main(
    env_name: str = "FrozenLake-v1",
    gamma: float = 0.9,
    alpha: float = 0.2,
    eps: float = 0.9,  # small values of eps inhibit training
    use_sarsa: bool = False,
    target_reward: float = 0.9,
    num_test_episodes: int = 20,
    log_level: str = "INFO",
    disable_logging: bool = False,
    tb_root: Path = typer.Option(
        default="runs/q_learning", file_okay=False, callback=ensure_dir_exists
    ),
    video_log_root: Path = typer.Option(
        default="videos/q_learning", file_okay=False, callback=ensure_dir_exists
    ),
):
    setup_logger(log_level, disable_logging)
    env: gym.Env[ObsType, ActType] = gym.make(  # type: ignore
        env_name,
        render_mode="rgb_array" if video_log_root else None,
    )

    agent = Agent(env, gamma, alpha, eps)
    run = f"gamma={gamma},alpha={alpha},eps={eps},target={target_reward}"
    writer = SummaryWriter(log_dir=f"{tb_root}/{run}")

    best_reward = 0.0
    pbar = tqdm.tqdm()
    for iter_no in itertools.count(1):
        if use_sarsa:
            agent.sarsa()
        else:
            agent.q_learn()

        reward = 0.0
        for _ in range(num_test_episodes):
            reward += agent.play_episode()
        reward /= num_test_episodes
        writer.add_scalar("reward", reward, iter_no)  # type: ignore
        pbar.set_description(f"it {iter_no}")
        pbar.update(1)
        pbar.set_postfix(  # type: ignore
            reward=f"{reward:.1f}",
            best_reward=f"{best_reward:.1f}",
        )
        if reward > best_reward:
            best_reward = reward
        if reward > target_reward:
            logger.info("Solved in {} iterations!", iter_no)
            break

    writer.close()
    if video_log_root:
        save_videos(env, agent.policy, f"{video_log_root}/{run}")


if __name__ == "__main__":
    typer.run(main)
