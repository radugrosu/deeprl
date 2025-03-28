import copy
from numpy.typing import NDArray
from loguru import logger
import itertools
from pathlib import Path
from gymnasium.spaces.utils import flatdim
from typing import Iterable, TypeAlias
import numpy as np
import gymnasium as gym
import collections
from torch.utils.tensorboard.writer import SummaryWriter
import typer

from utils import ensure_dir_exists, setup_logger, save_videos  # type: ignore


ObsType: TypeAlias = np.int64
ActType: TypeAlias = np.int64


class Agent:
    def __init__(self, env: gym.Env[ObsType, ActType], gamma: float):
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.gamma = gamma
        self.state, _ = self.env.reset()
        self.rewards: dict[tuple[ObsType, ActType, ObsType], float] = (
            collections.defaultdict(float)
        )
        self.transits: dict[tuple[ObsType, ActType], dict[ObsType, int]] = (
            collections.defaultdict(collections.Counter)
        )
        self.values: dict[ObsType, NDArray[np.float32]] = collections.defaultdict(
            lambda: np.zeros(len(list(self.get_available_actions())), dtype=np.float32)
        )

    def play_n_random_steps(self, count: int):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            self.rewards[self.state, action, new_state] = float(reward)
            self.transits[(self.state, action)][new_state] += 1
            if terminated or truncated:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

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
            self.rewards[(state, action, new_state)] = float(reward)
            self.transits[(state, action)][new_state] += 1
            total_reward += float(reward)
            if terminated or truncated:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in self.get_available_states():
            action_values: list[float] = []
            for action in self.get_available_actions():
                transitions = self.transits[state, action]
                total = sum(transitions.values())
                action_value = 0.0
                for next_state, count in transitions.items():
                    reward = self.rewards[state, action, next_state]
                    val = reward + self.gamma * np.max(self.values[next_state]).item()
                    action_value += (count / total) * val
                action_values.append(action_value)
            self.values[state] = np.array(action_values, dtype=np.float32)

    def policy(self, state: ObsType) -> NDArray[np.float32]:
        return self.values[state]


def main(
    env_name: str = "FrozenLake-v1",
    gamma: float = 0.9,
    target_reward: float = 0.9,
    num_test_episodes: int = 20,
    log_level: str = "INFO",
    disable_logging: bool = False,
    tb_root: Path = typer.Option(
        default="runs", file_okay=False, callback=ensure_dir_exists
    ),
    video_log_root: Path = typer.Option(
        default="videos/q_iteration", file_okay=False, callback=ensure_dir_exists
    ),
):
    setup_logger(log_level, disable_logging)
    env: gym.Env[ObsType, ActType] = gym.make(  # type: ignore
        env_name,
        render_mode="rgb_array" if video_log_root else None,
    )
    agent = Agent(env, gamma)
    run = f"gamma={gamma},target={target_reward}"
    writer = SummaryWriter(log_dir=f"{tb_root}/{run}")
    best_reward = 0.0
    for iter_no in itertools.count(start=1):
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(num_test_episodes):
            reward += agent.play_episode()
        reward /= num_test_episodes
        writer.add_scalar("reward", reward, iter_no)  # type: ignore
        if reward > best_reward:
            logger.info("Best reward updated {:.3f} -> {:.3f}", best_reward, reward)
            best_reward = reward
        if reward > target_reward:
            logger.info("Solved in {} iterations!", iter_no)
            break

    writer.close()
    if video_log_root:
        save_videos(env, agent.policy, f"{video_log_root}/{run}")


if __name__ == "__main__":
    typer.run(main)
