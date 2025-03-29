import time
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple

import ale_py  # type: ignore # necessary for loading atari  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tqdm
import typer
from loguru import logger
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary  # type: ignore

from dqn.net import DQN  # type: ignore
from dqn.wrappers import Bool, Float32, Int64, make_env  # type: ignore
from utils import ActType, ensure_exists, setup_logger  # type: ignore


class Batch(NamedTuple):
    state: torch.FloatTensor
    action: torch.LongTensor
    reward: torch.FloatTensor
    done: torch.BoolTensor
    next_state: torch.FloatTensor


class NpBatch(NamedTuple):
    state: Float32
    action: Int64
    reward: Float32
    done: Bool
    next_state: Float32

    def to(self, device: torch.device) -> Batch:
        return Batch(
            torch.tensor(self.state, dtype=torch.float32, device=device),  # type: ignore
            torch.tensor(self.action, device=device),  # type: ignore
            torch.tensor(self.reward, device=device),  # type: ignore
            torch.tensor(self.done, device=device),  # type: ignore
            torch.tensor(self.next_state, device=device),  # type: ignore
        )


class Experience(NamedTuple):
    state: Float32
    action: ActType
    reward: float
    done: bool
    next_state: Float32


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> NpBatch:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices)
        )
        return NpBatch(
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones),
            np.array(next_states),
        )


class Simulator:
    def __init__(
        self,
        env: gym.Env[Float32, ActType],
        exp_buffer: ExperienceBuffer,
        device: torch.device,
    ):
        self.env = env
        self.exp_buffer = exp_buffer
        self.device = device
        self.total_reward = 0.0
        self.state, _ = self.env.reset()

    def play_step(self, net: DQN, epsilon: float = 0.0) -> float | None:
        total_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_t = torch.tensor(self.state, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_vals_t = net(state_t)
            action: ActType = np.argmax(q_vals_t.squeeze(0).detach().cpu().numpy())

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        reward = float(reward)
        self.total_reward += reward
        done = terminated or truncated
        exp = Experience(self.state, action, reward, done, next_state)
        self.exp_buffer.append(exp)
        self.state = next_state
        if done:
            total_reward = self.total_reward
            self.total_reward = 0.0
            self.state, _ = self.env.reset()
        return total_reward


def write_scalar(
    writer: SummaryWriter, tag: str, scalar_value: Any, global_step: int | None = None
) -> None:
    writer.add_scalar(tag, scalar_value, global_step)  # type: ignore


def calc_loss(
    batch: NpBatch,
    dqn: DQN,
    target_policy: DQN,
    gamma: float,
    device: torch.device,
):
    states, actions, rewards, dones, next_states = batch
    states_t = torch.tensor(states, device=device)
    next_states_t = torch.tensor(next_states, device=device)
    actions_t = torch.tensor(actions, device=device)
    rewards_t = torch.tensor(rewards, device=device)
    done_mask = torch.tensor(dones, device=device)

    state_action_values = dqn(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = target_policy(next_states_t).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_t
    return F.mse_loss(state_action_values, expected_state_action_values)


def main(
    root: Path,
    env_name: str = "PongNoFrameskip-v4",
    cuda: bool = True,
    mean_reward_bound: int = 19,
    gamma: float = 0.99,
    batch_size: int = 64,
    replay_size: int = 10000,
    learning_rate: float = 5e-4,
    sync_target_frames: int = 5000,
    eps_decay_last_frame: int = 500000,
    eps_start: float = 1.0,
    eps_final: float = 0.01,
    mean_reward_window: int = 100,
    debug_log_interval: int = 100,
    ckp_root: str = "ckp",
    tb_root: str = "tb",
    log_root: str = "logs",
    log_level: str = "INFO",
    disable_logging: bool = False,
):
    root = ensure_exists(root, "dqn-pong")
    run = (
        f"gamma={gamma},bs={batch_size},lr={learning_rate},sf={sync_target_frames},"
        f"rs={replay_size},epsf={eps_final},rb={mean_reward_bound}"
    )
    setup_logger(
        log_level,
        disable_logging,
        Path(ensure_exists(root, log_root), f"{run}.log").as_posix(),
    )
    writer = SummaryWriter(log_dir=ensure_exists(root, tb_root, run).as_posix())

    device = torch.device("cuda" if cuda else "cpu")
    env = make_env(env_name)
    input_shape: tuple[int, int, int] = env.observation_space.shape  # type: ignore
    n_actions = int(env.action_space.n)  # type: ignore

    net = DQN(input_shape, n_actions).to(device)
    tgt_net = DQN(input_shape, n_actions).to(device)
    summary(
        net,
        input_size=(1, *input_shape),
        col_names=("input_size", "output_size", "num_params"),
    )

    buffer = ExperienceBuffer(replay_size)
    simulator = Simulator(env, buffer, device)
    epsilon = eps_start

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    total_rewards: list[float] = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    pbar = tqdm.tqdm()
    while True:
        frame_idx += 1
        epsilon = max(eps_final, eps_start - frame_idx / eps_decay_last_frame)
        reward = simulator.play_step(net, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-mean_reward_window:])
            pbar.set_description(f"it {frame_idx}")
            pbar.update(1)
            pbar.set_postfix(  # type: ignore
                done=f"{len(total_rewards)}",
                reward=f"{reward:.1f}",
                best_reward=f"{best_m_reward:.1f}" if best_m_reward is not None else "",
                eps=f"{epsilon:.2f}",
                speed=f"{speed:.2f}f/s",
            )
            write_scalar(writer, "epsilon", epsilon, frame_idx)
            write_scalar(writer, "speed", speed, frame_idx)
            write_scalar(writer, "reward_mean", m_reward, frame_idx)
            write_scalar(writer, "reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(  # type: ignore
                    net.state_dict(),
                    Path(
                        ensure_exists(root, ckp_root), f"best_{m_reward:.0f}.pt"
                    ).as_posix(),
                )
                if best_m_reward is not None:
                    logger.info(
                        "Best reward updated {:.3f} -> {:.3f}", best_m_reward, m_reward
                    )
                best_m_reward = m_reward
            if m_reward > mean_reward_bound:
                logger.info("Solved in {} frames!", frame_idx)
                break

        if len(buffer) < replay_size:
            continue

        if frame_idx % sync_target_frames == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_t = calc_loss(batch, net, tgt_net, gamma, device)
        loss_t.backward()  # type: ignore
        optimizer.step()  # type: ignore

        if frame_idx % debug_log_interval == 0:
            with torch.no_grad():
                # Get Q-values for the batch states
                q_values = net(torch.tensor(batch.state, device=device))
                # Get target Q-values (before masking dones)
                next_q_values = tgt_net(
                    torch.tensor(batch.next_state, device=device)
                ).max(1)[0]
                target_q_values = (
                    torch.tensor(batch.reward, device=device) + gamma * next_q_values
                )

            write_scalar(writer, "loss", loss_t.item(), frame_idx)
            write_scalar(writer, "q_values_mean", q_values.mean().item(), frame_idx)
            write_scalar(
                writer, "target_q_values_mean", target_q_values.mean().item(), frame_idx
            )

    writer.close()


if __name__ == "__main__":
    typer.run(main)
