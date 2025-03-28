import collections
import time
from pathlib import Path

import ale_py  # type: ignore # necessary for loading atari  # noqa: F401
import numpy as np
import torch
import typer
from gymnasium.wrappers import RecordVideo

from dqn.net import DQN  # type: ignore
from dqn.wrappers import make_env  # type: ignore
from utils import ActType, ensure_exists  # type: ignore


def main(
    ckp: str,
    env_name: str = "PongNoFrameskip-v4",
    vis: bool = False,
    fps: int = 25,
    video_root: str | None = None,
):
    env = make_env(
        env_name, render_mode="rgb_array" if video_root is not None else None
    )
    if video_root is not None:
        env = RecordVideo(
            env,
            video_folder=ensure_exists(video_root, env_name, Path(ckp).stem).as_posix(),
            episode_trigger=lambda e: True,
        )  # Record all episodes
    input_shape: tuple[int, int, int] = env.observation_space.shape  # type: ignore
    n_actions = int(env.action_space.n)  # type: ignore
    net = DQN(input_shape, n_actions)
    state_dict = torch.load(ckp, map_location=lambda stg, _: stg)  # type: ignore
    net.load_state_dict(state_dict)

    state, _ = env.reset()
    total_reward = 0.0
    action_counter: dict[ActType, int] = collections.Counter()
    while True:
        start_ts = time.time()
        if vis:
            env.render()
        state_t = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals_t = net(state_t)
        act_t = np.argmax(q_vals_t.squeeze(0).detach().numpy())
        action = np.int64(act_t.item())
        action_counter[action] += 1
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
        if vis:
            delta = 1 / fps - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print(f"Total reward: {total_reward:.1f}")
    if video_root is not None:
        env.close()


if __name__ == "__main__":
    typer.run(main)
