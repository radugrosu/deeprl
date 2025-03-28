from pathlib import Path
import sys
from typing import Any, Callable, TypeAlias
from loguru import logger
from jaxtyping import Float32
from numpy.typing import NDArray
import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

ObsType: TypeAlias = Float32[NDArray[np.float32], "4"]
ActType: TypeAlias = np.int64


def setup_logger(
    log_level: str,
    disable_logging: bool,
    file_sink: str | None = None,
    ensure_exists: bool = True,
):
    handlers: list[dict[str, Any]] = [
        {"sink": sys.stdout, "format": "[{time}] {message}"},
    ]
    if file_sink is not None:
        if not ensure_exists:
            parent = Path(file_sink).parent
            assert parent.exists(), "Parent directory does not exist"
        handlers += [{"sink": file_sink, "serialize": True}]
    logger.configure(
        handlers=handlers,  # type: ignore
        levels=[{"name": log_level}],
    )
    if disable_logging:
        logger.disable(__name__)


def ensure_exists(*args: Path | str) -> Path:
    path = Path(*args)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir_exists(dir_path: Path | None) -> Path | None:
    if dir_path is not None:
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


def obs2tensor(obs: ObsType) -> Float32[torch.FloatTensor, "1 obs_size"]:
    return torch.from_numpy(  # type: ignore
        obs,
    ).unsqueeze(0)


def tensor2probs(
    act_probs_t: Float32[torch.FloatTensor, "1 n_actions"],
) -> NDArray[np.int64]:
    return act_probs_t.detach().numpy().squeeze()  # type: ignore


def save_videos[ObsType](
    env: gym.Env[ObsType, ActType],
    policy: Callable[[ObsType], NDArray[np.float32]],
    video_folder: str,
) -> None:
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,
    )  # Record all episodes
    obs, _ = env.reset()
    done = False
    while not done:
        act_preferences = policy(obs)
        action = np.argmax(act_preferences)
        obs, _, done, _, _ = env.step(action)
    env.close()
