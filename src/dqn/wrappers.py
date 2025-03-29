from typing import Any, SupportsFloat, TypeAlias, cast
import cv2
import gymnasium as gym
import numpy as np
from collections import deque
from numpy.typing import NDArray

Bool = NDArray[np.bool]
Float32: TypeAlias = NDArray[np.float32]
Int64: TypeAlias = NDArray[np.int64]
UInt8: TypeAlias = NDArray[np.uint8]
ActType: TypeAlias = np.int64


class FireResetEnv[ObsType: NDArray[np.uint8]](
    gym.Wrapper[ObsType, ActType, ObsType, ActType]
):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        """For environments where the user need to press FIRE for the game to start."""
        super().__init__(env)
        assert hasattr(env.unwrapped, "get_action_meanings")
        action_meanings: list[str] = cast(
            list[str],
            env.unwrapped.get_action_meanings(),  # type: ignore
        )
        assert len(action_meanings) >= 3
        assert action_meanings[1] == "FIRE"

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset()
        for _ in range(60):
            obs, *_, info = self.env.step(
                np.int64(0)
            )  # no-op, but doesn't seem to start any other way
        return obs, info


class MaxAndSkipEnv[ObsType: NDArray[np.uint8] | NDArray[np.uint8]](
    gym.Wrapper[ObsType, ActType, ObsType, ActType]
):
    def __init__(self, env: gym.Env[ObsType, ActType], skip: int = 4):
        """Return only every `skip`-th frame, the pointwise max of the last 2 frames
        Note: The pooling is meant to address occasional flickering in the game.
        """
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer: deque[ObsType] = deque(maxlen=2)
        self._skip = skip

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        info: dict[str, Any] = {}
        terminated, truncated = False, False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, info = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, info


def process_frame84(frame: NDArray[Any]) -> NDArray[np.uint8]:
    if frame.size == 210 * 160 * 3:
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    elif frame.size == 250 * 160 * 3:
        img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
    else:
        assert False, "Unknown resolution."
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)


class ProcessFrame84(gym.ObservationWrapper[UInt8, ActType, UInt8]):
    def __init__(self, env: gym.Env[UInt8, ActType]):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, observation: UInt8) -> UInt8:
        return process_frame84(observation)


class ImageToPyTorch(gym.ObservationWrapper[UInt8, ActType, UInt8]):
    def __init__(self, env: gym.Env[UInt8, ActType]):
        super().__init__(env)
        old_shape: tuple[int, int, int] = self.observation_space.shape  # type: ignore
        new_shape = old_shape[-1], old_shape[0], old_shape[1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation: UInt8) -> UInt8:
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper[UInt8, ActType, UInt8]):
    def __init__(self, env: gym.Env[UInt8, ActType], n_steps: int):
        super().__init__(env)
        orig_space: gym.spaces.Box = env.observation_space  # type: ignore
        low: UInt8 = orig_space.low  # type: ignore
        high: UInt8 = orig_space.high  # type: ignore
        self.observation_space = gym.spaces.Box(
            low.repeat(n_steps, axis=0),
            high.repeat(n_steps, axis=0),
            dtype=np.uint8,
        )
        self.buffer: UInt8 = np.zeros_like(
            self.observation_space.low,  # type: ignore
            np.uint8,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[UInt8, dict[str, Any]]:
        self.buffer = np.zeros_like(
            self.observation_space.low,  # type: ignore
            dtype=np.uint8,
        )
        obs, info = self.env.reset()
        return self.observation(obs), info

    def observation(self, observation: UInt8) -> UInt8:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ScaledFloatFrame(gym.ObservationWrapper[Float32, ActType, UInt8]):
    def __init__(self, env: gym.Env[UInt8, ActType]):
        super().__init__(env)

    def observation(self, observation: UInt8) -> Float32:
        return np.asarray(observation).astype(np.float32) / 255.0


def make_env(
    env_name: str, render_mode: str | None = None, skip_frames: int = 4
) -> ScaledFloatFrame:
    env: gym.Env[UInt8, ActType] = gym.make(env_name, render_mode=render_mode)  # type: ignore
    env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip_frames)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
