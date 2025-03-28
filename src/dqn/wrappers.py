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


class FireResetEnv(gym.Wrapper[Float32, ActType, Float32, ActType]):
    def __init__(self, env: gym.Env[Float32, ActType]):
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
    ) -> tuple[Float32, SupportsFloat, bool, bool, dict[str, bool]]:
        return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Float32, dict[str, Any]]:
        self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(np.int64(1))
        if terminated or truncated:
            self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(np.int64(2))
        if terminated or truncated:
            self.env.reset()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper[Float32, ActType, Float32, ActType]):
    def __init__(self, env: gym.Env[Float32, ActType], skip: int = 4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer: deque[Float32] = deque(maxlen=2)
        self._skip = skip

    def step(
        self, action: ActType
    ) -> tuple[Float32, SupportsFloat, bool, bool, dict[str, bool]]:
        total_reward = 0.0
        terminated, truncated, info = False, False, {}
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
    ) -> tuple[Float32, dict[str, Any]]:
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, info = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, info


def process_frame84(frame: Float32) -> Float32:
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
    return x_t.astype(np.float32)


class ProcessFrame84(gym.ObservationWrapper[Float32, ActType, Float32]):
    def __init__(self, env: gym.Env[Float32, ActType]):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, observation: Float32) -> Float32:
        return process_frame84(observation)


class ImageToPyTorch(gym.ObservationWrapper[Float32, ActType, Float32]):
    def __init__(self, env: gym.Env[Float32, ActType]):
        super().__init__(env)
        old_shape: tuple[int, int, int] = self.observation_space.shape  # type: ignore
        new_shape = old_shape[-1], old_shape[0], old_shape[1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation: Float32) -> Float32:
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper[Float32, ActType, Float32]):
    def observation(self, observation: Float32) -> Float32:
        return np.asarray(observation).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper[Float32, ActType, Float32]):
    def __init__(self, env: gym.Env[Float32, ActType], n_steps: int):
        super().__init__(env)
        orig_space: gym.spaces.Box = env.observation_space  # type: ignore
        low: Float32 = orig_space.low  # type: ignore
        high: Float32 = orig_space.high  # type: ignore
        self.observation_space = gym.spaces.Box(
            low.repeat(n_steps, axis=0),
            high.repeat(n_steps, axis=0),
            dtype=np.float32,
        )
        self.buffer: Float32 = np.zeros_like(
            self.observation_space.low,  # type: ignore
            np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Float32, dict[str, Any]]:
        self.buffer = np.zeros_like(
            self.observation_space.low,  # type: ignore
            dtype=np.float32,
        )
        obs, info = self.env.reset()
        return self.observation(obs), info

    def observation(self, observation: Float32) -> Float32:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name: str, render_mode: str | None = None) -> ScaledFloatFrame:
    env: gym.Env[Float32, ActType] = gym.make(env_name, render_mode=render_mode)  # type: ignore
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env
