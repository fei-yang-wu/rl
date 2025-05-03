# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import itertools
import warnings
from typing import Any
import gymnasium as gym

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import _classproperty, make_composite_from_td

_has_isaac = importlib.util.find_spec("isaaclab") is not None


class IsaacLabWrapper(GymWrapper):
    """Wrapper for IsaacLabEnvs environments.

    The original library can be found `here <https://isaac-sim.github.io/IsaacLab/main/index.html>`_.

    .. note:: IsaacLab environments cannot be executed consecutively, ie. instantiating one
        environment after another (even if it has been cleared) will cause
        CUDA memory issues. We recommend creating one environment per process only.
        If you need more than one environment, the best way to achieve that is
        to spawn them across processes.

    .. note:: IsaacLab works on CUDA devices by essence. Make sure your machine
        has GPUs available and the required setup for IsaacLab.

    """

    @property
    def lib(self):
        import isaaclab

        return isaaclab

    def __init__(
        self,
        env: gym.Env,
        **kwargs,  # noqa: F821
    ):
        warnings.warn(
            "IsaacLab environment support is an experimental feature that's being maintained by Georgia Tech LIDAR."
        )
        super().__init__(
            env, torch.device(env.device), batch_size=torch.Size([]), **kwargs
        )
        if not hasattr(self, "task"):
            # by convention in IsaacLabEnvs
            self.task = env.__name__

    @property
    def _is_batched(self):
        return True

    def _make_specs(self, env: gym.Env) -> None:  # noqa: F821

        super()._make_specs(env, batch_size=self.batch_size)
        self.full_done_spec = Composite(
            {
                key: spec.squeeze(-1)
                for key, spec in self.full_done_spec.items(True, True)
            },
            shape=self.batch_size,
        )

        self.observation_spec["obs"] = self.observation_spec["observation"]
        del self.observation_spec["observation"]
        print("observation_spec", self.observation_spec)
        print("full_done_spec", self.full_done_spec)

        data = self.rollout(3).get("next")[..., 0]
        del data[self.reward_key]
        for done_key in self.done_keys:
            try:
                del data[done_key]
            except KeyError:
                continue
        specs = make_composite_from_td(data)

        obs_spec = self.observation_spec
        obs_spec.unlock_(recurse=True)
        obs_spec.update(specs)
        obs_spec.lock_(recurse=True)

    def _output_transform(self, output):
        obs, reward, terminated, truncated, _ = output
        if self.from_pixels:
            obs["pixels"] = self._env.render(mode="rgb_array")
        return (
            obs,
            reward.unsqueeze(-1),
            terminated,
            truncated,
            terminated | truncated,
            {},
        )

    def _reset_output_transform(self, reset_data):
        reset_data, _ = reset_data
        if self.from_pixels:
            reset_data["pixels"] = self._env.render(mode="rgb_array")
        return reset_data, {}

    def _set_seed(self, seed: int | None) -> None:
        # as of #665c32170d84b4be66722eea405a1e08b6e7f761 the seed points nowhere in gym.make for IsaacGymEnvs
        ...

    def read_action(self, action):
        """Reads the action obtained from the input TensorDict and transforms it in the format expected by the contained environment.

        Args:
            action (Tensor or TensorDict): an action to be taken in the environment

        Returns: an action in a format compatible with the contained environment.

        """
        return action

    def read_done(
        self,
        terminated: bool = None,
        truncated: bool | None = None,
        done: bool | None = None,
    ) -> tuple[bool, bool, bool]:
        if terminated is not None:
            terminated = terminated.bool()
        if truncated is not None:
            truncated = truncated.bool()
        if done is not None:
            done = done.bool()
        return terminated, truncated, done, done.any()

    def read_reward(self, total_reward):
        return total_reward

    def read_obs(
        self, observations: dict[str, Any] | torch.Tensor | np.ndarray
    ) -> dict[str, Any]:
        """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

        Args:
            observations (observation under a format dictated by the inner env): observation to be read.

        """
        if isinstance(observations, dict):
            if "state" in observations and "observation" not in observations:
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                observations["observation"] = observations.pop("state")
        if not isinstance(observations, (TensorDictBase, dict)):
            (key,) = itertools.islice(self.observation_spec.keys(True, True), 1)
            observations = {key: observations}
        return observations
