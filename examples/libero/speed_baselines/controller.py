from __future__ import annotations

import numpy as np

_PATCHED = False


def apply_sup_controller_patches() -> None:
    """Apply the three LIBERO runtime patches from the supplied SuP code."""

    global _PATCHED
    if _PATCHED:
        return

    import robosuite.controllers.base_controller as base_controller_module
    import robosuite.environments.base as base_module
    import robosuite.models.grippers.panda_gripper as panda_gripper_module

    def scale_action_without_clip(self, action):
        if self.action_scale is None:
            self.action_scale = np.abs(self.output_max - self.output_min) / np.abs(
                self.input_max - self.input_min
            )
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
        return (action - self.action_input_transform) * self.action_scale + self.action_output_transform

    def mujoco_step(self, action):
        self.timestep += 1
        policy_step = True
        for _ in range(int(self.control_timestep / self.model_timestep)):
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            self._update_observables()
            policy_step = False
        self.cur_time += self.control_timestep
        reward, done, info = self._post_action(action)
        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()
        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations, reward, done, info

    def panda_speed(_self):
        return 0.02

    base_controller_module.Controller.scale_action = scale_action_without_clip
    panda_gripper_module.PandaGripper.speed = property(panda_speed)
    base_module.MujocoEnv.step = mujoco_step
    _PATCHED = True
