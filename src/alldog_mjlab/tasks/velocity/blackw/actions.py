"""Mixed wheel-leg action term for blackW.

One action term per leg, four raw actions per leg in the policy order
hip -> thigh -> calf -> wheel. The first three are position residuals around the
default joint pose (scale 0.25); the wheel action is a velocity target (scale 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg


@dataclass(kw_only=True)
class WheelLegActionCfg(BaseActionCfg):
    """One leg: 3 position residuals + 1 wheel velocity target."""

    def __post_init__(self) -> None:
        self.transmission_type = TransmissionType.JOINT

    def build(self, env) -> "WheelLegAction":
        return WheelLegAction(self, env)


class WheelLegAction(BaseAction):
    cfg: WheelLegActionCfg

    def __init__(self, cfg: WheelLegActionCfg, env) -> None:
        super().__init__(cfg=cfg, env=env)
        assert self.action_dim == 4, self.action_dim
        default_pos = self._entity.data.default_joint_pos
        assert default_pos is not None
        self._default_pos = default_pos[:, self._target_ids[:3]].clone()

    def apply_actions(self) -> None:
        # HIM exploration can occasionally produce non-finite policy actions on
        # contact-heavy platform terrain; sanitize before sending them to MuJoCo.
        actions = torch.nan_to_num(
            self._processed_actions, nan=0.0, posinf=100.0, neginf=-100.0
        )
        self._processed_actions = actions
        leg_target = actions[:, :3] + self._default_pos
        wheel_target = actions[:, 3:]
        self._entity.set_joint_position_target(leg_target, joint_ids=self._target_ids[:3])
        self._entity.set_joint_velocity_target(wheel_target, joint_ids=self._target_ids[3:])
