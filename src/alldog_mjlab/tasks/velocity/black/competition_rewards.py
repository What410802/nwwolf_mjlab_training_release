"""Reward / termination terms for the competition-proxy obstacle courses.

The terrain generator marks a far-end goal flat patch (``flat_patches["goal"]``)
for the obstacle course sub-terrains.  Non-course sub-terrains get the spawn
origin as the fill value, so a goal is only treated as valid when it is at
least ``MIN_GOAL_DISTANCE_M`` ahead of the env origin along +x.  The reward is
sparse (1.0 on the step the robot root crosses the goal line) and a matching
termination ends the episode, giving PPO a task-success signal on top of the
velocity-tracking reward.
"""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
MIN_GOAL_DISTANCE_M = 0.8


def _goal_positions(env, goal_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    terrain = env.scene.terrain
    if terrain is None or terrain.flat_patches is None:
        return None
    patches = terrain.flat_patches.get(goal_name)
    if patches is None:
        return None
    rows = terrain.terrain_levels
    cols = terrain.terrain_types
    goal = patches[rows, cols, 0]  # (num_envs, 3)
    origins = terrain.env_origins
    valid = (goal[:, 0] - origins[:, 0]) > MIN_GOAL_DISTANCE_M
    return goal, valid


def competition_goal_reward(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sparse task-success reward: 1.0 on the step the root crosses the goal line."""
    asset = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    goal_data = _goal_positions(env, goal_name)
    if goal_data is None:
        return reward
    goal, valid = goal_data
    root_x = asset.data.root_link_pos_w[:, 0]
    reward[(root_x > goal[:, 0]) & valid] = 1.0
    return reward


def competition_goal_reached(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Termination: end the episode once the root crosses the goal line."""
    return competition_goal_reward(env, goal_name=goal_name, asset_cfg=asset_cfg) > 0.5


def competition_goal_progress(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense progress reward along +x towards the far-end goal (course columns only).

    ``progress = clip((x - origin_x - 0.5) / (goal_x - origin_x), 0, 1)`` so standing
    at spawn gives 0 and crossing the goal line gives 1.  Non-course columns get a
    goal patch filled with the spawn origin and are masked out.
    """
    asset = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    goal_data = _goal_positions(env, goal_name)
    if goal_data is None:
        return reward
    goal, valid = goal_data
    origins = env.scene.terrain.env_origins
    span = (goal[:, 0] - origins[:, 0]).clamp(min=1e-3)
    progress = ((asset.data.root_link_pos_w[:, 0] - origins[:, 0] - 0.5) / span).clamp(0.0, 1.0)
    return torch.where(valid, progress, torch.zeros_like(progress))
