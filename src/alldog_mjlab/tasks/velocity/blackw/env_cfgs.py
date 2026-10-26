"""blackW wheel-legged task assembly (first milestone: flat velocity tracking).

This module starts from the validated Black flat task and replaces exactly the
robot / wheels' action interface / observation contract / reset events. The
reward, command and termination wiring stay on the Black baseline so the first
blackW policy can be trained and evaluated end-to-end before adding competition
terrain.
"""

from __future__ import annotations

import os

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from alldog_mjlab.robots.blackw import (
    BLACKW_FOOT_NAMES,
    BLACKW_JOINT_NAMES,
    BLACKW_WHEEL_JOINTS,
    get_blackw_robot_cfg,
)
from alldog_mjlab.tasks.velocity.black.env_cfgs import (
    _build_black_env_cfg,
    black_competition_env_cfg,
    competition_goal_progress,
    competition_goal_reward,
)
from alldog_mjlab.tasks.velocity.blackw.actions import WheelLegActionCfg
from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg

from alldog_mjlab.tasks.velocity.blackw.exact_terrain import (
    ExactHighPlatformTerrainCfg,
    ExactHighWallTerrainCfg,
    ExactSlopeTerrainCfg,
    ExactStairsTerrainCfg,
    ExactTrenchTerrainCfg,
)

BLACKW_ACTION_TERM_ORDER = tuple(f"wheel_leg_{leg.lower()}" for leg in BLACKW_FOOT_NAMES)
BLACKW_ACTOR_OBS_TERM_ORDER = (
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)
BLACKW_ACTOR_HISTORY_LENGTH = 6
BLACKW_ACTOR_OBS_SCALE = {
    "command": (2.0, 2.0, 0.25),
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,
    "actions": 1.0,
}


def _wheel_leg_scale(leg: str) -> dict[str, float]:
    return {
        f"{leg}_hip_joint": 0.25,
        f"{leg}_thigh_joint": 0.25,
        f"{leg}_calf_joint": 0.25,
        f"{leg}_wheel_joint": 10.0,
    }


def _blackw_joint_pos_rel(env, asset_cfg: SceneEntityCfg, biased: bool = False):
    """Joint position error, with wheel positions forced to zero (official contract)."""
    obs = envs_mdp.joint_pos_rel(env, asset_cfg=asset_cfg, biased=biased)
    names = asset_cfg.joint_names
    wheel_cols = [i for i, name in enumerate(names) if name in BLACKW_WHEEL_JOINTS]
    if wheel_cols:
        obs = obs.clone()
        obs[:, wheel_cols] = 0.0
    return obs


def _override_robot(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.scene.entities = {"robot": get_blackw_robot_cfg()}


def _override_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
    sensors = []
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            continue  # blackW has no foot sites; clearance reward is removed below
        if sensor.name == "feet_ground_contact" and isinstance(sensor, ContactSensorCfg):
            sensor.primary = ContactMatch(
                mode="body",
                pattern=tuple(f"{leg}_foot" for leg in BLACKW_FOOT_NAMES),
                entity="robot",
            )
        sensors.append(sensor)
    cfg.scene.sensors = tuple(sensors)


def _override_actions(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.actions = {
        f"wheel_leg_{leg.lower()}": WheelLegActionCfg(
            entity_name="robot",
            actuator_names=tuple(
                f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf", "wheel")
            ),
            scale=_wheel_leg_scale(leg),
            preserve_order=True,
        )
        for leg in BLACKW_FOOT_NAMES
    }


def _override_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    actor = cfg.observations["actor"]
    actor.history_length = BLACKW_ACTOR_HISTORY_LENGTH
    actor.flatten_history_dim = True
    terms = actor.terms
    terms["joint_pos"] = replace(
        terms["joint_pos"],
        func=_blackw_joint_pos_rel,
        params={
            **terms["joint_pos"].params,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BLACKW_JOINT_NAMES, preserve_order=True
            ),
        },
    )
    terms["joint_vel"] = replace(
        terms["joint_vel"],
        func=envs_mdp.joint_vel_rel,
        params={
            **terms["joint_vel"].params,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BLACKW_JOINT_NAMES, preserve_order=True
            ),
        },
    )
    # blackW's IMU sensor is named imu_gyro, not the black XML's imu_ang_vel; use the
    # entity-state observations instead of the base config's builtin-sensor terms.
    terms["base_ang_vel"] = replace(
        terms["base_ang_vel"], func=envs_mdp.base_ang_vel, params={}
    )
    terms["projected_gravity"] = replace(
        terms["projected_gravity"], func=envs_mdp.projected_gravity, params={}
    )
    actor.terms = {name: terms[name] for name in BLACKW_ACTOR_OBS_TERM_ORDER}
    for name, scale in BLACKW_ACTOR_OBS_SCALE.items():
        actor.terms[name] = replace(actor.terms[name], scale=scale)

    joint_obs = ObservationTermCfg(
        func=envs_mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BLACKW_JOINT_NAMES, preserve_order=True
            )
        },
    )
    joint_vel_obs = ObservationTermCfg(
        func=envs_mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BLACKW_JOINT_NAMES, preserve_order=True
            )
        },
    )
    cfg.observations["critic"] = replace(
        cfg.observations["critic"],
        terms={
            "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
            "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
            "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
            "joint_pos": joint_obs,
            "joint_vel": joint_vel_obs,
            "actions": ObservationTermCfg(func=envs_mdp.last_action),
            "command": ObservationTermCfg(
                func=envs_mdp.generated_commands,
                params={"command_name": "twist"},
            ),
            "height_scan": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": "terrain_scan"},
                scale=0.2,
            ),
        },
    )


def _override_events(cfg: ManagerBasedRlEnvCfg) -> None:
    reset_base = cfg.events["reset_base"]
    cfg.events = {
        "reset_base": reset_base,
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.03, 0.03),
                "velocity_range": (-1.0, 1.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=BLACKW_JOINT_NAMES),
            },
        ),
    }


def _platform_height_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Small dense signal for lifting the root toward a platform top."""
    import torch

    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(root_z - 0.70, min=0.0, max=0.5)


def _climb_height_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Dense trunk-height signal used by the official-obstacle training recipe.

    Flat standing root is ~0.69 m; any lift above the crouched range gets a
    gradient even before the sparse platform-top success fires.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(root_z - 0.55, min=0.0, max=0.60)


def _lane_deviation(env, max_abs_y: float = 0.9, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Terminate lateral bypass: the Part2 judge scores only inside the lane."""
    import torch

    asset = env.scene[asset_cfg.name]
    origins = env.scene.terrain.env_origins
    lateral = asset.data.root_link_pos_w[:, 1] - origins[:, 1]
    return torch.abs(lateral) > max_abs_y


def _nonfinite_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Terminate/reset envs whose physics state became non-finite.

    nan_guard only dumps; without a termination the NaN state can reach the
    next mjwarp.step and turn into a device-side assert.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    data = asset.data
    finite = (
        torch.isfinite(data.root_link_pos_w).all(dim=-1)
        & torch.isfinite(data.root_link_quat_w).all(dim=-1)
        & torch.isfinite(data.joint_pos).all(dim=-1)
        & torch.isfinite(data.joint_vel).all(dim=-1)
    )
    # nan_guard also watches qacc/qacc_warmstart; a finite root can still have a
    # non-finite acceleration that asserts on the next mjwarp.step.
    raw = getattr(env, "sim", None)
    raw = getattr(raw, "data", None)
    for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata"):
        tensor = getattr(raw, name, None) if raw is not None else None
        if tensor is not None:
            finite &= torch.isfinite(tensor).all(dim=-1)
    return ~finite


def _low_base_height(env, min_root_z: float = 0.55, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    import torch

    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_root_z


def _platform_jump_reward(
    env,
    x_min: float = 1.4,
    x_max: float = 3.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reward upward base motion only near the platform; encourages a step/jump."""
    import torch

    asset = env.scene[asset_cfg.name]
    origins = env.scene.terrain.env_origins
    x = asset.data.root_link_pos_w[:, 0] - origins[:, 0]
    vz = asset.data.root_link_lin_vel_w[:, 2]
    near = (x >= x_min) & (x <= x_max)
    return near * torch.clamp(vz, 0.0, 2.0)


def _platform_command(cfg: ManagerBasedRlEnvCfg) -> None:
    """Platform task command set: all 3 deployment axes plus stop/turn combos.

    Deployment uses [vx, vy, wz]; training must not collapse to vx-only. Keep
    forward-dominant speed (obstacle approach) but vary vy/wz and allow stops.
    """
    import os

    command = cfg.commands["twist"]
    stage = os.environ.get("BLACKW_COMMAND_STAGE", "wide").lower()
    if stage == "fixed":
        command.ranges.lin_vel_x = (0.5, 0.5)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.resampling_time_range = (10.0, 10.0)
        command.rel_forward_envs = 1.0
        command.rel_standing_envs = 0.0
    elif stage == "narrow":
        # Skill-acquisition stage: still all three axes, but small vy/yaw so most
        # samples actually interact with the obstacle.
        command.ranges.lin_vel_x = (0.25, 1.0)
        command.ranges.lin_vel_y = (-0.15, 0.15)
        command.ranges.ang_vel_z = (-0.2, 0.2)
        command.rel_forward_envs = 1.0
        command.rel_standing_envs = 0.0
    else:
        command.ranges.lin_vel_x = (0.0, 1.0)
        command.ranges.lin_vel_y = (-0.30, 0.30)
        command.ranges.ang_vel_z = (-1.0, 1.0)
        command.rel_forward_envs = 0.0
        command.rel_standing_envs = 0.10
    command.ranges.heading = None
    command.heading_command = False
    command.resampling_time_range = (4.0, 6.0)
    command.rel_forward_envs = 0.0
    command.rel_standing_envs = 0.10


def _platform_top_success(
    env,
    min_root_z: float = 0.72,
    goal_x_min: float = 1.9,
    goal_x_max: float = 3.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Official-platform success proxy: root over the platform AND raised onto it.

    The previous goal-x-only reward let a CCD-disabled/penetrating robot reach x
    without climbing (false 70% success).  Require the base to be raised to the
    10 cm platform height as well.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    origins = env.scene.terrain.env_origins
    x = asset.data.root_link_pos_w[:, 0] - origins[:, 0]
    z = asset.data.root_link_pos_w[:, 2]
    return (x >= goal_x_min) & (x <= goal_x_max) & (z >= min_root_z)


def _override_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    # Foot clearance uses Black foot sites, which do not exist on blackW; the rest of
    # the frozen Black reward set is robot-agnostic.
    cfg.rewards.pop("foot_clearance", None)
    # Black's base-height target is 0.43 m for point-foot; blackW stands at
    # ~0.69 m (wheel radius + leg extension). Leaving 0.43 makes the policy
    # crouch/crawl instead of walking, and was a real training regression.
    if "base_height" in cfg.rewards:
        cfg.rewards["base_height"].params["target_height"] = 0.69


def _apply_blackw_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    _override_robot(cfg)
    _override_sensors(cfg)
    _override_actions(cfg)
    _override_observations(cfg)
    _override_events(cfg)
    _override_rewards(cfg)
    return cfg


def blackw_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _apply_blackw_overrides(_build_black_env_cfg(play=play, rough=False))


def blackw_competition_env_cfg(
    play: bool = False, subterrains: tuple[str, ...] | None = None
) -> ManagerBasedRlEnvCfg:
    """Competition-proxy terrain (Part1/Part2 obstacles) with the blackW contract.

    ``subterrains`` restricts the generator to a subset (specialist policies), e.g.
    only the high-platform, slope or trench columns for the high Part2 tiers.
    """
    cfg = _apply_blackw_overrides(black_competition_env_cfg(play=play))
    if subterrains and cfg.scene.terrain is not None:
        generator = cfg.scene.terrain.terrain_generator
        assert generator is not None
        for name in subterrains:
            if name not in generator.sub_terrains:
                raise ValueError(f"unknown blackW sub-terrain: {name}")
        generator.sub_terrains = {
            name: generator.sub_terrains[name] for name in subterrains
        }
    if cfg.scene.terrain is not None and not play:
        # Stage 2 targets the high-value Part2 tiers: initialise on rows 0..5
        # (high platform 30/40, slope 35/45, trench 20/30), not only the easy rows.
        cfg.scene.terrain.max_init_terrain_level = 5
    if not play:
        # A goal-x-only reward lets the policy walk around obstacle patches (y up to
        # several metres on real Part2 scenes). The judges score inside a narrow lane;
        # terminate lateral bypass so the policy must interact with the obstacle.
        cfg.terminations["lane_deviation"] = TerminationTermCfg(
            func=_lane_deviation, params={"max_abs_y": 0.9}
        )
    return cfg


def blackw_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return blackw_competition_env_cfg(play=play, subterrains=("platform",))


def blackw_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return blackw_competition_env_cfg(
        play=play, subterrains=("smooth_slope_up", "smooth_slope_down")
    )


def blackw_trench_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return blackw_competition_env_cfg(play=play, subterrains=("trench",))


def _exact_env_cfg(terrain_cfg, play: bool) -> ManagerBasedRlEnvCfg:
    cfg = _apply_blackw_overrides(black_competition_env_cfg(play=play))
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator
    generator = cfg.scene.terrain.terrain_generator
    terrain_cfg.flat_patch_sampling = {
        "goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.3)
    }
    # IMPORTANT: the Part2 success/failure predicates branch on the sub-terrain
    # *name* (platform/slope/stairs/trench/high_wall).  Naming this "exact" made
    # every branch miss and all part2_success / part2_clear permanently false.
    if isinstance(terrain_cfg, ExactHighPlatformTerrainCfg):
        terrain_key = "platform"
    elif isinstance(terrain_cfg, ExactSlopeTerrainCfg):
        terrain_key = "slope"
    elif isinstance(terrain_cfg, ExactTrenchTerrainCfg):
        terrain_key = "trench"
    elif isinstance(terrain_cfg, ExactHighWallTerrainCfg):
        terrain_key = "high_wall"
    elif isinstance(terrain_cfg, ExactStairsTerrainCfg):
        terrain_key = "stairs"
    else:
        terrain_key = "exact"
    generator.sub_terrains = {terrain_key: terrain_cfg}
    # Exact ramps/platforms create many wheel/edge contacts; blackW smoke hit
    # nconmax=172 on the 45-degree ramp. Give the exact tasks headroom.
    cfg.sim.nconmax = 1024
    cfg.sim.njmax = 4096
    # Wheel cylinders vs platform box hit a Mujoco-Warp CCD assert in HIM
    # exploration; these tasks are low-speed, so disable CCD.
    cfg.sim.ccd_iterations = 0
    if not play:
        # Start exact tasks on the easy rows; all-hardest rows gave zero goal
        # signal for the vertical 40 cm platform and 45-degree ramp.
        cfg.scene.terrain.max_init_terrain_level = 1
        cfg.terminations["lane_deviation"] = TerminationTermCfg(
            func=_lane_deviation, params={"max_abs_y": 0.9}
        )
    return cfg


def blackw_platform_exact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _exact_env_cfg(
        ExactHighPlatformTerrainCfg(side_walls=False, lane_half_width=0.70), play
    )
    _platform_command(cfg)
    if "base_height" in cfg.rewards:
        cfg.rewards["base_height"].weight = -20.0
    if not play:
        cfg.rewards["platform_height"] = RewardTermCfg(
            func=_platform_height_reward, weight=50.0
        )
        cfg.rewards["platform_jump"] = RewardTermCfg(
            func=_platform_jump_reward, weight=5.0
        )
        cfg.rewards["climb_height"] = RewardTermCfg(
            func=_climb_height_reward,
            weight=float(os.environ.get("BLACKW_CLIMB_W", "8.0")),
        )
        # Goal-x alone is gameable (CCD-off penetration / walking around). Replace
        # the sparse x-goal with a real platform-top condition for the pilot.
        cfg.rewards.pop("competition_goal", None)
        cfg.terminations.pop("competition_goal_reached", None)
        # Strict four-wheel support success instead of root-x/root-z proxy.
        cfg.terminations["part2_success"] = TerminationTermCfg(
            func=_part2_success, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        cfg.terminations["part2_failure"] = TerminationTermCfg(
            func=_part2_failure, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        cfg.rewards["part2_clear"] = RewardTermCfg(
            func=_part2_clear_reward,
            weight=float(os.environ.get("BLACKW_CLEAR_W", "200.0")),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        cfg.terminations["low_base_height"] = TerminationTermCfg(
            func=_low_base_height, params={"min_root_z": 0.40}
        )
        # Manual success-gated row: BLACKW_MAX_ROW=0/1/...; only promote after
        # the previous row passes the 20/20 low-tier gate.
        if cfg.scene.terrain is not None:
            cfg.scene.terrain.max_init_terrain_level = int(
                os.environ.get("BLACKW_MAX_ROW", "0")
            )
        cfg.terminations["lane_deviation"].params["max_abs_y"] = 0.60
    return cfg


def blackw_slope_exact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _exact_env_cfg(ExactSlopeTerrainCfg(), play)


def blackw_trench_exact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _exact_env_cfg(ExactTrenchTerrainCfg(), play)


def blackw_official_mixed_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Official Part2 geometry mixed terrain for HIM training.

    This replaces the proxy sub-terrains rather than adding another proxy. The
    first curriculum row is the easiest official tier for each category.
    """
    cfg = _apply_blackw_overrides(black_competition_env_cfg(play=play))
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator
    sampling = {"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.3)}
    terrains = {
        "platform": ExactHighPlatformTerrainCfg(flat_patch_sampling=sampling),
        "high_wall": ExactHighWallTerrainCfg(flat_patch_sampling=sampling),
        "slope": ExactSlopeTerrainCfg(flat_patch_sampling=sampling),
        "stairs": ExactStairsTerrainCfg(flat_patch_sampling=sampling),
        "trench": ExactTrenchTerrainCfg(flat_patch_sampling=sampling),
    }
    cfg.scene.terrain.terrain_generator.sub_terrains = terrains
    # Official trench is ~5.8 m from spawn; 8 m patches let it overflow rows.
    cfg.scene.terrain.terrain_generator.size = (16.0, 8.0)
    cfg.sim.nconmax = 1024
    cfg.sim.njmax = 4096
    cfg.sim.ccd_iterations = 0
    if not play:
        cfg.scene.terrain.max_init_terrain_level = int(
            os.environ.get("BLACKW_MAX_ROW", "0")
        )
        cfg.terminations["lane_deviation"] = TerminationTermCfg(
            func=_lane_deviation, params={"max_abs_y": 0.80}
        )
        command = cfg.commands["twist"]
        command.ranges.lin_vel_x = (0.25, 1.0)
        command.ranges.lin_vel_y = (-0.15, 0.15)
        command.ranges.ang_vel_z = (-0.2, 0.2)
        command.ranges.heading = None
        command.heading_command = False
        command.resampling_time_range = (4.0, 6.0)
        command.rel_forward_envs = 1.0
        command.rel_standing_envs = 0.0
        cfg.rewards["competition_progress"] = RewardTermCfg(
            func=competition_goal_progress,
            weight=float(os.environ.get("BLACKW_PROGRESS_W", "3.0")),
            params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
        )
        cfg.rewards["competition_goal"] = RewardTermCfg(
            func=competition_goal_reward, weight=10.0,
            params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
        )
        cfg.rewards["climb_height"] = RewardTermCfg(
            func=_climb_height_reward,
            weight=float(os.environ.get("BLACKW_CLIMB_W", "8.0")),
        )
        # Judge-aligned terminal terms: all four wheels must reach the support
        # surface; a trench wheel drop invalidates the attempt. Borrowed from the
        # parallel Part2 agent's read-only recipe (see cross-agent-comparison.md).
        cfg.terminations.pop("competition_goal_reached", None)
        cfg.rewards.pop("competition_goal", None)
        cfg.terminations["part2_success"] = TerminationTermCfg(
            func=_part2_success, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        cfg.terminations["part2_failure"] = TerminationTermCfg(
            func=_part2_failure, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        cfg.rewards["part2_clear"] = RewardTermCfg(
            func=_part2_clear_reward,
            weight=float(os.environ.get("BLACKW_CLEAR_W", "200.0")),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    return cfg


def blackw_platform_easy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Platform-shape curriculum stage: 5->10 cm before the official 10 cm tier."""
    cfg = _exact_env_cfg(
        ExactHighPlatformTerrainCfg(
            platform_height_range=(0.05, 0.10),
            side_walls=False,
            lane_half_width=0.70,
        ),
        play,
    )
    _platform_command(cfg)
    if "base_height" in cfg.rewards:
        cfg.rewards["base_height"].weight = -20.0
    if not play:
        cfg.rewards["platform_height"] = RewardTermCfg(
            func=_platform_height_reward, weight=50.0
        )
        cfg.rewards["platform_jump"] = RewardTermCfg(
            func=_platform_jump_reward, weight=5.0
        )
        cfg.rewards["climb_height"] = RewardTermCfg(
            func=_climb_height_reward,
            weight=float(os.environ.get("BLACKW_CLIMB_W", "8.0")),
        )
        cfg.rewards.pop("competition_goal", None)
        cfg.terminations.pop("competition_goal_reached", None)
        cfg.terminations["official_platform_success"] = TerminationTermCfg(
            func=_platform_top_success, params={"min_root_z": 0.71}
        )
        cfg.terminations["low_base_height"] = TerminationTermCfg(
            func=_low_base_height, params={"min_root_z": 0.40}
        )
        if cfg.scene.terrain is not None:
            cfg.scene.terrain.max_init_terrain_level = 0
        cfg.terminations["lane_deviation"].params["max_abs_y"] = 0.60
    return cfg


def blackw_platform_ccd_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Easy platform stage with CCD enabled (official wheel-box contact)."""
    cfg = blackw_platform_easy_env_cfg(play=play)
    cfg.sim.ccd_iterations = 50
    return cfg


def blackw_platform_micro_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """1-3 cm platform stage to verify that success/reward can fire at all."""
    cfg = _exact_env_cfg(
        ExactHighPlatformTerrainCfg(
            platform_height_range=(0.01, 0.03),
            side_walls=False,
            lane_half_width=0.70,
        ),
        play,
    )
    _platform_command(cfg)
    if "base_height" in cfg.rewards:
        cfg.rewards["base_height"].weight = -20.0
    if not play:
        cfg.rewards["platform_height"] = RewardTermCfg(
            func=_platform_height_reward, weight=50.0
        )
        cfg.rewards["platform_jump"] = RewardTermCfg(
            func=_platform_jump_reward, weight=5.0
        )
        cfg.rewards["climb_height"] = RewardTermCfg(
            func=_climb_height_reward,
            weight=float(os.environ.get("BLACKW_CLIMB_W", "8.0")),
        )
        cfg.rewards.pop("competition_goal", None)
        cfg.terminations.pop("competition_goal_reached", None)
        cfg.terminations["official_platform_success"] = TerminationTermCfg(
            func=_platform_top_success, params={"min_root_z": 0.70}
        )
        cfg.terminations["low_base_height"] = TerminationTermCfg(
            func=_low_base_height, params={"min_root_z": 0.40}
        )
        if cfg.scene.terrain is not None:
            cfg.scene.terrain.max_init_terrain_level = 0
        cfg.terminations["lane_deviation"].params["max_abs_y"] = 0.60
        cfg.sim.ccd_iterations = 50
    return cfg


def _part2_terrain_state(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_radius: float = 0.103,
):
    """Per-env official Part2 success / rule-violation state from wheel poses.

    v5: wheel-bottom support *band* + base height band + ``|vz|`` gate, matching
    the strict judge; old v4 only checked a lower wheel bound, which allowed
    jumping over the platform to count as success.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    origins = terrain.env_origins
    origins = origins if isinstance(origins, torch.Tensor) else torch.as_tensor(origins)
    origins = origins.to(env.device)
    types = torch.as_tensor(terrain.terrain_types, device=env.device)
    levels = torch.as_tensor(terrain.terrain_levels, device=env.device)
    num_rows = int(terrain.cfg.terrain_generator.num_rows)
    difficulty = levels.to(torch.float32) / max(1.0, float(num_rows - 1))
    body_names = asset.body_names
    wheel_ids = [body_names.index(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]
    wpos = asset.data.body_link_pos_w[:, wheel_ids, :]
    wx = wpos[..., 0] - origins[:, 0:1]
    wz = wpos[..., 2] - wheel_radius
    rx = asset.data.root_link_pos_w[:, 0] - origins[:, 0]
    root_z = asset.data.root_link_pos_w[:, 2]
    root_vz = asset.data.root_link_lin_vel_w[:, 2]
    n = env.num_envs
    success = torch.zeros(n, dtype=torch.bool, device=env.device)
    violation = torch.zeros(n, dtype=torch.bool, device=env.device)
    columns = list(terrain.cfg.terrain_generator.sub_terrains.keys())

    def branch(name):
        return types == columns.index(name) if name in columns else torch.zeros(
            n, dtype=torch.bool, device=env.device)

    wx_min = wx.min(dim=1).values

    if "platform" in columns:
        m = branch("platform")
        top = 0.10 + 0.30 * difficulty
        on_surface = ((wz >= (top - 0.06).unsqueeze(1))
                      & (wz <= (top + 0.20).unsqueeze(1))).all(dim=1)
        success |= m & on_surface & (rx >= 2.0) & (rx <= 3.3) & (
            root_z >= top + 0.30) & (root_z <= top + 0.75) & (root_vz.abs() < 0.4)
    if "stairs" in columns:
        m = branch("stairs")
        # Match the official judge stairs semantics, not the old full
        # up+landing+down traversal. The judge succeeds once the robot reaches
        # goal_x_min=3.3 with enough base height; it does not require descending
        # the far flight. The old hard-coded down_end=6.20 made stairs_success
        # unreachable (false negative), while the policy could pass the judge.
        stairs_cfg = terrain.cfg.terrain_generator.sub_terrains.get("stairs")
        if stairs_cfg is not None and hasattr(stairs_cfg, "total_height_range"):
            lo, hi = stairs_cfg.total_height_range
            total = float(lo) + difficulty * (float(hi) - float(lo))
            step_h = total / float(stairs_cfg.num_steps)
        else:
            step_h = 0.10 + 0.0333 * difficulty
        on_stairs = (wz >= -0.06).all(dim=1) & (
            wz <= (step_h + 0.30).unsqueeze(1)).all(dim=1)
        success |= m & (rx >= 3.30) & on_stairs & (
            root_z >= step_h + 0.38) & (root_z <= step_h + 1.10) & (
            root_vz.abs() < 0.4)
    if "slope" in columns:
        m = branch("slope")
        rise = torch.tan(torch.deg2rad(20.0 + 25.0 * difficulty)) * 2.0
        on_surface = ((wz >= (rise - 0.06).unsqueeze(1))
                      & (wz <= (rise + 0.20).unsqueeze(1))).all(dim=1)
        success |= m & on_surface & (rx >= 3.90) & (
            root_z >= rise + 0.30) & (root_z <= rise + 0.75) & (root_vz.abs() < 0.4)
    if "high_wall" in columns:
        m = branch("high_wall")
        success |= m & (wx_min >= 2.02) & (wz.max(dim=1).values <= 0.08) & (rx >= 2.27)
    if "plum" in columns:
        m = branch("plum")
        in_cubes = (wx >= 3.90) & (wx <= 11.00)
        violation |= m & (in_cubes & (wz < 0.05)).any(dim=1)
        on_platform = ((wz >= 0.14) & (wz <= 0.40)).all(dim=1)
        success |= m & (wx_min >= 11.10) & on_platform & (
            rx >= 11.10) & (root_z >= 0.62) & (root_z <= 1.10) & (root_vz.abs() < 0.4)
    if "trench" in columns:
        m = branch("trench")
        gap_w = 0.10 + 0.20 * difficulty
        near_x = 3.80
        far_x = near_x + gap_w
        allowed_drop = min(0.5 * 0.30, float(wheel_radius))
        floor_z = 0.30 - allowed_drop
        in_slot = (wx >= near_x - 0.06) & (wx <= (far_x + 0.06).unsqueeze(1))
        violation |= m & (in_slot & (wz < floor_z - 0.01)).any(dim=1)
        on_plateau = ((wz >= 0.24) & (wz <= 0.50)).all(dim=1)
        success |= m & (wx_min >= far_x + 0.15) & on_plateau & (
            root_z >= 0.68) & (root_z <= 1.10) & (root_vz.abs() < 0.4)
    return success, violation


def _part2_clear_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """One-step success bonus for the official Part2 success predicate."""
    import torch

    success, _ = _part2_terrain_state(env, asset_cfg)
    return success.to(torch.float32)


def _part2_failure(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Termination: attempt became invalid per the strict Part2 judge."""
    _, violation = _part2_terrain_state(env, asset_cfg)
    return violation


def _part2_success(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Termination: all four wheels reached the target support surface."""
    success, _ = _part2_terrain_state(env, asset_cfg)
    return success



