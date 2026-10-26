"""Simulated front depth camera for the Black velocity tasks.

The camera is a MuJoCo camera mounted on the robot's existing ``d435_link``
(RealSense D435 mount at the front of the trunk).  Rendering goes through
``mujoco_warp``'s ``RenderContext`` (see ``mjlab/sensor/sensor_context.py``), i.e.
the same GPU sensing graph that already serves the terrain raycast sensors -- no
EGL window is involved.

Contract implemented here (policy side):

- raw camera: 64 x 48, vertical fov 70 deg, tilted 15 deg down, depth 0.1-3.0 m;
- observation: ``depth_image`` returns ``(B, 1, H, W)`` float32 in ``[0, 1]``
  (depth / cutoff_distance, clamped), which is what ``rsl_rl``'s ``CNNModel``
  expects for a 2-D observation group;
- the actor then sees ``45-D proprio + CNN latent``; the critic keeps its
  privileged 259-D layout.  Dropout / latency / extrinsic DR is not implemented
  yet -- this file is the first step (sensor + obs pipeline + cost smoke).

Depth is a *raw simulated range image*: no heightmap sensor is used (forbidden in
the actor), any terrain-relative quantity has to be regressed by the network.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from mjlab.managers import ObservationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg

from alldog_mjlab.tasks.velocity.black.black_config import BLACK_CONFIG

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Observation group name for the camera; the runner cfg maps it to the actor.
CAMERA_OBS_GROUP = "camera"
DEPTH_OBS_TERM = "depth"


def depth_camera_cfg() -> CameraSensorCfg:
    """Build the depth camera sensor cfg from ``BLACK_CONFIG.depth``.

    ``BLACK_DEPTH_WIDTH`` / ``BLACK_DEPTH_HEIGHT`` can lower the render
    resolution for a one-off cost A/B without editing the frozen config.
    """
    cfg = BLACK_CONFIG.depth
    width = int(os.environ.get("BLACK_DEPTH_WIDTH", cfg.width))
    height = int(os.environ.get("BLACK_DEPTH_HEIGHT", cfg.height))
    return CameraSensorCfg(
        name="depth_camera",
        parent_body=f"robot/{cfg.parent_body}",
        pos=cfg.position,
        quat=cfg.quaternion,
        fovy=cfg.fovy,
        width=width,
        height=height,
        data_types=("depth",),
        use_textures=False,
        use_shadows=False,
        # Terrain and obstacles live in geom group 0; the robot itself is group 3,
        # so the camera only sees the environment (no self-occlusion by legs).
        enabled_geom_groups=(0,),
    )


def camera_depth(
    env: ManagerBasedRlEnv,
    sensor_name: str = "depth_camera",
    cutoff_distance: float | None = None,
    min_depth: float | None = None,
) -> torch.Tensor:
    """Normalized depth image in CNN format ``(B, 1, H, W)``.

    ``depth / cutoff_distance`` clipped to ``[0, 1]``, with ``min_depth`` as the
    near clamp (a pixel closer than that is treated as ``min_depth``).
    """
    params = BLACK_CONFIG.depth
    cutoff = params.cutoff_distance if cutoff_distance is None else float(cutoff_distance)
    near = params.min_depth if min_depth is None else float(min_depth)
    sensor: CameraSensor = env.scene[sensor_name]
    depth = sensor.data.depth
    assert depth is not None, f"camera '{sensor_name}' has no depth data"
    depth = depth.permute(0, 3, 1, 2)  # (B, H, W, 1) -> (B, 1, H, W)
    # mujoco_warp renders "no hit" (sky) as 0 and can return values far beyond the
    # cutoff for rays that escape the terrain: both mean "at or beyond max range".
    far = torch.full_like(depth, cutoff)
    depth = torch.where(torch.isfinite(depth) & (depth > 0.0), depth, far)
    depth = torch.clamp(depth, min=near, max=cutoff)
    return torch.clamp(depth / cutoff, 0.0, 1.0)


def add_camera_observations(cfg, enable_corruption: bool = True) -> None:
    """Register the camera obs group and its depth term on an env cfg."""
    from mjlab.utils.noise import UniformNoiseCfg as Unoise

    params = BLACK_CONFIG.depth
    cfg.observations[CAMERA_OBS_GROUP] = ObservationGroupCfg(
        terms={
            DEPTH_OBS_TERM: ObservationTermCfg(
                func=camera_depth,
                params={"sensor_name": "depth_camera"},
                noise=Unoise(n_min=-params.obs_noise, n_max=params.obs_noise),
            )
        },
        concatenate_terms=True,
        enable_corruption=enable_corruption,
    )
