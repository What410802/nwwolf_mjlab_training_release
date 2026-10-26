"""Exact Part2 high-platform / slope / trench terrain patches for blackW.

The mixed competition proxy uses easier shapes (e.g. a half-height step before the
platform), which lets a policy learn a step-up strategy that does not transfer to
the real Part2 vertical platform. These classes mirror the actual level XML geometry
(``part2_high_platform_*.xml``, ``part2_slope_*.xml``, ``part2_trench_*.xml``) while
keeping a dense difficulty sweep for PPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains import SubTerrainCfg

_FLOOR_RGBA = (0.45, 0.45, 0.45, 1.0)
_OBSTACLE_RGBA = (0.25, 0.55, 0.9, 1.0)
_RAMP_RGBA = (0.75, 0.72, 0.55, 1.0)
_TRENCH_RGBA = (0.3, 0.1, 0.1, 1.0)
_FLOOR_T = 0.05


def _box(body, size, pos, rgba, quat=None):
    kwargs = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos)
    if quat is not None:
        kwargs["quat"] = quat
    geom = body.add_geom(**kwargs)
    return TerrainGeometry(geom=geom, color=rgba)


def _origin(size):
    return np.array([size[0] / 2, size[1] / 2, 0.0])


def _floor(body, size, rgba=_FLOOR_RGBA):
    return _box(
        body,
        (size[0] / 2, size[1] / 2, _FLOOR_T / 2),
        (size[0] / 2, size[1] / 2, -_FLOOR_T / 2),
        rgba,
    )


def _goal(cfg, x):
    return {"goal": np.array([[x, cfg.size[1] / 2, 0.0]])}


@dataclass(kw_only=True)
class ExactHighPlatformTerrainCfg(SubTerrainCfg):
    """Official Part2 high platform: one 1.5 x 1.6 m box, no helper step.

    Geometry follows ``terrain.zip/map.xml`` (platform_30cm/40cm geoms:
    ``size=(0.75, 0.8, h/2)``); the previous local level XML's 1.2 x 1.2 m
    platform was too small. ``side_walls`` keeps the robot in the official
    0.7 m scoring lane so it cannot learn the proxy bypass.
    """

    platform_height_range: tuple[float, float] = (0.10, 0.40)
    platform_half_x: float = 0.75
    platform_half_y: float = 0.80
    platform_x: float = 2.5
    # Official ability_field has no physical side walls; the judge only enforces
    # a lateral lane. Keep the option for debugging, but do not train with walls.
    side_walls: bool = False
    lane_half_width: float = 0.70
    wall_height: float = 0.60
    wall_thickness: float = 0.05

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        h = self.platform_height_range[0] + difficulty * (
            self.platform_height_range[1] - self.platform_height_range[0]
        )
        cx = self.size[0] / 2 + self.platform_x
        geom = _box(
            body,
            (self.platform_half_x, self.platform_half_y, h / 2),
            (cx, self.size[1] / 2, h / 2),
            _OBSTACLE_RGBA,
        )
        geometries = [_floor(body, self.size), geom]
        if self.side_walls:
            half_y = self.size[1] / 2
            for sign in (-1.0, 1.0):
                geometries.append(
                    _box(
                        body,
                        (self.size[0] / 2, self.wall_thickness / 2, self.wall_height / 2),
                        (
                            self.size[0] / 2,
                            half_y + sign * self.lane_half_width,
                            self.wall_height / 2,
                        ),
                        _OBSTACLE_RGBA,
                    )
                )
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geometries,
            flat_patches=_goal(self, self.size[0] / 2 + self.platform_x + 0.4),
        )


@dataclass(kw_only=True)
class ExactSlopeTerrainCfg(SubTerrainCfg):
    """Official Part2 slope: 2 m horizontal run, 1.4 m wide, 20..45 degrees.

    From ``map.xml`` the real slope meshes have vertical rise
    ``2 * tan(angle)`` (0.728 / 1.155 / 1.400 / 2.000 m), followed by a
    landing platform.
    """

    slope_deg_range: tuple[float, float] = (20.0, 45.0)
    ramp_horizontal: float = 2.0
    ramp_width: float = 1.4
    landing_half_x: float = 0.4
    ramp_start_x: float = 1.5

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        deg = self.slope_deg_range[0] + difficulty * (
            self.slope_deg_range[1] - self.slope_deg_range[0]
        )
        angle = float(np.deg2rad(deg))
        rise = float(np.tan(angle) * self.ramp_horizontal)
        ramp_cx = self.size[0] / 2 + self.ramp_start_x + self.ramp_horizontal / 2
        ramp_cz = rise / 2
        half_len = (self.ramp_horizontal / 2) / float(np.cos(angle))
        quat = (np.cos(angle / 2), 0.0, -np.sin(angle / 2), 0.0)
        ramp = _box(
            body,
            (half_len, self.ramp_width / 2, 0.02),
            (ramp_cx, self.size[1] / 2, ramp_cz),
            _RAMP_RGBA,
            quat=quat,
        )
        landing_cx = ramp_cx + self.ramp_horizontal / 2 + self.landing_half_x
        top = _box(
            body,
            (self.landing_half_x, self.ramp_width / 2, rise / 2),
            (landing_cx, self.size[1] / 2, rise / 2),
            _OBSTACLE_RGBA,
        )
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=[_floor(body, self.size), ramp, top],
            flat_patches=_goal(self, landing_cx),
        )


@dataclass(kw_only=True)
class ExactTrenchTerrainCfg(SubTerrainCfg):
    """Official Part2 trench: 0.3 m raised plateau, 1 m entry/exit ramps, 0.1..0.3 m gap.

    From ``terrain.zip/map.xml`` the official trench is not a pit: the crossing
    sits on a 0.30 m high plateau (1.4 m channel) entered/exited by 1 m wedge
    ramps, and the "gap" is a 0.10/0.20/0.30 m slot through that plateau.
    """

    gap_width_range: tuple[float, float] = (0.10, 0.30)
    plateau_height: float = 0.30
    ramp_run: float = 1.0
    near_half_x: float = 0.50
    channel_half_y: float = 0.70
    start_x: float = 1.5

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        width = self.gap_width_range[0] + difficulty * (
            self.gap_width_range[1] - self.gap_width_range[0]
        )
        cy = self.size[1] / 2
        x0 = self.size[0] / 2 + self.start_x
        angle = float(np.arctan2(self.plateau_height, self.ramp_run))
        ramp_len = float(np.hypot(self.ramp_run, self.plateau_height))
        half_y = self.channel_half_y
        geoms = [_floor(body, self.size)]
        # Entry ramp: ground -> plateau top.
        entry_cx = x0 + self.ramp_run / 2
        geoms.append(_box(
            body,
            (ramp_len / 2, half_y, 0.02),
            (entry_cx, cy, self.plateau_height / 2),
            _RAMP_RGBA,
            quat=(np.cos(angle / 2), 0.0, -np.sin(angle / 2), 0.0),
        ))
        # Near plateau block, then the gap, then the far block.
        near_cx = x0 + self.ramp_run + self.near_half_x
        geoms.append(_box(body, (self.near_half_x, half_y, self.plateau_height / 2),
                          (near_cx, cy, self.plateau_height / 2), _OBSTACLE_RGBA))
        far_start = near_cx + self.near_half_x + width
        far_cx = far_start + self.near_half_x
        geoms.append(_box(body, (self.near_half_x, half_y, self.plateau_height / 2),
                          (far_cx, cy, self.plateau_height / 2), _OBSTACLE_RGBA))
        # Exit ramp: plateau top -> ground.
        exit_cx = far_cx + self.near_half_x + self.ramp_run / 2
        geoms.append(_box(
            body,
            (ramp_len / 2, half_y, 0.02),
            (exit_cx, cy, self.plateau_height / 2),
            _RAMP_RGBA,
            quat=(np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0),
        ))
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geoms,
            flat_patches=_goal(self, exit_cx + self.ramp_run / 2 - 0.35),
        )


@dataclass(kw_only=True)
class ExactHighWallTerrainCfg(SubTerrainCfg):
    """Official Part2 high wall: 1.4 m wide x 0.12 m thick, 10..40 cm high."""

    wall_height_range: tuple[float, float] = (0.10, 0.40)
    wall_half_y: float = 0.70
    wall_thickness: float = 0.12
    wall_x: float = 2.5

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        h = self.wall_height_range[0] + difficulty * (
            self.wall_height_range[1] - self.wall_height_range[0]
        )
        cx = self.size[0] / 2 + self.wall_x
        wall = _box(
            body,
            (self.wall_thickness / 2, self.wall_half_y, h / 2),
            (cx, self.size[1] / 2, h / 2),
            _OBSTACLE_RGBA,
        )
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=[_floor(body, self.size), wall],
            flat_patches=_goal(self, self.size[0] / 2 + self.wall_x + 0.5),
        )


@dataclass(kw_only=True)
class ExactStairsTerrainCfg(SubTerrainCfg):
    """Official Part2 stairs: 6 up steps + 0.8 m landing + 6 down steps.

    ``map.xml`` names the whole object ``stairs_XXcm_{up,landing,down}`` and the
    official ``stairs_XXcm.png`` label covers both flights, so "通过台阶" means
    climbing up, crossing the landing and descending back to the ground on the
    far side.  Official levels are single-step heights 10/15/20/30 cm and total
    heights 0.60/0.90/1.20/1.80 m; the difficulty interpolates 0.6..1.8 m.
    """

    total_height_range: tuple[float, float] = (0.60, 1.80)
    num_steps: int = 6
    tread: float = 0.30
    half_width: float = 0.70
    start_x: float = 1.65
    landing_half_x: float = 0.40

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        total = self.total_height_range[0] + difficulty * (
            self.total_height_range[1] - self.total_height_range[0]
        )
        geoms = [_floor(body, self.size)]
        cy = self.size[1] / 2
        x = self.size[0] / 2 + self.start_x
        # Up flight: each step is a full column of the level height.
        for i in range(self.num_steps):
            top = total * (i + 1) / self.num_steps
            geoms.append(_box(body, (self.tread / 2, self.half_width, top / 2),
                              (x, cy, top / 2), _OBSTACLE_RGBA))
            x += self.tread
        # Landing.
        landing_cx = x + self.landing_half_x
        geoms.append(_box(body, (self.landing_half_x, self.half_width, total / 2),
                          (landing_cx, cy, total / 2), _OBSTACLE_RGBA))
        x += 2.0 * self.landing_half_x
        # Down flight: mirrored columns from total back to one step height.
        for i in range(self.num_steps):
            top = total * (self.num_steps - 1 - i) / self.num_steps
            if top <= 0.0:
                x += self.tread
                continue
            geoms.append(_box(body, (self.tread / 2, self.half_width, top / 2),
                              (x + self.tread / 2, cy, top / 2), _OBSTACLE_RGBA))
            x += self.tread
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geoms,
            flat_patches=_goal(self, x + 0.4),
        )


@dataclass(kw_only=True)
class ExactPlumTerrainCfg(SubTerrainCfg):
    """Official Part2 tier-1 plum pile field (``plum_entry/platform/cubes/exit``).

    Geometry copied from ``map.xml`` (``plum_entry_ramp`` 1.2 m run / 0.2 m rise,
    0.8 x 0.8 m entry and exit platforms at 0.2 m, 24 cubes 0.2 x 0.2 x 0.2 m in
    two staggered rows: same-row spacing 0.6 m, row y offset 0.4 m, first cube
    0.1 m after the entry platform).  Spawn-relative coordinates keep the entry
    ramp starting at ``start_x``.
    """

    start_x: float = 1.8
    ramp_run: float = 1.2
    platform_half: float = 0.4
    cube_half: float = 0.1
    cube_top: float = 0.2
    cubes_per_row: int = 12
    cube_spacing: float = 0.6
    row_dy: float = 0.4
    row1_dy: float = 0.2
    gap_before_first_cube: float = 0.1
    gap_before_exit: float = 0.4

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        cy = self.size[1] / 2
        x = self.size[0] / 2 + self.start_x
        geoms = [_floor(body, self.size)]
        # Entry ramp (0 -> 0.2 m over 1.2 m).
        angle = float(np.arctan2(self.cube_top, self.ramp_run))
        ramp_len = float(np.hypot(self.ramp_run, self.cube_top))
        geoms.append(_box(
            body, (ramp_len / 2, self.platform_half, 0.02),
            (x + self.ramp_run / 2, cy, self.cube_top / 2), _RAMP_RGBA,
            quat=(np.cos(angle / 2), 0.0, -np.sin(angle / 2), 0.0)))
        x += self.ramp_run
        # Entry platform.
        geoms.append(_box(body, (self.platform_half, self.platform_half,
                                 self.cube_top / 2),
                          (x + self.platform_half, cy, self.cube_top / 2),
                          _OBSTACLE_RGBA))
        first_cube_cx = x + 2.0 * self.platform_half + self.gap_before_first_cube + self.cube_half
        # Two staggered rows of piles (row 2 shifted by half a spacing in x).
        row1_y = cy + self.row1_dy
        row2_y = cy + self.row1_dy - self.row_dy
        for row, y in enumerate((row1_y, row2_y)):
            row_off = 0.5 * row * self.cube_spacing
            for i in range(self.cubes_per_row):
                geoms.append(_box(
                    body, (self.cube_half, self.cube_half, self.cube_top / 2),
                    (first_cube_cx + row_off + i * self.cube_spacing, y,
                     self.cube_top / 2), _OBSTACLE_RGBA))
        last_cube_right = (first_cube_cx + 0.5 * self.cube_spacing
                           + (self.cubes_per_row - 1) * self.cube_spacing + self.cube_half)
        # Exit platform + exit ramp.
        exit_cx = last_cube_right + self.gap_before_exit + self.platform_half
        geoms.append(_box(body, (self.platform_half, self.platform_half,
                                 self.cube_top / 2),
                          (exit_cx, cy, self.cube_top / 2), _OBSTACLE_RGBA))
        exit_end = exit_cx + self.platform_half
        geoms.append(_box(
            body, (ramp_len / 2, self.platform_half, 0.02),
            (exit_end + self.ramp_run / 2, cy, self.cube_top / 2), _RAMP_RGBA,
            quat=(np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0)))
        x = exit_end + self.ramp_run
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geoms,
            flat_patches=_goal(self, x + 0.4),
        )
