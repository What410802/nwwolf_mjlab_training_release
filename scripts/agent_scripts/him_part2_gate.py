"""Strict Part2 low-tier gate for a HIM checkpoint.

Run on the training machine from the MjLab training repo, e.g.:

  python him_part2_gate.py --task blackw-official-him \
    --checkpoint logs/.../model_300.pt --category platform --row 0 \
    --attempts 20 --out gate_platform_r0.json

It uses the task's strict `part2_success` termination and samples the command
range configured on the env; no learning is performed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

CATEGORIES = ("platform", "slope", "stairs", "high_wall", "trench")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="blackw-official-him")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--category", choices=CATEGORIES, required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--attempts", type=int, default=20)
    ap.add_argument("--vx-min", type=float, default=0.0)
    ap.add_argument("--vx-max", type=float, default=1.0)
    ap.add_argument("--vy", type=float, default=0.30)
    ap.add_argument("--wz", type=float, default=1.0)
    # 0.6 / 1.0 / 1.3 command-envelope robustness sweep.
    ap.add_argument("--command-scale", type=float, default=1.0)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # Spawn/heading jitter around the training reset pose.
    ap.add_argument("--jitter-x", type=float, default=0.05)
    ap.add_argument("--jitter-y", type=float, default=0.05)
    ap.add_argument("--jitter-yaw", type=float, default=0.05)
    ap.add_argument("--physics", choices=("train", "deploy"), default="train")
    ap.add_argument("--check-config-only", action="store_true")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.check_config_only and (args.checkpoint is None or args.out is None):
        raise SystemExit("--checkpoint and --out are required unless --check-config-only")
    # IMPORTANT: use the training cfg, not play=True.  Black HIM's strict
    # part2_success/failure terms are training-only; play=True silently returns
    # an empty metric (all-zero gate) and episode_length_s=1e9.
    cfg = load_env_cfg(args.task, play=False)
    # Keep the terminal state observable: we must read part2_success/failure
    # before the auto-reset overwrites the episode.
    cfg.auto_reset = False
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    # Clean observations for evaluation (no actor/critic noise).
    for group in cfg.observations.values():
        group.enable_corruption = False
    assert "part2_success" in cfg.terminations, (
        f"strict Part2 terms missing from {args.task} training cfg: "
        f"{list(cfg.terminations)}"
    )
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator
    gen = cfg.scene.terrain.terrain_generator
    if args.category not in gen.sub_terrains:
        raise ValueError(f"{args.category} not in {list(gen.sub_terrains)}")
    gen.sub_terrains = {args.category: gen.sub_terrains[args.category]}
    cfg.scene.terrain.max_init_terrain_level = args.row
    cmd = cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (args.vx_min * args.command_scale, args.vx_max * args.command_scale)
    cmd.ranges.lin_vel_y = (-args.vy * args.command_scale, args.vy * args.command_scale)
    cmd.ranges.ang_vel_z = (-args.wz * args.command_scale, args.wz * args.command_scale)
    cmd.resampling_time_range = (2.0, 4.0)
    if args.physics == "deploy":
        # Match the official Part2 scene <option> block as closely as mujoco_warp
        # allows: euler / elliptic cone / impratio=100 / 100 solver iters / CCD.
        # PGS is not implemented by mujoco_warp; newton is kept as an approximation.
        sim = cfg.sim.mujoco
        sim.integrator = "euler"
        sim.cone = "elliptic"
        sim.jacobian = "sparse"
        sim.solver = "newton"
        sim.iterations = 100
        sim.ls_iterations = 50
        sim.ls_tolerance = 0.01
        sim.impratio = 100.0
        sim.ccd_iterations = 35
        sim.timestep = 0.002
        cfg.decimation = 10
        robot = cfg.scene.entities.get("robot")
        if robot is not None and getattr(robot, "init_state", None) is not None:
            robot.init_state.pos = (0.0, 0.0, 0.52)
    # Add explicit spawn/yaw jitter; the training reset only randomizes velocity
    # and joint state, so a gate from one fixed spawn overstates robustness.
    reset_base = cfg.events.get("reset_base")
    if reset_base is not None:
        reset_base.params["pose_range"] = {
            "x": (-args.jitter_x, args.jitter_x),
            "y": (-args.jitter_y, args.jitter_y),
            "yaw": (-args.jitter_yaw, args.jitter_yaw),
        }

    if args.check_config_only:
        step_dt = cfg.sim.mujoco.timestep * cfg.decimation
        print(
            "gate cfg ok: "
            f"task={args.task} category={args.category} row={args.row} "
            f"terms={list(cfg.terminations)} "
            f"episode_steps={int(cfg.episode_length_s / step_dt)} "
            f"terrain={args.category}"
        )
        return

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    term_mgr = env.termination_manager
    assert "part2_success" in term_mgr.active_terms, (
        f"part2_success not active after env build: {term_mgr.active_terms}"
    )
    clip_actions = load_rl_cfg(args.task).clip_actions
    vec = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
    agent_cfg = load_rl_cfg(args.task)
    runner = load_runner_cls(args.task)(vec, asdict(agent_cfg), device="cuda:0")
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = vec.reset()
    done_count = 0
    success = 0
    failure = 0
    out_of_course = 0
    records = []
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    max_steps = int(args.max_steps) if args.max_steps > 0 else int(cfg.episode_length_s / step_dt)
    with torch.no_grad():
        for _ in range(max_steps):
            action = policy(obs)
            action = torch.nan_to_num(
                action, nan=0.0,
                posinf=float(clip_actions) if clip_actions is not None else 100.0,
                neginf=-float(clip_actions) if clip_actions is not None else -100.0,
            )
            if clip_actions is not None:
                action = torch.clamp(action, -clip_actions, clip_actions)
            obs, _, dones, _ = vec.step(action)
            done_ids = dones.bool()
            if done_ids.any():
                succ = term_mgr.get_term("part2_success")
                fail = term_mgr.get_term("part2_failure")
                out = term_mgr.get_term("part2_out_of_course") if "part2_out_of_course" in term_mgr.active_terms else None
                ids = torch.where(done_ids)[0].tolist()
                for i in ids:
                    done_count += 1
                    ok = bool(succ[i])
                    success += int(ok)
                    failure += int(bool(fail[i]))
                    out_of_course += int(bool(out[i])) if out is not None else 0
                    records.append({"attempt": done_count, "success": ok})
                env.reset(env_ids=torch.where(done_ids)[0])
                obs = vec.get_observations()
                if done_count >= args.attempts:
                    break

    out = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "category": args.category,
        "row": args.row,
        "attempts_requested": args.attempts,
        "attempts_done": done_count,
        "success": success,
        "failure": failure,
        "out_of_course": out_of_course if "part2_out_of_course" in term_mgr.active_terms else None,
        # No separate wheel-drop term is exposed by the training task; do not
        # pretend it was measured. The strict part2_failure remains the proxy.
        "wheel_drop": None,
        "metric_availability": {
            "part2_success": True,
            "part2_failure": True,
            "part2_out_of_course": "part2_out_of_course" in term_mgr.active_terms,
            "wheel_drop": False,
        },
        # This is a training-predicate SCREEN, not official judge acceptance.
        "screen_pass": done_count >= args.attempts and success == args.attempts,
        "acceptance_eligible": False,
        "predicate": "training_part2_v6_geometry",
        "success_rate": (success / done_count) if done_count else 0.0,
        "physics": args.physics,
        "command_scale": args.command_scale,
        "jitter": {"x": args.jitter_x, "y": args.jitter_y, "yaw": args.jitter_yaw},
        "command_ranges": {
            "vx": [args.vx_min, args.vx_max],
            "vy": [-args.vy, args.vy],
            "wz": [-args.wz, args.wz],
        },
        "records": records,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
