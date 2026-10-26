"""blackW wheel-legged velocity task (first milestone: flat)."""

from mjlab.tasks.registry import register_mjlab_task

from alldog_mjlab.tasks.velocity.black.rl_cfg import black_ppo_runner_cfg
from alldog_mjlab.tasks.velocity.black.curriculum_checkpoint import BlackVelocityOnPolicyRunner

from .env_cfgs import (
    blackw_competition_env_cfg,
    blackw_platform_exact_env_cfg,
    blackw_slope_exact_env_cfg,
    blackw_trench_exact_env_cfg,
    blackw_flat_env_cfg,
    blackw_platform_env_cfg,
    blackw_slope_env_cfg,
    blackw_trench_env_cfg,
)

register_mjlab_task(
    task_id="blackw-competition",
    env_cfg=blackw_competition_env_cfg(),
    play_env_cfg=blackw_competition_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="rough"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="blackw-flat",
    env_cfg=blackw_flat_env_cfg(),
    play_env_cfg=blackw_flat_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="flat"),
    runner_cls=BlackVelocityOnPolicyRunner,
)


for _task_id, _builder in (
    ("blackw-platform", blackw_platform_env_cfg),
    ("blackw-slope", blackw_slope_env_cfg),
    ("blackw-trench", blackw_trench_env_cfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_builder(),
        play_env_cfg=_builder(play=True),
        rl_cfg=black_ppo_runner_cfg(stage="rough"),
        runner_cls=BlackVelocityOnPolicyRunner,
    )


for _task_id, _builder in (
    ("blackw-platform-exact", blackw_platform_exact_env_cfg),
    ("blackw-slope-exact", blackw_slope_exact_env_cfg),
    ("blackw-trench-exact", blackw_trench_exact_env_cfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_builder(),
        play_env_cfg=_builder(play=True),
        rl_cfg=black_ppo_runner_cfg(stage="rough"),
        runner_cls=BlackVelocityOnPolicyRunner,
    )


# HIM tasks: actor history 6x57 + estimator velocity + terminal recorder; optional
# official TorchScript warm start via BLACKW_OFFICIAL_WARMSTART.
from .him import (
    blackw_competition_him_env_cfg,
    blackw_flat_him_env_cfg,
    blackw_official_him_env_cfg,
    blackw_platform_ccd_him_env_cfg,
    blackw_platform_easy_him_env_cfg,
    blackw_platform_micro_him_env_cfg,
    blackw_platform_exact_him_env_cfg,
)
from .him_rl_cfg import blackw_him_runner_cfg
from .him_runner import BlackWHimOfficialRunner

for _task_id, _builder in (
    ("blackw-flat-him", blackw_flat_him_env_cfg),
    ("blackw-competition-him", blackw_competition_him_env_cfg),
    ("blackw-platform-exact-him", blackw_platform_exact_him_env_cfg),
    ("blackw-platform-easy-him", blackw_platform_easy_him_env_cfg),
    ("blackw-platform-micro-him", blackw_platform_micro_him_env_cfg),
    ("blackw-platform-ccd-him", blackw_platform_ccd_him_env_cfg),
    ("blackw-official-him", blackw_official_him_env_cfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_builder(),
        play_env_cfg=_builder(play=True),
        rl_cfg=blackw_him_runner_cfg(),
        runner_cls=BlackWHimOfficialRunner,
    )
