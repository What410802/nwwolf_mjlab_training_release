# 远程 GPU 训练机配置（bis-ms7e56）

> 2026-10-20 在 `bis-ms7e56.tail4413b4.ts.net` 上跑通 mjlab 训练的**原生 uv** 环境。此前用 Docker 验证过，原生 uv 跑通后已弃用 Docker（不再需要镜像与容器；本文只记原生方式）。运行仓库 `MyMonoRepo.d` 里的 `@20261019_locomotion/docs/environment.md` 记运行时视角，本文是训练机的配置备份。

## 1. 机器信息（2026-10-20）

| 项 | 值 |
|---|---|
| SSH | `bis-ms7e56.tail4413b4.ts.net`（Tailscale），用户 `bis` |
| GPU | NVIDIA GeForce RTX 5060 Ti，16311 MiB |
| 驱动 | 内核模块与用户态均为 `595.99.02`（Open Kernel Module） |
| CUDA | 系统 toolkit `13.4`（`/usr/local/cuda`，nvcc 13.4.92）；PyTorch 使用 PyPI 的 cu13 依赖 |
| Python（系统） | 3.14（不用于训练） |
| uv | `0.13.3`，装在 `~/.local/bin/uv` |
| pixi | 主机级 `~/.pixi/bin/pixi`（运行时工具用，不参与训练环境） |
| CPU / 内存 / 磁盘 | 12 核 / 30 GB / 约 739 GB 可用 |
| 网络 | 可直连 PyPI 与 GitHub |

## 2. 前置检查

```bash
nvidia-smi
cat /proc/driver/nvidia/version | head -1        # 应与 nvidia-smi 的驱动版本一致
/usr/local/cuda/bin/nvcc --version | tail -2
```

驱动更新后若内核模块与用户态版本不一致，`nvidia-smi` 会报 `Failed to initialize NVML: Driver/library version mismatch`；此时先重启机器（或由 root 卸载/重载 nvidia 模块），版本一致后再继续。

## 3. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version    # 0.13.3
```

## 4. 克隆训练仓库（fork）

```bash
export RC_WORK="$HOME/rc_work"
mkdir -p "$RC_WORK"
git clone https://github.com/What410802/nwwolf_mjlab_training_release.git "$RC_WORK/nwwolf_mjlab_training_release"
cd "$RC_WORK/nwwolf_mjlab_training_release"
git remote add upstream https://github.com/N-W-wolf/nwwolf_mjlab_training_release.git
git remote -v   # origin = 我们的 fork，upstream = 队内上游
```

若已有从上游 clone 的目录，改成：

```bash
git remote rename origin upstream
git remote add origin https://github.com/What410802/nwwolf_mjlab_training_release.git
git fetch origin
```

## 5. 创建环境

```bash
export UV_CACHE_DIR="$RC_WORK/.uv-cache"   # 缓存放仓库外，重建环境快
uv sync --locked
```

结果（2026-10-20，缓存已存在）：**Python 3.12.15、137 个包、约 5 s**；关键版本为 `mjlab 1.6.0`、`mjlab` 依赖 `mujoco 3.11.0`、`mujoco-warp 3.11.0`、`torch 2.14.0+cu130`、`warp-lang 1.17.0`、`rsl-rl-lib 5.4.2`、`numpy 2.5.3`。首次下载（冷缓存）约 5 分钟量级。

> 建议在 fork 里把 `pyproject.toml` 的 `mjlab>=1.6.0` 改成 `mjlab==1.6.0`，`uv.lock` 一起提交，避免上游发版后环境漂移。

## 6. 验证环境

```bash
# PyTorch / CUDA
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# MuJoCo Warp
.venv/bin/python -c "import warp as wp; wp.init(); print(wp.__version__, wp.get_device())"

# EGL 渲染（录屏/截图用；无 DISPLAY 也可）
MUJOCO_GL=egl .venv/bin/python - <<'PY'
import time, mujoco
xml = "<mujoco><worldbody><camera name='cam' pos='0 -3 1'/>" \
      "<geom type='box' size='1 1 .1'/></worldbody></mujoco>"
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
r = mujoco.Renderer(m, 240, 320)
for _ in range(10): r.update_scene(d, camera="cam"); r.render()
t = time.time()
for _ in range(200):
    mujoco.mj_step(m, d); r.update_scene(d, camera="cam"); r.render()
print("EGL per_frame=%.3f ms" % ((time.time()-t)/200*1000))
PY

# 训练冒烟
MUJOCO_GL=egl .venv/bin/train black-flat \
    --env.scene.num-envs 512 --agent.max-iterations 11 --agent.logger tensorboard
```

预期：`torch 2.14.0+cu130 13.0 True NVIDIA GeForce RTX 5060 Ti`；Warp 列出 `cuda:0`；EGL 320×240 简单场景约 0.06 ms/帧；`black-flat` 512 envs × 11 iteration 约 7 s，`Iteration time ≈ 0.67 s`。

EGL 退出时可能出现 `EGLError` 的 `Exception ignored in ...` 清理告警（MuJoCo EGL context 析构顺序问题），不影响渲染与训练；不需要 `DISPLAY`。没有 EGL 时可回退 `MUJOCO_GL=osmesa`，但它是纯 CPU 渲染，只建议用于兼容性兜底。

## 7. 训练 / 导出 / 看曲线

命令与任务/参数以本仓库 `README.md` 为准，常用：

```bash
export UV_CACHE_DIR="$HOME/rc_work/.uv-cache"
cd "$HOME/rc_work/nwwolf_mjlab_training_release"

# 训练（例：rough；默认 4096 envs，显存不足再降）
MUJOCO_GL=egl .venv/bin/train black-rough --agent.logger tensorboard

# 导出 TorchScript 策略
uv run export --task-id black-flat

# TensorBoard（本地看：ssh -L 6006:localhost:6006 bis-ms7e56...）
.venv/bin/tensorboard --logdir logs/rsl_rl --port 6006
```

## 8. 性能基准（原生 uv，2026-10-20）

`black-flat`，每个 env 数跑 20 iteration 后取 `Iteration time`，峰值显存由 `nvidia-smi` 采样；`MUJOCO_GL=egl`。

| envs | iteration time | 峰值显存 | 约合吞吐 |
|---|---|---|---|
| 512 | 0.67 s | — | — |
| 1024 | 0.82 s | 1774 MiB | ≈80k env steps/s |
| 2048 | 1.05 s | 2444 MiB | ≈125k env steps/s |
| 4096 | 1.57 s | 3891 MiB | ≈167k env steps/s |
| 8192 | 2.62 s | 6828 MiB | ≈200k env steps/s |

16 GB 显存余量充足；建议 S1 从 4096 envs 起步、按收敛曲线决定是否上 8192，10k iteration 的默认上限不必跑满。

## 9. 目录与产物

```text
~/rc_work/
├── .uv-cache/                         # uv 缓存（重建环境用）
└── nwwolf_mjlab_training_release/
    ├── .venv/                         # uv 环境（gitignored）
    ├── logs/rsl_rl/                   # TensorBoard 事件与 checkpoint（gitignored）
    ├── src/alldog_mjlab/              # 任务/地形/奖励/导出
    ├── REMOTE_SETUP.md           # 本文件
    ├── pyproject.toml / uv.lock
```

- 训练配置、地形、奖励、导出脚本进 Git；`logs/`、`.venv/` 不进。
- 选中的权重按 `weights/<stage>` 分支或 GitHub Release 管理，不把大量 checkpoint 放主分支。

## 10. 历史与已移除的 Docker 方案

2026-10-19 曾用 `nvidia/cuda:13.2.0-devel-ubuntu24.04` + uv 的镜像跑通训练和 512–8192 envs 基准；2026-10-20 原生 uv 验证通过（数字与 Docker 基本一致：512 envs 0.67 vs 0.66 s）后改为原生 uv，Docker 镜像与 `docker/` 目录不再需要。若日后需要固定 CUDA toolkit 版本或迁移到其他机器，可以重新引入镜像，但当前不维护。
