# Qwen3.5-4B 控制 Franka 仿真抓球

## LIBERO-Object 评测

Qwen3.5-4B 通过独立工具控制 Franka。10 道题各跑 50 个官方初始状态，通过 **496/500 局（99.2%）**。旧版在相同状态下通过 293/500 局（58.6%）。新版有 469 局获得相机核验确认。 [逐题成绩、逐局记录与双视角回放](results/libero-object-20261126/README.md) · [工具与评测说明](docs/tool_agent.md)

## MuJoCo 抓球示例

MuJoCo 桌面上有橙球和绿球。默认指令是「用爪子将绿色的小球垂直抬高20cm」。Qwen3.5-4B 通过本机 Ollama 读取指令、全局及腕部相机的 RGB 画面和工具回执，选择目标颜色、要求的高度和要调用的工具。

## 本机运行

在项目目录里执行下面的命令开始一局。需要 Ollama 服务正在运行，并已下载 `qwen3.5:4b`。

```bash
python3 robotctl.py agent
```

想先单独验收相机、抓取规划和机械臂动作，可以用固定的工具顺序运行同一个场景。

```bash
python3 robotctl.py scripted
```

也可对调球的位置并改变绿球半径，验收程序能否从图像重新估计球的位置和尺寸。

```bash
python3 robotctl.py scripted --ball-xy 0.46 -0.11 --green-xy 0.49 0.08 --green-radius 0.032
```

用 `--single-ball` 复现旧橙球场景。`--target` 和 `--lift-cm` 指定验收任务，`--instruction` 可改写给模型的指令。

每局输出到 `runs/` 的独立目录。终端会显示实际的工具调用和回执。

| 文件 | 内容 |
| --- | --- |
| `grasp.gif` | 全局视角的连续回放，右下角显示同步腕部画面 |
| `detail.png` | 结束时的夹爪近景 |
| `f1-overview.jpg`、`f1-wrist.jpg` | 同一时刻的全局和腕部 RGB，作为 Agent 的观察输入 |
| `f2-verification-overview.jpg`、`f2-verification-wrist.jpg` | 抓取完成后用于核验的两张图 |
| `events.jsonl` | 每次工具调用及回执 |
| `result.json` | 从相机得到的核验，以及独立物理评分 |

## 新电脑准备

使用 Python 3.10 或更新版本。MuJoCo 需要可用的离屏 EGL 渲染环境，Linux 可以用 Mesa 软件渲染。

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r requirements.txt
git clone https://github.com/waltstephen/CaPtain-Harness.git CaPtain-Harness
git -C CaPtain-Harness checkout f4abc693f88064d3b45a4f00f5d02779993ca9fc
ollama pull qwen3.5:4b
python robotctl.py agent
```

Ollama 保存模型权重，仓库保存仿真场景、机械臂的小型网格文件和控制代码。已下载的 CaPtain-Harness 可以直接放在项目目录里。

## 代码分工

- `robotctl.py` 启动一局，连接模拟器和 Agent，保存回放与评分。
- `harness/` 复用 CaPtain 的模型客户端。Agent 可以调用 `observe`、`plan_grasp`、`execute` 和 `verify_grasp`。工具检查参数及计划编号。抓取轨迹由 Python 服务内的固定程序计算和执行。
- `simulation/ball_vision.py` 从两张 RGB 图分割指定颜色的球形轮廓，用相机标定和三角测量估计球心、距离和半径。`simulation/franka.py` 用逆运动学计算关节目标，检查桌面碰撞，执行抓取和抬升，并根据图像核验做一次限幅高度微调。

当前使用轻量 MuJoCo 桌面场景和理想相机标定。颜色和球形是识别条件，尺寸和位置由 RGB 估计。`--ball-radius` 和 `--green-radius` 只用于配置仿真物理。验收要求视觉复核通过，独立评分再检查指定球的抬升误差、双指接触、球速、全程横向偏移和另一球是否被扰动。高度误差、横向偏移及另一球的位移均以 1 厘米为验收上限。

Franka Panda 的关节、惯性和碰撞模型源于 MuJoCo Menagerie。场景保留了所需的碰撞网格，并加上了桌面、橙球与相机。对应许可在 `simulation/LICENSE-franka-panda.txt`。[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
