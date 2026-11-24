# agent_ctrl_arm_20261124

Franka Panda 在 MuJoCo 桌面上夹住一颗橙色小球，抬离桌面并保持。Qwen3.5-4B 从本机 Ollama 读取指令、相机画面和工具回执，每轮选择一个已定义的工具。

## 本机运行

在项目目录里执行下面的命令开始一局。需要 Ollama 服务正在运行，并已下载 `qwen3.5:4b`。

```bash
python3 robotctl.py agent
```

想先单独验收相机、抓取规划和机械臂动作，可以用固定的工具顺序运行同一个场景。

```bash
python3 robotctl.py scripted
```

每局输出到 `runs/` 的独立目录。终端会显示实际的工具调用和回执。

| 文件 | 内容 |
| --- | --- |
| `grasp.gif` | 仿真相机录到的连续画面 |
| `detail.png` | 结束时的夹爪近景 |
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
- `simulation/` 放 Franka 场景和运动代码。球的位置由 RGB 颜色分割和深度测量估计。逆运动学计算关节目标，规划时检查机械臂会不会碰桌面，接着用夹爪的物理接触夹起小球。

当前路线使用轻量 MuJoCo 桌面场景。识别限定为场景中已知尺寸的橙球，抓取限定在空桌面。模型能读相机与夹爪回执，物体的仿真真值留给独立评分器。评分同时检查抬升高度、两个手指的接触和球的速度。

Franka Panda 的关节、惯性和碰撞模型源于 MuJoCo Menagerie。场景保留了所需的碰撞网格，并加上了桌面、橙球与相机。对应许可在 `simulation/LICENSE-franka-panda.txt`。[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
