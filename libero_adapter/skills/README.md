# 由 Qwen 选择 LIBERO 技能

每轮把工具说明和已有回执发给 Qwen。Qwen 返回一个工具调用，执行层校验参数并执行该次调用，新的回执交给 Qwen 继续判断。

| 技能文件 | 可调用函数 |
|---|---|
| `observation.py` | `observe` 选择全局或腕部相机，`inspect_place` 获取放置后图像 |
| `planning.py` | `plan_grasp` 生成路点，`raise_grasp` 为支持的高物体提高抓取位置 |
| `movement.py` | `move_arm` 只执行一个路点 |
| `gripper.py` | `set_gripper` 单独张开或闭合夹爪 |
| `feedback.py` | `get_gripper_state` 读取实测开口比例，`finish_task` 校验报告所引用的检查帧 |

在仓库根目录执行以下命令。

```bash
../benchmarks/.venv-libero/bin/python -m libero_adapter.skills.agent --task 0 --seed 0
../benchmarks/.venv-libero/bin/python -m libero_adapter.skills.agent --task 7 --seed 0
```

运行使用本机 Ollama 的 `qwen3.5:4b`，默认保存双视角 GIF、逐次调用和判分。`--no-replay` 可跳过 GIF。该原型每局最多 22 轮模型调用和 400 个控制步。Qwen 要报告已入篮，需要引用同一次松爪之后拍摄的有效检查帧。官方目标由仿真在回合结束时独立判定。历史 500 局成绩使用 `libero_adapter.evaluate` 当时的工具接口和预算。
