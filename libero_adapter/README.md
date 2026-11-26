# LIBERO-Object 首题接入

本适配器运行 LIBERO-Object 第 0 题。Franka 需要拿起字母汤罐，再放进篮子。默认使用官方初始状态 0，仿真和评分都调用已安装的 LIBERO。

## 运行

在仓库根目录执行以下命令。

```bash
../benchmarks/.venv-libero/bin/python -m libero_adapter.run scripted
../benchmarks/.venv-libero/bin/python -m libero_adapter.run agent
```

`scripted` 使用固定的工具顺序。`agent` 由本机 Ollama 的 `qwen3.5:4b` 决定工具调用，并在结束前看新拍的双 RGB 图。可以用 `--seed` 选择其他官方初始状态，用 `--output` 指定尚不存在的输出目录。默认输出在项目的 `runs/libero-*` 内。

- `observe` 保存全局和腕部 RGB。
- `plan_pick_place` 从全局 RGB 分割汤罐标签、篮口，用相机标定和名义高度将像素投影到机器人坐标。
- `execute` 通过 LIBERO 的 OSC 控制器移动、夹持并放置。
- `verify_place` 返回新拍的双 RGB 图。官方成功判分由宿主在回合结束后单独记录，模型只能看到工具回执和图像。

`result.json` 保存官方判分及模型的图像核验文字。`events.jsonl` 保存工具调用和返回结果。`model.jsonl` 保存模型请求的图片数量和耗时。`pick-place.gif` 从全局相机和腕部相机逐控制步采集，按 20 帧/秒连续回放，腕部画面位于右下角。

当前首题的色彩阈值和名义高度是任务专用先验。相机关闭仿真深度，抓取规划使用 RGB、标定和机器人反馈。这是一道官方题的接入验证，整套基准的成功率需要另外逐题、逐初始状态评测。
