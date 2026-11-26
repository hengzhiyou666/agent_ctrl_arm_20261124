# LIBERO-Object 评测审计（最终）

2026-11-26T18:20:50+08:00 的账本快照已判定 **500/500 局**，官方成功 **496/500（99.2%）**。旧版同批初始状态成功 **293/500（58.6%）**，其中 206 局由失败变为成功，3 局由成功变为失败。运行故障 0 局。旧版完整账本共成功 293/500 局。

官方成功且 Qwen 确认入篮 469 局，官方成功但 Qwen 未确认 27 局，Qwen 确认但官方未通过 0 局。

| 任务 | 官方成功 / 完成 | 旧版同状态成功 | 新版改善 / 退步 | Qwen 确认 | 有恢复后续 / 其中通过 | 失败类别 |
|---|---:|---:|---:|---:|---:|---|
| 0 字母汤罐 | 50/50 | 50 | 0 / 0 | 48 | 2 / 2 | — |
| 1 奶油奶酪 | 48/50 | 46 | 3 / 1 | 27 | 24 / 22 | control_budget ×2 |
| 2 沙拉酱 | 50/50 | 0 | 50 / 0 | 50 | 0 / 0 | — |
| 3 烧烤酱 | 49/50 | 50 | 0 / 1 | 49 | 0 / 0 | bad_initial_frame_id ×1 |
| 4 番茄酱瓶 | 50/50 | 0 | 50 / 0 | 50 | 0 / 0 | — |
| 5 番茄酱罐 | 50/50 | 50 | 0 / 0 | 48 | 2 / 2 | — |
| 6 黄油 | 49/50 | 47 | 3 / 1 | 49 | 0 / 0 | turn_budget ×1 |
| 7 牛奶盒 | 50/50 | 0 | 50 / 0 | 50 | 0 / 0 | — |
| 8 巧克力布丁 | 50/50 | 50 | 0 / 0 | 50 | 0 / 0 | — |
| 9 橙汁盒 | 50/50 | 0 | 50 / 0 | 48 | 2 / 2 | — |

## 工具选择与恢复

**13401/13401 次工具执行**能按轮次关联该局的 Qwen 响应 ID 和调用 ID，覆盖 23 种工具和 107 种实际调用序列。

记录动作错误、抓持异常或放置不确定信号的回合有 30 局，其中 30 局后续真正执行了观察、退让或再次规划等工具，之后 28 局通过官方判定。这是日志顺序关系，单次决策的效果仍需配对对照验证。

| 工具 | 调用次数 | 使用回合 | 回执状态 |
|---|---:|---:|---|
| `move_arm` | 2000 | 499 | motion_failed ×6、waypoint_reached ×1994 |
| `observe` | 1994 | 500 | ok ×1994 |
| `check_grip` | 1957 | 498 | gripped ×540、possible_grip ×516、tool_rejected ×901 |
| `set_gripper` | 994 | 498 | gripper_command_completed ×991、gripper_command_failed ×3 |
| `plan_motion` | 634 | 499 | ok ×503、tool_rejected ×131 |
| `locate_basket` | 632 | 499 | ok ×632 |
| `propose_grasps` | 567 | 499 | ok ×505、tool_rejected ×62 |
| `detect_object` | 549 | 500 | ok ×515、tool_rejected ×34 |
| `verify_place` | 546 | 496 | inside ×469、tool_rejected ×30、uncertain ×47 |
| `locate_object` | 543 | 500 | ok ×507、tool_rejected ×36 |
| `inspect_place` | 521 | 496 | ok ×521 |
| `check_grasps` | 505 | 499 | ok ×505 |
| `finish_task` | 500 | 495 | report_accepted ×493、tool_rejected ×7 |
| `segment_object` | 393 | 389 | ok ×393 |
| `reconstruct_shape` | 350 | 128 | ok ×130、tool_rejected ×220 |
| `move_wrist_view` | 270 | 128 | motion_failed ×11、viewpoint_reached ×259 |
| `estimate_weight` | 218 | 218 | scenario_estimate ×218 |
| `find_ridge` | 95 | 95 | hypothesis ×95 |
| `test_lift` | 48 | 48 | test_lift_reached ×48 |
| `estimate_grip_force` | 32 | 32 | conditional_estimate ×32 |
| `recall_attempts` | 24 | 24 | ok ×24 |
| `diagnose_drop` | 22 | 22 | still_gripped ×1、uncertain ×21 |
| `retreat_arm` | 7 | 4 | retreated ×6、tool_rejected ×1 |

已启动接近动作的抓法按日志统计为 `higher_top_down` ×200、`top_down` ×303。

## 运动中监视

3319 段动作进入监视，3308 段发出至少一次请求。请求共 3319 次，模型有效答复 21 次，请求错误 3298 次，过期或未采用 3318 次。控制窗口内接受继续 1 次，接受停止 0 次，实际返回动作中断 0 次。

监视请求包含超时在内的回报耗时 P50/P95 为 3.214/3.234 秒。有效响应 P50/P95 为 2.848/2.992 秒。错误和未采用可以发生在同一次请求。

- 未采用 `action_already_ended` ×3308
- 未采用 `snapshot_expired` ×10
- 请求错误 `Expected exactly one review_motion decision.` ×1
- 请求错误 `model request failed after 1 attempts: timeout waiting for the chat completion` ×3297

## 失败和常见拒绝

- `check_grip` ×901，This frame is missing or stale. Use a new observe frame or omit frame_id.
- `plan_motion` ×131，Basket estimate is stale. Observe and locate it again before planning.
- `reconstruct_shape` ×129，Captured wrist views for o1: none. Move to a different wrist view and observe it before reconstructing.
- `reconstruct_shape` ×85，Captured wrist views for o1: center. Move to a different wrist view and observe it before reconstructing.
- `propose_grasps` ×51，Use a shape_id reconstructed for this object_id.
- `locate_object` ×33，unknown detection_id; call detect_object first
- `verify_place` ×30，Use inspect_place after completing a release at the basket.
- `detect_object` ×24，unknown frame_id; capture a frame with observe

`check_grip` 的旧帧拒绝可通过省略可选帧编号，或先拍摄一张新图纠正。腕部扫描后用最新帧定位篮筐。重建前核对已有的扫描视角和对象编号。

| 题号 / 状态 | 归类 | 物理步数 | 终局回执 | 回放 |
|---|---|---:|---|---|
| 1 / 1 | `control_budget` | 280 | The agent reached the 44-turn budget. Last rejected action: LIBERO horizon exhausted | [双视角 GIF](replays/task_01/failure/state_01/pick-place.gif) · [工具记录](evidence/failures/task_01/state_01/events.jsonl) |
| 1 / 49 | `control_budget` | 280 | The agent reached the 44-turn budget. Last rejected action: A grasp may still be held. Use its lift waypoint or deliberately release before retreating. | [双视角 GIF](replays/task_01/failure/state_49/pick-place.gif) · [工具记录](evidence/failures/task_01/state_49/events.jsonl) |
| 3 / 0 | `bad_initial_frame_id` | 0 | The agent reached the 44-turn budget. Last rejected action: unknown frame_id; capture a frame with observe | [初始双视角](replays/task_03/failure/state_00/pick-place.gif) · [工具记录](evidence/failures/task_03/state_00/events.jsonl) |
| 6 / 8 | `turn_budget` | 150 | The agent reached the 44-turn budget. Last rejected action: This frame is missing or stale. Use a new observe frame or omit frame_id. | [双视角 GIF](replays/task_06/failure/state_08/pick-place.gif) · [工具记录](evidence/failures/task_06/state_08/events.jsonl) |

[审计原始数据](audit.json) · [评测逐局账本](trials.csv) · [评测配置](manifest.json)
