"""Render a LIBERO Object evaluation ledger and its recorded episodes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from html import escape
import json
import os
from pathlib import Path
import textwrap
from urllib.parse import quote


OBJECTS = (
    "字母汤罐", "奶油奶酪", "沙拉酱", "烧烤酱", "番茄酱瓶（ketchup）",
    "番茄酱罐（tomato sauce）", "黄油", "牛奶盒", "巧克力布丁", "橙汁盒",
)
MILK_TASK = 7
RECEIPT_FIELDS = (
    "status", "error", "reason", "message", "frame_id", "source_frame_id", "target", "camera",
    "detection_id", "mask_id", "depth_id", "object_id", "point_cloud_id", "position_m",
    "bbox_xyxy", "valid_surface_samples", "method", "ridge_id", "midpoint_m", "ridge_axis",
    "jaw_closing_axis", "position", "center_m", "size_m", "width_m", "grasp_id", "candidate_id",
    "pose_id", "plan_id", "strategy", "grasp_height_m", "grasp_point_m", "grasp_center_m",
    "target_position_m", "target_quaternion", "waypoint", "outcome", "official_success", "simulation_steps",
)
CSS = """
:root {font-family: system-ui, sans-serif; color: #17212a; background: #f7f9fc}
body {margin: 28px auto; padding: 0 18px; max-width: 1400px}
h1, h2, h3 {color: #102e49} p {line-height: 1.6}
table {border-collapse: collapse; width: 100%; margin: 16px 0 28px}
td, th {padding: 9px; text-align: left; vertical-align: top; border: 1px solid #cedae5}
th {background: #e7f0f7} td {background: #fff}
.table {overflow-x: auto}
img {display: block; max-width: 100%; height: auto; margin: 6px 0}
.flow {max-width: 800px; list-style: none; padding: 0}
.flow li {position: relative; margin: 0 0 30px; padding: 12px; background: white;
          border: 1px solid #90a8ba; border-left: 6px solid #23608d; border-radius: 6px}
.flow li.bad {border-left-color: #a33434}
.flow li + li::before {content: "↓"; position: absolute; top: -28px; left: 18px; color: #376581}
.flow pre {white-space: pre-wrap; overflow-wrap: anywhere}
code {overflow-wrap: anywhere} small {color: #425c73}
"""


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_events(path):
    if not path.is_file():
        return []
    # Preserve every persisted row and its physical order, including malformed rows.
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"unparsed_line": line, "line_number": line_number}
        events.append(event)
    return events


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def relative_url(path, root):
    return quote(os.path.relpath(path, root))


def bool_cell(value):
    return str(value).strip().lower() == "true"


def choose_rows(rows):
    selected = {}
    for row in sorted(rows, key=lambda row: (
        int(row["task_id"]),
        not bool_cell(row.get("episode_completed")),
        -int(row.get("control_steps") or 0) if not bool_cell(row.get("official_success")) else 0,
        int(row["seed"]),
    )):
        if row["status"] != "scored" or row.get("official_success") not in {"True", "False", "true", "false"}:
            continue
        key = (int(row["task_id"]), bool_cell(row["official_success"]))
        selected.setdefault(key, row)
    return selected


@dataclass(frozen=True)
class Episode:
    path: Path
    result: dict
    additional: bool

    @property
    def task(self):
        return self.result.get("task_index", self.result.get("task_id"))

    @property
    def seed(self):
        return self.result["seed"]

    @property
    def success(self):
        return self.result["official_success"]

    @property
    def events(self):
        return self.path / "events.jsonl"

    @property
    def gif(self):
        for name in ("pick-place.gif", "replay.gif"):
            if (self.path / name).is_file():
                return self.path / name
        return next(iter(sorted(self.path.glob("*.gif"))), None)


def load_episode(result_path, additional):
    result = read_json(result_path)
    if not isinstance(result, dict) or not isinstance(result.get("official_success"), bool):
        return None
    if result.get("suite", "libero_object") != "libero_object":
        return None
    episode = Episode(result_path.parent, result, additional)
    if episode.task not in range(len(OBJECTS)) or not isinstance(result.get("seed"), int):
        return None
    return episode


def collect_episodes(root, rows, replay_roots):
    paths = {
        root / row["result"] for row in rows
        if row["status"] == "scored" and row.get("result")
    }
    paths = paths or set(root.rglob("result.json"))
    seen = set()
    for candidates, additional in [(sorted(paths), False), *(
        (sorted(path.rglob("result.json")), True) for path in replay_roots
    )]:
        for path in candidates:
            real_path = path.resolve()
            if real_path in seen:
                continue
            seen.add(real_path)
            episode = load_episode(path, additional)
            if episode:
                yield episode


def choose_media(episodes):
    replay = {}
    for episode in sorted(episodes, key=lambda item: (item.additional, item.seed, item.path)):
        if episode.gif is not None:
            replay.setdefault((episode.task, episode.success), episode)
    return replay


def last_reason_from_events(episode):
    for event in reversed(read_events(episode.events)):
        if isinstance(event, dict) and isinstance(event.get("receipt"), dict):
            receipt = event["receipt"]
            reason = receipt.get("error") or receipt.get("reason") or receipt.get("message")
            if reason and (receipt.get("ok") is False or receipt.get("success") is False):
                return str(reason)
    return ""


def failure_reason(row, episode):
    for source in (row, episode.result if episode else {}):
        reason = source.get("failure_reason") or source.get("error")
        if reason:
            return str(reason)
    if episode:
        agent_report = episode.result.get("agent_report")
        if isinstance(agent_report, dict) and agent_report.get("reason"):
            return str(agent_report["reason"])
        return last_reason_from_events(episode) or "日志未记录明确失败原因"
    return "日志未记录明确失败原因"


def failure_signals(episode):
    targets, motion_failures = [], []
    closed = released = False
    for event in read_events(episode.events):
        if not isinstance(event, dict):
            continue
        arguments, receipt = event.get("arguments"), event.get("receipt")
        if not isinstance(arguments, dict) or not isinstance(receipt, dict):
            continue
        if event.get("tool") == "detect_object":
            target = arguments.get("target")
            if target and (not targets or target != targets[-1]):
                targets.append(target)
        if receipt.get("status") == "motion_failed":
            motion_failures.append((arguments.get("waypoint", "—"), receipt.get("reason", "执行失败")))
        if event.get("tool") == "set_gripper" and receipt.get("ok") is True:
            closed = closed or arguments.get("state") == "close"
            released = released or bool(receipt.get("basket_release_plan_id"))
    messages = []
    if len(targets) > 1:
        messages.append("曾请求识别 " + " → ".join(map(str, targets)))
    if motion_failures:
        last = " · ".join(f"{waypoint}（{reason}）" for waypoint, reason in motion_failures[-2:])
        messages.append(f"控制器失败 {len(motion_failures)} 次。{last}")
    if closed and not released:
        messages.append("本局记录到闭爪，未记录到篮筐上方松爪回执")
    return messages


def tool_rows(path):
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("tools")
    for item in data or []:
        entry = item.get("function", item)
        name = entry.get("name")
        if not name:
            continue
        params = entry.get("parameters") or entry.get("input_schema") or {}
        required = params.get("required", [])
        parameters = "、".join(
            key if key in required else f"{key}（可选）"
            for key in params.get("properties", {})
        )
        yield name, entry.get("description", ""), parameters or "无"


class Renderer:
    def __init__(self, root, html):
        self.root = root
        self.html = html

    def plain(self, value):
        if self.html:
            return escape(str(value))
        return str(value).replace("|", "&#124;").replace("\n", " ")

    def link(self, path, label):
        url = relative_url(path, self.root)
        if self.html:
            return f'<a href="{url}">{escape(label)}</a>'
        return f"[{label}]({url})"

    def gif(self, episode, otherwise, label=None):
        if episode is None or episode.gif is None:
            return otherwise
        preview = None
        if label is None and (episode.path / "table-preview.gif").is_file():
            preview = episode.path / "table-preview.gif"
        width = 220 if label is None else 640
        if label is None:
            label = f"{'补录' if episode.additional else '评测'} · 初始状态 {episode.seed}"
        image_url = relative_url(episode.gif, self.root)
        links = [self.link(episode.gif, label), self.link(episode.path / "result.json", "判分")]
        if episode.events.is_file():
            links.append(self.link(episode.events, "调用日志"))
        sample_failure = []
        if not episode.success:
            sample_failure = [self.plain(text) for text in [
                failure_reason({}, episode), *failure_signals(episode)
            ]]
        if self.html:
            caption = " · ".join(links)
            preview_url = relative_url(preview, self.root) if preview else image_url
            image = f'<img src="{preview_url}" alt="初始状态 {episode.seed} 的回放" width="{width}" loading="lazy">'
            sample = "".join(f"<div><small>{signal}</small></div>" for signal in sample_failure)
            return f'<a href="{image_url}">{image}</a><small>{caption}</small>{sample}'
        if preview:
            preview_url = relative_url(preview, self.root)
            links.insert(0, f"![初始状态 {episode.seed} 的回放]({preview_url})")
        links.extend(sample_failure)
        return " · ".join(links)

    def heading(self, title, level=2):
        if self.html:
            return f"<h{level}>{escape(title)}</h{level}>"
        return "#" * level + " " + title

    def paragraph(self, value):
        return f"<p>{value}</p>" if self.html else value

    def table(self, heading, rows):
        if self.html:
            header = "".join(f"<th>{value}</th>" for value in heading)
            body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
            return f'<div class="table"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'
        return "\n".join([
            "| " + " | ".join(heading) + " |",
            "| " + " | ".join("---" for _ in heading) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ])

    def trace(self, episode):
        events = read_events(episode.events)
        instruction = str(episode.result.get("custom_instruction") or episode.result.get("instruction", ""))
        initial = episode.path / "initial_observation.json"
        if self.html:
            initial_link = self.link(initial, "初始观测回执") if initial.is_file() else ""
            flow = [f"<li><strong>任务输入</strong><div>{escape(instruction)}</div><small>{initial_link}</small></li>"]
            for index, event in enumerate(events, 1):
                name, arguments, receipt = event_parts(event)
                css = ' class="bad"' if isinstance(receipt, dict) and receipt.get("ok") is False else ""
                image_links = event_images(event, episode.path, self)
                turn = event.get("turn") if isinstance(event, dict) else None
                turn_label = f"Qwen 第 {turn} 轮 · " if isinstance(turn, int) else ""
                flow.append(
                    f"<li{css}><strong>{index:02d} · {turn_label}{escape(name)}</strong>"
                    f"<pre>{escape(arguments)}</pre><div>{escape(receipt_excerpt(receipt))}</div>"
                    f"<details><summary>完整回执</summary><pre>{escape(dumps(receipt))}</pre></details>"
                    f"{image_links}</li>"
                )
            verdict = "成功" if episode.success else "失败"
            flow.append(f"<li><strong>LIBERO 判定{verdict}</strong></li>")
            return '<ol class="flow">' + "".join(flow) + "</ol>"
        lines = ["```mermaid", "flowchart TD", f'    start["任务输入<br/>{mermaid_text(instruction)}"]']
        for index, event in enumerate(events, 1):
            name, arguments, receipt = event_parts(event)
            label = f"{index:02d} · {name} {arguments}"
            lines.append(f'    s{index}["{mermaid_text(label)}<br/>{mermaid_text(receipt_excerpt(receipt))}"]')
            if index > 1:
                lines.append(f"    s{index-1} --> s{index}")
            else:
                lines.append("    start --> s1")
            if isinstance(receipt, dict) and receipt.get("ok") is False:
                lines.append(f"    style s{index} fill:#fff1f1,stroke:#a33434")
        verdict = "LIBERO 判定成功" if episode.success else "LIBERO 判定失败"
        if events:
            lines.append(f'    s{len(events)} --> verdict["{verdict}"]')
        return "\n".join([*lines, "```"])


def event_parts(event):
    if not isinstance(event, dict):
        return "日志记录", "{}", event
    name = str(event.get("tool") or event.get("type") or "日志记录")
    return name, dumps(event.get("arguments", {})), event.get("receipt", event)


def event_images(event, episode_dir, renderer):
    if not isinstance(event, dict) or not isinstance(event.get("images"), dict):
        return ""
    links = []
    for name, path in event["images"].items():
        if not isinstance(path, str):
            continue
        image = episode_dir / path
        if image.is_file() and image.resolve().is_relative_to(episode_dir.resolve()):
            links.append(renderer.link(image, name))
    return f"<small>{' · '.join(links)}</small>" if links else ""


def planned_grasps(events):
    candidate_by_id = {}
    for index, event in enumerate(events, 1):
        _, _, receipt = event_parts(event)
        if not isinstance(receipt, dict):
            continue
        for candidate in receipt.get("candidates", []):
            if isinstance(candidate, dict) and "grasp_id" in candidate:
                candidate_by_id[candidate["grasp_id"]] = candidate
        if "plan_id" not in receipt or receipt.get("grasp_id") not in candidate_by_id:
            continue
        candidate = candidate_by_id[receipt["grasp_id"]]
        yield (
            index, receipt["plan_id"], receipt["grasp_id"], candidate.get("strategy", "—"),
            candidate.get("basis", "—"), dumps(candidate.get("position_m", "—")),
            dumps(candidate.get("quaternion_wxyz", "—")),
            dumps(candidate.get("estimated_pad_contact_points_m", "—")),
        )


def receipt_excerpt(receipt):
    if not isinstance(receipt, dict):
        return dumps(receipt)
    fields = {field: receipt[field] for field in RECEIPT_FIELDS if receipt.get(field) is not None}
    for key in ("candidates", "checks"):
        if isinstance(receipt.get(key), list):
            fields[key + "_count"] = len(receipt[key])
    label = {False: "工具拒绝或执行失败"}.get(receipt.get("ok"), "工具回执")
    if not fields:
        fields = {key: value for key, value in receipt.items() if isinstance(value, (bool, int, float, str))}
    return label + (" · " + dumps(fields) if fields else "")


def mermaid_text(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_flow_svg(episode):
    """Export an offline top-to-bottom diagram of the same ordered log."""
    width = 1050
    instruction = episode.result.get("custom_instruction") or episode.result.get("instruction", "")
    nodes = [("任务输入", [instruction], False)]
    for index, event in enumerate(read_events(episode.events), 1):
        name, arguments, receipt = event_parts(event)
        failed = isinstance(receipt, dict) and receipt.get("ok") is False
        nodes.append((f"{index:02d} · {name}", [arguments, receipt_excerpt(receipt)], failed))
    verdict = "成功" if episode.success else "失败"
    nodes.append((f"LIBERO 最终判定{verdict}", [], not episode.success))
    elements = [
        '<text x="24" y="37" font-size="23" font-weight="700">牛奶盒实际工具调用</text>',
        f'<text x="24" y="61">第 {episode.task} 题 · 初始状态 {episode.seed} · '
        f'{len(nodes) - 2} 次日志记录</text>',
    ]
    y = 80
    for index, (heading, paragraphs, failed) in enumerate(nodes):
        color = "#a33434" if failed else "#23608d"
        fill = "#fff3f3" if failed else "#f3f9fe"
        lines = [line for value in paragraphs for line in textwrap.wrap(str(value), width=68)]
        height = 42 + 21 * len(lines)
        if index:
            elements.append(f'<path d="M 56 {y - 25} v 18 l -5 -5 m 5 5 l 5 -5" '
                            'fill="none" stroke="#376581" stroke-width="2"/>')
        elements.append(f'<rect x="24" y="{y}" width="{width - 48}" height="{height}" rx="7" '
                        f'fill="{fill}" stroke="{color}"/>')
        elements.append(f'<text x="43" y="{y + 25}" font-weight="700" fill="{color}">{escape(heading)}</text>')
        for offset, line in enumerate(lines):
            elements.append(f'<text x="43" y="{y + 47 + offset * 21}">{escape(line)}</text>')
        y += height + 25
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {y}" width="{width}" height="{y}" '
        'style="background:#fff;font-family:system-ui,sans-serif;font-size:14px;fill:#162f43">'
        + "\n".join(elements) + "</svg>\n"
    )


def render(renderer, summary, scored_rows, episodes, milk, tools_path):
    preview = choose_media(episodes)
    results = {(ep.task, ep.seed): ep for ep in episodes if not ep.additional}
    reasons = {}
    for row in scored_rows:
        if not bool_cell(row.get("official_success")):
            key = int(row["task_id"]), int(row["seed"])
            counter = reasons.setdefault(key[0], Counter())
            counter[failure_reason(row, results.get(key))] += 1
    overall = summary["overall"]
    rate = overall.get("success_rate")
    rate_text = f"，成功率 {rate:.1%}" if rate is not None and overall["scored"] else ""
    status = "评测已完成" if summary.get("complete") else "评测尚在进行或有待处理回合"
    if summary.get("complete") and not summary.get("full_evaluation"):
        status = "选定范围评测已完成"
    sections = [
        renderer.heading("LIBERO-Object 评测报告", 1),
        renderer.paragraph(
            f"{status}。官方判分 {overall['scored']}/{overall['planned']} 局，成功 {overall['successes']} 局，"
            f"失败 {overall['failures']} 局{rate_text}。基础设施错误 {overall['infrastructure_errors']} 局，"
            f"待跑 {overall['pending']} 局。成功率的分母是已评分局数。"
        ),
        renderer.paragraph(" · ".join([
            renderer.link(renderer.root / "summary.json", "评分汇总"),
            renderer.link(renderer.root / "trials.csv", "逐局成绩"),
        ])),
        renderer.heading("官方任务成绩和回放"),
    ]
    table_rows = []
    for task in summary["tasks"]:
        task_id = task["task_id"]
        success_rate = f"{task['success_rate']:.0%}" if task["scored"] else "—"
        evidence = reasons.get(task_id, Counter())
        reason = " · ".join(f"{text}（{count} 局）" for text, count in evidence.most_common(2))
        if len(evidence) > 2:
            reason += "。其余原因见逐局成绩"
        table_rows.append([
            str(task_id), renderer.plain(OBJECTS[task_id]),
            f"{task['successes']}/{task['scored']}", success_rate,
            f"{task['scored']}/{task['planned']}",
            renderer.gif(preview.get((task_id, True)), "待补回放" if task["successes"] else "暂无成功局"),
            renderer.gif(preview.get((task_id, False)), "待补回放" if task["failures"] else "暂无失败局"),
            renderer.plain(reason or "—"),
        ])
    sections.append(renderer.table(
        ["题号", "抓取目标", "成功局数", "成功率", "评分进度", "成功 GIF", "失败 GIF", "主要失败记录"],
        table_rows,
    ))
    sections.extend([
        renderer.paragraph("补录的 GIF 来自单独运行，按补录当局的实际结果归类，成绩以本次逐局评分为准。"),
        renderer.heading("Qwen 工具清单"),
    ])
    names = list(tool_rows(tools_path))
    if names:
        sections.append(renderer.paragraph(f"本次运行记录了 {len(names)} 个工具定义。"))
        sections.append(renderer.table(
            ["工具名", "提供给模型的说明", "参数"],
            [[renderer.plain(value) for value in row] for row in names],
        ))
        sections.append(renderer.paragraph(renderer.link(tools_path, "完整工具定义")))
    else:
        sections.append(renderer.paragraph("本次运行缺少工具定义快照。"))
    sections.append(renderer.heading("牛奶盒实际调用流程"))
    if milk:
        events = read_events(milk.events)
        verdict = "成功" if milk.success else "失败"
        source = "补录回合" if milk.additional else "评测回合"
        sections.append(renderer.paragraph(
            f"第 {MILK_TASK} 题，{source}，初始状态 {milk.seed}，官方判定{verdict}，"
            f"控制步 {milk.result.get('control_steps', '未记录')}，共 {len(events)} 条有序日志。"
        ))
        sections.append(renderer.paragraph(" · ".join([
            renderer.link(milk.events, "原始逐工具日志"),
            renderer.link(milk.path / "result.json", "官方判分记录"),
            renderer.link(renderer.root / "milk-flow.svg", "从上到下的流程图"),
            *[renderer.link(milk.path / name, "原始模型对话") for name in ("model.jsonl",) if (milk.path / name).is_file()],
        ])))
        sections.append(renderer.paragraph(renderer.gif(milk, "该局未保存回放 GIF。", "观看该局完整回放")))
        grasps = list(planned_grasps(events))
        if grasps:
            sections.append(renderer.heading("本局实际选中的夹爪位姿", 3))
            sections.append(renderer.table(
                ["日志步骤", "动作计划", "候选抓法", "抓取策略", "位姿依据", "夹爪位置 XYZ（米）",
                 "夹爪姿态（wxyz）", "估计双侧接触点 XYZ（米）"],
                [[renderer.plain(value) for value in row] for row in grasps],
            ))
        sections.append(renderer.trace(milk))
    else:
        sections.append(renderer.paragraph("本次运行尚无带调用日志的牛奶盒回合。"))
    if renderer.html:
        return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LIBERO-Object 评测报告</title>' + (
            f"<style>{CSS}</style><body>" + "\n".join(sections) + "</body></html>\n"
        )
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--replays", type=Path, action="append", default=[])
    parser.add_argument("--milk", type=Path, help="benchmark episode directory to use for the milk trace")
    parser.add_argument("--tools", type=Path, help="tool JSON schemas, defaults to RUN_ROOT/tools.json")
    parser.add_argument("--select-replays", action="store_true", help="print a scored seed per task and outcome")
    args = parser.parse_args()
    root = args.run_root.resolve()
    with (root / "trials.csv").open(newline="", encoding="utf-8") as csvfile:
        rows = list(csv.DictReader(csvfile))
    if args.select_replays:
        print(json.dumps([
            {"task_id": task, "seed": int(row["seed"]), "official_success": success}
            for (task, success), row in sorted(choose_rows(rows).items())
        ], ensure_ascii=False, indent=2))
        return
    summary = read_json(root / "summary.json")
    if not summary:
        parser.error(f"missing or invalid summary.json in {root}")
    episodes = list(collect_episodes(root, rows, args.replays))
    if args.milk:
        source = args.milk.resolve()
        episode = next((item for item in episodes if item.path.resolve() == source), None)
        if episode is None:
            episode = load_episode(source / "result.json", True)
        if episode is None or episode.task != MILK_TASK or not read_events(episode.events):
            parser.error("--milk must point to a scored milk episode with a recorded tool trace")
        milk = episode
    else:
        milk = next((
            episode for episode in sorted(episodes, key=lambda item: (
                not item.success, item.gif is None, item.additional, item.seed
            ))
            if episode.task == MILK_TASK and read_events(episode.events)
        ), None)
    schema_sources = [root / "tools.json"]
    schema_sources.extend(ep.path / "tools.json" for ep in episodes if not ep.additional)
    if milk:
        schema_sources.append(milk.path / "tools.json")
    schema = args.tools or next((path for path in schema_sources if path.is_file()), schema_sources[0])
    if milk:
        image = root / "milk-flow.svg"
        image.write_text(render_flow_svg(milk), encoding="utf-8")
        print(image)
    for filename, use_html in (("report.html", True), ("report.md", False)):
        output = root / filename
        output.write_text(render(
            Renderer(root, use_html), summary, [row for row in rows if row["status"] == "scored"],
            episodes, milk, schema,
        ), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
