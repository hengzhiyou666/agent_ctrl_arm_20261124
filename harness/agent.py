"""Model-to-tool loop. The model can request one structured operation at a time."""

from __future__ import annotations

import json

from captain_harness.react.chat_llm import ChatCompletionsClient

from harness.grasp_tools import GraspTools, TOOL_SCHEMAS, receipt_content
from simulation.franka import GraspError

INSTRUCTIONS = """
You control a simulated Franka arm. Use only the functions provided.
Make exactly one tool call per turn and wait for its reply. Start with observe.
Each observation has one frame_id shared by two color images, fixed overview first and gripper wrist second.
Choose the ball and lift height from the user instruction. Use green_ball for 绿色小球 and orange_ball for 橙色小球.
Call plan_grasp with target, the actual frame_id from observe, and lift_cm as a number in centimeters.
Use 15.5 cm only when the user did not specify a lift height. The planner estimates position and size from the two images.
Then call execute with the plan_id from plan_grasp, followed by verify_grasp with the same plan_id.
Use exact values from receipts, never invent IDs. Completion requires held, height_reached, and vertical all true.
""".strip()


def run_agent(tools: GraspTools, endpoint: str, model: str,
              instruction: str = "抓起画面中的橙色小球，把它抬离桌面并保持住。") -> dict:
    client = ChatCompletionsClient(endpoint=endpoint, max_output_tokens=256, temperature=0,
                                  timeout_s=85, connect_timeout_s=4, stream_max_retries=2,
                                  all_tools=True, hoist_tool_images=True,
                                  extra_body={"reasoning_effort": "none"})
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text",
              "text": instruction}]}]
    for _ in range(10):
        reply = client.respond({"model": model, "instructions": INSTRUCTIONS, "input": items,
                                "tools": TOOL_SCHEMAS, "parallel_tool_calls": False})
        calls = [item for item in reply.items if item.get("type") == "function_call"]
        if len(calls) != 1:
            items.append({"type": "message", "role": "user", "content": [{"type": "input_text",
                          "text": "请只调用一个提供的机器人工具。等待该工具返回结果。"}]})
            continue
        call = calls[0]
        try:
            args = json.loads(call.get("arguments", "{}"))
        except (ValueError, TypeError):
            args = None
        receipt, images = tools.call(call["name"], args)
        items.extend((call, {"type": "function_call_output", "call_id": call["call_id"],
                             "output": receipt_content(receipt, images)}))
        if call["name"] == "verify_grasp" and receipt.get("held") is True:
            if receipt.get("height_reached") is True and receipt.get("vertical") is True:
                return receipt
            raise GraspError("已抓住小球，但未通过高度或垂直方向核验，详见工具回执")
    raise GraspError("模型在本回合的调用预算内未确认完成抓取与抬升")
