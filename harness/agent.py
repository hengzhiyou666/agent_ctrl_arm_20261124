"""Model-to-tool loop. The model can request one structured operation at a time."""

from __future__ import annotations

import json

from captain_harness.react.chat_llm import ChatCompletionsClient

from harness.grasp_tools import GraspTools, TOOL_SCHEMAS, receipt_content
from simulation.franka import GraspError

INSTRUCTIONS = """
You control a simulated Franka arm. Use only the functions provided.
Make exactly one tool call per turn and wait for its reply. Start with observe.
After seeing the orange ball, call plan_grasp with target orange_ball and the actual frame_id from observe.
Then call execute with the plan_id from plan_grasp, followed by verify_grasp with the same plan_id.
Use exact values from receipts, never invent IDs. When held is true, the run is complete.
""".strip()


def run_agent(tools: GraspTools, endpoint: str, model: str) -> dict:
    client = ChatCompletionsClient(endpoint=endpoint, max_output_tokens=256, temperature=0,
                                  timeout_s=85, connect_timeout_s=4, stream_max_retries=2,
                                  all_tools=True, extra_body={"reasoning_effort": "none"})
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text",
              "text": "抓起画面中的橙色小球，把它抬离桌面并保持住。"}]}]
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
        receipt, image = tools.call(call["name"], args)
        items.extend((call, {"type": "function_call_output", "call_id": call["call_id"],
                             "output": receipt_content(receipt, image)}))
        if call["name"] == "verify_grasp" and receipt.get("held") is True:
            return receipt
    raise GraspError("模型在本回合的调用预算内未确认抓住小球")
