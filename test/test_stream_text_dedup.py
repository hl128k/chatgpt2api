from __future__ import annotations

import json
import unittest

from services.config import config
from services.protocol.conversation import (
    conversation_events,
    iter_conversation_payloads,
    normalize_messages,
)


def append_event(text: str) -> str:
    return json.dumps({"p": "/message/content/parts/0", "o": "append", "v": text})


def message_event(text: str, message_id: str) -> str:
    return json.dumps({
        "o": "add",
        "v": {
            "message": {
                "id": message_id,
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [text]},
            }
        },
    })


def collect_deltas(payloads: list[str], history_text: str = "", history_messages: list[str] | None = None) -> str:
    parts = []
    for event in iter_conversation_payloads(iter(payloads), history_text, history_messages):
        if event.get("type") == "conversation.delta":
            parts.append(str(event.get("delta") or ""))
    return "".join(parts)


class StreamTextDedupTests(unittest.TestCase):
    def test_new_message_snapshot_is_not_replayed_wholesale(self) -> None:
        answer = "少爷今天有什么我可以效劳的吗？\n\n关于这次升级，我的建议是先看显存。"
        payloads = [append_event("少爷今天有什么我可以效劳的吗？"), append_event("\n\n关于这次升级，")]
        # Upstream restarts the answer as a second message: every snapshot grows from the
        # greeting again, which used to be emitted in full on every single event.
        for end in range(1, len(answer) + 1):
            payloads.append(message_event(answer[:end], "msg-new"))
        payloads.append("[DONE]")

        self.assertEqual(collect_deltas(payloads), answer)

    def test_interleaved_variants_do_not_repeat_the_whole_answer(self) -> None:
        variant_a = "少爷，A 方案更省心。"
        variant_b = "少爷，A 方案更省心，B 方案更快。"
        payloads = [
            message_event(variant_a[:6], "msg-a"),
            message_event(variant_b[:6], "msg-b"),
            message_event(variant_a, "msg-a"),
            message_event(variant_b, "msg-b"),
            "[DONE]",
        ]

        output = collect_deltas(payloads)
        self.assertLessEqual(output.count("少爷"), 1)
        self.assertTrue(output.endswith("B 方案更快。"))

    def test_shrinking_snapshot_does_not_duplicate_streamed_text(self) -> None:
        payloads = [
            append_event("Hello "),
            append_event(","),
            append_event(" world"),
            "[DONE]",
        ]

        output = collect_deltas(payloads)
        self.assertEqual(output.count("Hello"), 1)
        self.assertIn("world", output)

    def test_section_retake_is_stitched_onto_the_emitted_tail(self) -> None:
        first = "少爷，建议先看显存。\n\n## 二、实际生图能力差多少？\n\n少爷，我查到一份 AMD 发布的"
        retake = "## 二、实际生图能力差多少？\n\n少爷，我查到一份 AMD 发布的官方测试，比较适合判断能做什么。"
        payloads = [message_event(first, "msg-1"), message_event(retake, "msg-2"), "[DONE]"]

        output = collect_deltas(payloads)
        self.assertEqual(output, first + "官方测试，比较适合判断能做什么。")
        self.assertEqual(output.count("## 二、"), 1)

    def test_section_retake_with_leading_blank_lines_is_not_repeated(self) -> None:
        header = "## 二、实际生图能力差多少？"
        para = "截至 2026 年 10 月，AMD 已经把本地生图的支持做得比早期完善不少。"
        payloads = [
            message_event("这条路线也许比换整台电脑更划算。", "msg-1"),
            # 第一次重来贴着上一句，后续快照都带前导空行。
            message_event(header, "msg-2"),
        ]
        for end in (8, 20, len(para)):
            payloads.append(message_event("\n\n" + header + "\n\n" + para[:end], "msg-2"))
        payloads.append("[DONE]")

        output = collect_deltas(payloads)
        self.assertEqual(output.count(header), 1)
        self.assertEqual(output.count("截至 2026 年 10 月"), 1)
        self.assertTrue(output.endswith(para))

    def test_history_replay_is_skipped(self) -> None:
        previous = "上一轮的回答。"
        payloads = [
            message_event(previous, "msg-old"),
            append_event("本轮新回答。"),
            "[DONE]",
        ]

        output = collect_deltas(payloads, history_text=previous, history_messages=[previous])
        self.assertEqual(output, "本轮新回答。")

    def test_global_system_prompt_is_injected_once(self) -> None:
        original = config.data.get("global_system_prompt")
        config.data["global_system_prompt"] = "你是严格的代码审查员。"
        try:
            messages = normalize_messages([{"role": "user", "content": "介绍下你自己"}])
            normalized_again = normalize_messages(messages)
        finally:
            if original is None:
                config.data.pop("global_system_prompt", None)
            else:
                config.data["global_system_prompt"] = original

        self.assertEqual(sum(1 for item in messages if item["role"] == "system"), 1)
        self.assertEqual(sum(1 for item in normalized_again if item["role"] == "system"), 1)


class ConversationEventsNormalizationTests(unittest.TestCase):
    def test_pre_normalized_messages_are_not_reinjected(self) -> None:
        original = config.data.get("global_system_prompt")
        config.data["global_system_prompt"] = "全局提示。"
        captured: dict[str, object] = {}

        class FakeBackend:
            access_token = "token-1"

            def stream_conversation(self, messages=None, **_kwargs):
                captured["messages"] = messages
                yield "[DONE]"

            def close(self):
                pass

        try:
            messages = normalize_messages([{"role": "user", "content": "在吗"}])
            list(conversation_events(FakeBackend(), messages=messages, model="auto"))
        finally:
            if original is None:
                config.data.pop("global_system_prompt", None)
            else:
                config.data["global_system_prompt"] = original

        sent = captured["messages"]
        self.assertEqual(sum(1 for item in sent if item["role"] == "system"), 1)


if __name__ == "__main__":
    unittest.main()
