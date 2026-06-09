from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from plugin import (  # noqa: E402
    ChatState,
    ChatTarget,
    IdleProactiveChatPlugin,
    ResolvedChat,
    compute_required_idle_seconds,
    is_in_quiet_hours,
    parse_target_chat,
)


def run(coro: Any) -> Any:
    """运行不依赖真实 asyncio 事件循环的异步测试协程。"""

    try:
        value = None
        while True:
            value = coro.send(value)
            if hasattr(value, "send"):
                value = run(value)
    except StopIteration as exc:
        return exc.value


class FakeChat:
    def __init__(self) -> None:
        self.open_calls: list[dict[str, Any]] = []

    async def open_session(self, **kwargs: Any) -> dict[str, Any]:
        self.open_calls.append(kwargs)
        chat_type = kwargs["chat_type"]
        target_id = kwargs.get("group_id") or kwargs.get("user_id")
        return {
            "success": True,
            "stream": {
                "stream_id": f"stream:{chat_type}:{target_id}",
                "session_id": f"stream:{chat_type}:{target_id}",
                "group_name": "测试群" if chat_type == "group" else "",
                "user_nickname": "测试用户" if chat_type == "private" else "",
            },
        }


class FailingChat(FakeChat):
    async def open_session(self, **kwargs: Any) -> dict[str, Any]:
        self.open_calls.append(kwargs)
        raise AssertionError("on_load 不应同步打开或创建聊天流")


class PartiallyFailingChat(FakeChat):
    async def open_session(self, **kwargs: Any) -> dict[str, Any]:
        target_id = kwargs.get("group_id") or kwargs.get("user_id")
        if target_id == "bad":
            raise TimeoutError("模拟聊天流解析超时")
        return await super().open_session(**kwargs)


class FakeMessage:
    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages = messages or []
        self.calls: list[dict[str, Any]] = []

    async def get_by_time_in_chat(self, chat_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"chat_id": chat_id, **kwargs})
        return self.messages


class FakeMaisaka:
    def __init__(self, trigger_result: dict[str, Any] | None = None) -> None:
        self.append_calls: list[dict[str, Any]] = []
        self.trigger_calls: list[dict[str, Any]] = []
        self.trigger_result = {"success": True, "queued": True} if trigger_result is None else trigger_result

    async def append_context(self, stream_id: str, segments: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.append_calls.append({"stream_id": stream_id, "segments": segments, **kwargs})
        return {"success": True}

    async def trigger_proactive(self, stream_id: str, intent: str, **kwargs: Any) -> dict[str, Any]:
        self.trigger_calls.append({"stream_id": stream_id, "intent": intent, **kwargs})
        return self.trigger_result


class FakeContext:
    def __init__(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        trigger_result: dict[str, Any] | None = None,
    ) -> None:
        self.logger = logging.getLogger("idle_proactive_chat_test")
        self.chat = FakeChat()
        self.message = FakeMessage(messages)
        self.maisaka = FakeMaisaka(trigger_result)


class FakeTask:
    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        return None


class FakeAsyncioModule:
    def __init__(self) -> None:
        self.created_tasks: list[str] = []

    def create_task(self, coro: Any) -> FakeTask:
        self.created_tasks.append(getattr(getattr(coro, "cr_code", None), "co_name", "unknown"))
        coro.close()
        return FakeTask()


def make_plugin(tmp_path: Path, *, target_chats: list[str] | None = None) -> IdleProactiveChatPlugin:
    plugin = IdleProactiveChatPlugin(state_path=tmp_path / "state.json")
    config = IdleProactiveChatPlugin.build_default_config()
    config["plugin"]["enabled"] = True
    config["targets"]["target_chats"] = target_chats or ["qq:group:123"]
    config["quiet_hours"]["enabled"] = False
    plugin.set_plugin_config(config)
    return plugin


def test_parse_target_chat_accepts_group_and_private_targets() -> None:
    assert parse_target_chat(" qq:group:123456 ") == ChatTarget("qq", "group", "123456")
    assert parse_target_chat("qq:private:987654") == ChatTarget("qq", "private", "987654")


@pytest.mark.parametrize("raw", ["", "qq:channel:123", "qq:group:", "qq:group", "group:123", "qq::123"])
def test_parse_target_chat_rejects_invalid_targets(raw: str) -> None:
    assert parse_target_chat(raw) is None


def test_quiet_hours_supports_normal_and_cross_midnight_windows() -> None:
    assert is_in_quiet_hours("01:30", enabled=True, start="00:00", end="08:00")
    assert not is_in_quiet_hours("12:00", enabled=True, start="00:00", end="08:00")
    assert is_in_quiet_hours("23:30", enabled=True, start="23:00", end="07:00")
    assert is_in_quiet_hours("06:59", enabled=True, start="23:00", end="07:00")
    assert not is_in_quiet_hours("12:00", enabled=True, start="23:00", end="07:00")
    assert not is_in_quiet_hours("01:30", enabled=False, start="00:00", end="08:00")


def test_compute_required_idle_seconds_uses_exponential_backoff_with_cap() -> None:
    values = [compute_required_idle_seconds(7200, 2.0, 86400, attempts) for attempts in range(6)]
    assert values == [7200, 14400, 28800, 57600, 86400, 86400]


def test_on_load_defers_target_resolution_to_background_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import plugin as plugin_module

    fake_asyncio = FakeAsyncioModule()
    monkeypatch.setattr(plugin_module, "_asyncio", fake_asyncio)
    plugin = make_plugin(tmp_path, target_chats=["qq:private:123"])
    ctx = FakeContext()
    ctx.chat = FailingChat()
    plugin._set_context(ctx)

    run(plugin.on_load())

    assert ctx.chat.open_calls == []
    assert "_resolve_target_chats_safely" in fake_asyncio.created_tasks
    assert "_schedule_loop" in fake_asyncio.created_tasks


def test_receive_hook_updates_last_inbound_and_resets_backoff(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    ctx = FakeContext()
    plugin._set_context(ctx)
    plugin._resolved_chats = {
        "stream:group:123": ResolvedChat(
            target=ChatTarget("qq", "group", "123"),
            stream_id="stream:group:123",
            display_name="测试群",
        )
    }
    plugin._states["stream:group:123"] = ChatState(
        last_inbound_at=100.0,
        last_trigger_at=200.0,
        waiting_for_reply=True,
        backoff_attempts=3,
    )

    run(
        plugin.handle_before_process(
            {
                "session_id": "stream:group:123",
                "timestamp": "500.5",
                "platform": "qq",
                "is_notify": False,
                "message_info": {
                    "group_info": {"group_id": "123", "group_name": "测试群"},
                    "user_info": {"user_id": "456", "user_nickname": "用户"},
                },
            }
        )
    )

    state = plugin._states["stream:group:123"]
    assert state.last_inbound_at == 500.5
    assert state.waiting_for_reply is False
    assert state.backoff_attempts == 0


def test_resolve_targets_opens_session_and_restores_latest_user_message(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path, target_chats=["qq:group:123"])
    ctx = FakeContext(messages=[{"timestamp": "900.25"}])
    plugin._set_context(ctx)

    run(plugin._resolve_target_chats(now=1000.0))

    assert ctx.chat.open_calls == [
        {
            "platform": "qq",
            "chat_type": "group",
            "group_id": "123",
            "user_id": "",
        }
    ]
    assert ctx.message.calls[0]["chat_id"] == "stream:group:123"
    assert ctx.message.calls[0]["filter_mai"] is True
    assert ctx.message.calls[0]["limit"] == 1
    assert ctx.message.calls[0]["limit_mode"] == "latest"
    assert plugin._states["stream:group:123"].last_inbound_at == 900.25


def test_resolve_targets_skips_failed_target_and_keeps_other_targets(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path, target_chats=["qq:group:bad", "qq:group:123"])
    ctx = FakeContext(messages=[{"timestamp": "900.25"}])
    ctx.chat = PartiallyFailingChat()
    plugin._set_context(ctx)

    run(plugin._resolve_target_chats(now=1000.0))

    assert [call.get("group_id") for call in ctx.chat.open_calls] == ["123"]
    assert "stream:group:123" in plugin._resolved_chats
    assert plugin._target_key_to_stream_id == {"qq:group:123": "stream:group:123"}


def test_check_once_triggers_only_one_due_chat_with_maisaka_payload(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    ctx = FakeContext()
    plugin._set_context(ctx)
    plugin._resolved_chats = {
        "stream:group:1": ResolvedChat(ChatTarget("qq", "group", "1"), "stream:group:1", "群1"),
        "stream:group:2": ResolvedChat(ChatTarget("qq", "group", "2"), "stream:group:2", "群2"),
    }
    plugin._states = {
        "stream:group:1": ChatState(last_inbound_at=0.0),
        "stream:group:2": ChatState(last_inbound_at=100.0),
    }

    triggered = run(plugin._run_check_once(now=10_000.0))

    assert triggered == 1
    assert len(ctx.maisaka.append_calls) == 1
    assert len(ctx.maisaka.trigger_calls) == 1
    trigger_call = ctx.maisaka.trigger_calls[0]
    assert trigger_call["stream_id"] == "stream:group:1"
    assert trigger_call["priority"] == "low"
    assert trigger_call["metadata"]["platform"] == "qq"
    assert trigger_call["metadata"]["chat_type"] == "group"
    assert trigger_call["metadata"]["target_id"] == "1"
    assert plugin._states["stream:group:1"].waiting_for_reply is True
    assert plugin._states["stream:group:1"].backoff_attempts == 1
    assert plugin._states["stream:group:2"].backoff_attempts == 0


def test_trigger_failure_does_not_increment_backoff(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    ctx = FakeContext(trigger_result={"success": False, "error": "boom"})
    plugin._set_context(ctx)
    chat = ResolvedChat(ChatTarget("qq", "group", "1"), "stream:group:1", "群1")
    plugin._resolved_chats = {chat.stream_id: chat}
    plugin._states = {chat.stream_id: ChatState(last_inbound_at=0.0)}

    triggered = run(plugin._trigger_chat_if_needed(chat, now=10_000.0))

    state = plugin._states[chat.stream_id]
    assert triggered is False
    assert state.waiting_for_reply is False
    assert state.backoff_attempts == 0
    assert state.last_error_at == 10_000.0
