"""静默主动发言插件。

当配置的聊天流长时间没有新的入站消息时，本插件会向 Maisaka 写入一条
静默事实上下文，并触发 Maisaka 主动任务，让机器人按当前人设自然发起聊天。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import contextlib
import json
import logging
import time

try:
    import asyncio as _asyncio
except OSError:
    # 当前 Windows 测试环境可能无法初始化 _overlapped，生产运行时通常可正常导入 asyncio。
    _asyncio = None

from maibot_sdk import Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import HookMode


LOGGER = logging.getLogger(__name__)


class _AsyncioUnavailableCancelledError(Exception):
    """asyncio 不可用时用于 on_unload 的占位取消异常。"""


_CANCELLED_ERROR = _asyncio.CancelledError if _asyncio is not None else _AsyncioUnavailableCancelledError


@dataclass(frozen=True)
class ChatTarget:
    """配置中的一个白名单聊天目标。"""

    platform: str
    chat_type: str
    target_id: str

    @property
    def key(self) -> str:
        """返回稳定的配置目标键。"""

        return f"{self.platform}:{self.chat_type}:{self.target_id}"


@dataclass(frozen=True)
class ResolvedChat:
    """已经解析为真实聊天流的目标。"""

    target: ChatTarget
    stream_id: str
    display_name: str = ""
    resolved_at: float = 0.0


@dataclass
class ChatState:
    """单个聊天流的静默触发状态。"""

    last_inbound_at: float = 0.0
    last_trigger_at: float = 0.0
    waiting_for_reply: bool = False
    backoff_attempts: int = 0
    last_error_at: float = 0.0

    @classmethod
    def from_dict(cls, data: Any) -> "ChatState":
        """从 JSON 字典恢复状态。"""

        if not isinstance(data, dict):
            return cls()
        return cls(
            last_inbound_at=_coerce_float(data.get("last_inbound_at"), 0.0),
            last_trigger_at=_coerce_float(data.get("last_trigger_at"), 0.0),
            waiting_for_reply=bool(data.get("waiting_for_reply", False)),
            backoff_attempts=max(0, _coerce_int(data.get("backoff_attempts"), 0)),
            last_error_at=_coerce_float(data.get("last_error_at"), 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为可写入 JSON 的字典。"""

        return asdict(self)


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "message-circle-plus"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本")


class TargetsConfig(PluginConfigBase):
    """白名单目标配置。"""

    __ui_label__ = "监控目标"
    __ui_icon__ = "list-checks"
    __ui_order__ = 1

    target_chats: list[str] = Field(
        default_factory=list,
        description='白名单目标，格式: ["qq:group:群号", "qq:private:用户号"]',
    )


class IdleConfig(PluginConfigBase):
    """静默与退避配置。"""

    __ui_label__ = "静默退避"
    __ui_icon__ = "timer-reset"
    __ui_order__ = 2

    idle_threshold_seconds: int = Field(default=7200, ge=1, description="默认静默触发阈值，单位秒")
    backoff_multiplier: float = Field(default=2.0, ge=1.0, description="无人回复时的指数退避倍数")
    max_backoff_seconds: int = Field(default=86400, ge=1, description="退避最大间隔，单位秒")
    failure_retry_seconds: int = Field(default=600, ge=1, description="触发失败后的最小重试间隔，单位秒")


class ScheduleConfig(PluginConfigBase):
    """后台检查配置。"""

    __ui_label__ = "调度"
    __ui_icon__ = "clock"
    __ui_order__ = 3

    check_interval_seconds: int = Field(default=300, ge=10, description="后台检查间隔，单位秒")
    max_triggers_per_check: int = Field(default=1, ge=1, description="每轮检查最多触发几个聊天流")


class QuietHoursConfig(PluginConfigBase):
    """免打扰时间配置。"""

    __ui_label__ = "免打扰"
    __ui_icon__ = "moon"
    __ui_order__ = 4

    enabled: bool = Field(default=True, description="是否启用免打扰时间段")
    start: str = Field(default="00:00", description="免打扰开始时间，HH:MM")
    end: str = Field(default="08:00", description="免打扰结束时间，HH:MM")
    timezone: str = Field(default="Asia/Shanghai", description="免打扰时区")


class ProactiveConfig(PluginConfigBase):
    """Maisaka 主动任务配置。"""

    __ui_label__ = "主动任务"
    __ui_icon__ = "sparkles"
    __ui_order__ = 5

    intent_template: str = Field(
        default=(
            "这个聊天流已经安静了 {idle_duration}。请结合当前人设、关系、记忆和最近上下文，"
            "自然地发起一句轻松的聊天或暖场。不要提到插件、定时任务、监控、静默检测等系统实现细节。"
        ),
        description="触发 Maisaka 主动任务时使用的意图模板",
    )


class IdleProactiveChatConfig(PluginConfigBase):
    """静默主动发言插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    idle: IdleConfig = Field(default_factory=IdleConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    quiet_hours: QuietHoursConfig = Field(default_factory=QuietHoursConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)


def parse_target_chat(raw_target: str) -> Optional[ChatTarget]:
    """解析 ``平台:group/private:号码`` 格式的白名单目标。"""

    parts = [part.strip() for part in str(raw_target or "").split(":")]
    if len(parts) != 3:
        return None

    platform, chat_type, target_id = parts
    if not platform or chat_type not in {"group", "private"} or not target_id:
        return None
    return ChatTarget(platform=platform, chat_type=chat_type, target_id=target_id)


def compute_required_idle_seconds(
    idle_threshold_seconds: int,
    backoff_multiplier: float,
    max_backoff_seconds: int,
    backoff_attempts: int,
) -> int:
    """计算当前退避次数下需要等待的静默秒数。"""

    base_seconds = max(0, int(idle_threshold_seconds))
    multiplier = max(1.0, float(backoff_multiplier))
    cap_seconds = max(1, int(max_backoff_seconds))
    attempts = max(0, int(backoff_attempts))
    required_seconds = int(base_seconds * (multiplier**attempts))
    return min(required_seconds, cap_seconds)


def is_in_quiet_hours(now: str | datetime, *, enabled: bool, start: str, end: str) -> bool:
    """判断给定时间是否落在免打扰窗口内。"""

    if not enabled:
        return False

    current_minutes = _minutes_from_time_input(now)
    start_minutes = _parse_hhmm(start)
    end_minutes = _parse_hhmm(end)
    if current_minutes is None or start_minutes is None or end_minutes is None:
        return False
    if start_minutes == end_minutes:
        return False
    if start_minutes < end_minutes:
        return start_minutes <= current_minutes < end_minutes
    return current_minutes >= start_minutes or current_minutes < end_minutes


class IdleProactiveChatPlugin(MaiBotPlugin):
    """当白名单聊天流静默时触发 Maisaka 主动聊天。"""

    config_model = IdleProactiveChatConfig

    def __init__(self, *, state_path: str | Path | None = None) -> None:
        super().__init__()
        self._state_path = Path(state_path) if state_path is not None else Path(__file__).with_name("data") / "state.json"
        self._states: dict[str, ChatState] = {}
        self._resolved_chats: dict[str, ResolvedChat] = {}
        self._target_key_to_stream_id: dict[str, str] = {}
        self._scheduler_task: Any = None

    async def on_load(self) -> None:
        """加载状态、解析白名单，并启动后台调度。"""

        self._load_states()
        await self._resolve_target_chats(now=time.time())
        if self.config.plugin.enabled and _asyncio is not None:
            self._scheduler_task = _asyncio.create_task(self._schedule_loop())
            self._get_logger().info("静默主动发言插件已加载，后台调度已启动")
            return
        if _asyncio is None:
            self._get_logger().warning("当前运行时无法导入 asyncio，静默主动发言后台调度未启动")
        else:
            self._get_logger().info("静默主动发言插件已加载，插件当前未启用")

    async def on_unload(self) -> None:
        """停止后台任务并保存状态。"""

        if self._scheduler_task is not None and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            with contextlib.suppress(_CANCELLED_ERROR):
                await self._scheduler_task
        self._scheduler_task = None
        self._save_states()
        self._get_logger().info("静默主动发言插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        """处理配置热重载。"""

        del config_data
        await self._resolve_target_chats(now=time.time())
        self._save_states()
        self._get_logger().info("静默主动发言配置已更新: scope=%s, version=%s", scope, version)

    @HookHandler(
        "chat.receive.before_process",
        name="idle_proactive_chat_track_inbound",
        description="旁路记录白名单聊天流的最新入站消息时间，并在收到回复后重置主动发言退避。",
        mode=HookMode.OBSERVE,
    )
    async def handle_before_process(self, message: dict[str, Any], **kwargs: Any) -> None:
        """记录入站消息活跃时间。"""

        del kwargs
        if not self.config.plugin.enabled or bool(message.get("is_notify")):
            return

        stream_id = self._resolve_stream_id_from_message(message)
        if not stream_id or stream_id not in self._resolved_chats:
            return

        message_time = _extract_timestamp(message, default=time.time())
        state = self._get_state(stream_id, default_last_inbound_at=message_time)
        state.last_inbound_at = message_time
        state.last_error_at = 0.0
        if state.waiting_for_reply:
            state.waiting_for_reply = False
            state.backoff_attempts = 0
        self._save_states()

    async def _schedule_loop(self) -> None:
        """后台轮询检查静默聊天流。"""

        if _asyncio is None:
            return

        while True:
            try:
                await _asyncio.sleep(max(10, int(self.config.schedule.check_interval_seconds)))
                await self._run_check_once()
            except _CANCELLED_ERROR:
                raise
            except Exception as exc:
                self._get_logger().error("静默主动发言调度异常: %s", exc, exc_info=True)
                await _asyncio.sleep(60)

    async def _resolve_target_chats(self, *, now: float | None = None) -> None:
        """把配置目标解析为真实聊天流。"""

        current_time = time.time() if now is None else float(now)
        resolved_chats: dict[str, ResolvedChat] = {}
        target_key_to_stream_id: dict[str, str] = {}

        for raw_target in self.config.targets.target_chats:
            target = parse_target_chat(raw_target)
            if target is None:
                self._get_logger().warning("忽略无效主动发言目标配置: %s", raw_target)
                continue

            stream = await self._open_target_session(target)
            stream_id = self._extract_stream_id_from_stream(stream)
            if not stream_id:
                self._get_logger().warning("主动发言目标未解析到聊天流: %s", target.key)
                continue

            display_name = self._extract_display_name_from_stream(stream, target)
            resolved_chats[stream_id] = ResolvedChat(
                target=target,
                stream_id=stream_id,
                display_name=display_name,
                resolved_at=current_time,
            )
            target_key_to_stream_id[target.key] = stream_id

            state = self._get_state(stream_id, default_last_inbound_at=0.0)
            if state.last_inbound_at <= 0:
                restored_time = await self._restore_latest_inbound_at(stream_id, now=current_time)
                state.last_inbound_at = restored_time if restored_time > 0 else current_time

        self._resolved_chats = resolved_chats
        self._target_key_to_stream_id = target_key_to_stream_id
        self._save_states()

    async def _open_target_session(self, target: ChatTarget) -> Any:
        """按平台目标打开或创建聊天流。"""

        return await self.ctx.chat.open_session(
            platform=target.platform,
            chat_type=target.chat_type,
            group_id=target.target_id if target.chat_type == "group" else "",
            user_id=target.target_id if target.chat_type == "private" else "",
        )

    async def _restore_latest_inbound_at(self, stream_id: str, *, now: float) -> float:
        """从历史消息恢复最近一次非 Bot 入站消息时间。"""

        try:
            result = await self.ctx.message.get_by_time_in_chat(
                stream_id,
                start_time="0",
                end_time=str(now),
                limit=1,
                limit_mode="latest",
                filter_mai=True,
                filter_command=True,
            )
        except Exception as exc:
            self._get_logger().warning("恢复聊天流最近消息失败: stream_id=%s error=%s", stream_id, exc)
            return 0.0

        messages = self._extract_messages(result)
        if not messages:
            return 0.0
        return max(_extract_timestamp(message, default=0.0) for message in messages if isinstance(message, dict))

    async def _run_check_once(self, *, now: float | None = None) -> int:
        """执行一轮静默检查，返回成功触发的聊天流数量。"""

        if not self.config.plugin.enabled:
            return 0
        if not self._resolved_chats and self.config.targets.target_chats:
            await self._resolve_target_chats(now=now)
        if self._is_quiet_now(now=now):
            return 0

        current_time = time.time() if now is None else float(now)
        max_attempts = max(1, int(self.config.schedule.max_triggers_per_check))
        attempted_count = 0
        triggered_count = 0
        for chat in self._select_due_chats(current_time):
            if attempted_count >= max_attempts:
                break
            attempted_count += 1
            if await self._trigger_chat_if_needed(chat, now=current_time):
                triggered_count += 1
        return triggered_count

    def _select_due_chats(self, now: float) -> list[ResolvedChat]:
        """选择已经满足静默与退避条件的聊天流，最久未活跃者优先。"""

        due_chats: list[tuple[float, ResolvedChat]] = []
        for stream_id, chat in self._resolved_chats.items():
            state = self._get_state(stream_id, default_last_inbound_at=now)
            if state.last_error_at and now - state.last_error_at < self.config.idle.failure_retry_seconds:
                continue

            required_idle = compute_required_idle_seconds(
                self.config.idle.idle_threshold_seconds,
                self.config.idle.backoff_multiplier,
                self.config.idle.max_backoff_seconds,
                state.backoff_attempts,
            )
            reference_time = max(state.last_inbound_at, state.last_trigger_at)
            if now - reference_time >= required_idle:
                due_chats.append((reference_time, chat))

        due_chats.sort(key=lambda item: item[0])
        return [chat for _, chat in due_chats]

    async def _trigger_chat_if_needed(self, chat: ResolvedChat, *, now: float) -> bool:
        """向 Maisaka 写入静默事实并触发一次主动任务。"""

        state = self._get_state(chat.stream_id, default_last_inbound_at=now)
        idle_seconds = max(0, int(now - max(state.last_inbound_at, state.last_trigger_at)))
        facts_text = self._build_facts_text(chat, state, idle_seconds)
        intent = self._build_intent(chat, idle_seconds)
        metadata = {
            "platform": chat.target.platform,
            "chat_type": chat.target.chat_type,
            "target_id": chat.target.target_id,
            "stream_id": chat.stream_id,
            "idle_seconds": idle_seconds,
            "backoff_attempts": state.backoff_attempts,
        }

        try:
            await self.ctx.maisaka.append_context(
                chat.stream_id,
                [{"type": "text", "content": facts_text}],
                visible_text=facts_text,
                source_kind="plugin:idle_proactive_chat",
                message_id=f"idle-proactive-chat:{chat.stream_id}:{int(now)}",
            )
            result = await self.ctx.maisaka.trigger_proactive(
                chat.stream_id,
                intent,
                reason="聊天流长时间没有新入站消息",
                priority="low",
                metadata=metadata,
            )
        except Exception as exc:
            state.last_error_at = now
            self._save_states()
            self._get_logger().error("触发静默主动发言失败: stream_id=%s error=%s", chat.stream_id, exc, exc_info=True)
            return False

        if isinstance(result, dict) and result.get("success") is False:
            state.last_error_at = now
            self._save_states()
            self._get_logger().warning("静默主动发言主动任务触发失败: %s", result.get("error"))
            return False

        state.last_trigger_at = now
        state.waiting_for_reply = True
        state.backoff_attempts = max(0, state.backoff_attempts) + 1
        state.last_error_at = 0.0
        self._save_states()
        self._get_logger().info("已触发静默主动发言: stream_id=%s idle_seconds=%s", chat.stream_id, idle_seconds)
        return True

    def _build_facts_text(self, chat: ResolvedChat, state: ChatState, idle_seconds: int) -> str:
        """构造写入 Maisaka 上下文的静默事实。"""

        return (
            "[静默主动发言触发]\n"
            f"聊天：{chat.display_name or chat.target.key}\n"
            f"平台：{chat.target.platform}\n"
            f"类型：{chat.target.chat_type}\n"
            f"目标 ID：{chat.target.target_id}\n"
            f"聊天流 ID：{chat.stream_id}\n"
            f"已静默：{_format_duration(idle_seconds)}\n"
            f"退避次数：{state.backoff_attempts}"
        )

    def _build_intent(self, chat: ResolvedChat, idle_seconds: int) -> str:
        """根据配置模板构造 Maisaka 主动任务意图。"""

        template = self.config.proactive.intent_template.strip() or ProactiveConfig().intent_template
        return template.format(
            platform=chat.target.platform,
            chat_type=chat.target.chat_type,
            target_id=chat.target.target_id,
            stream_id=chat.stream_id,
            display_name=chat.display_name or chat.target.key,
            idle_seconds=idle_seconds,
            idle_duration=_format_duration(idle_seconds),
        )

    def _is_quiet_now(self, *, now: float | None = None) -> bool:
        """判断当前是否处于免打扰时间。"""

        quiet_config = self.config.quiet_hours
        if not quiet_config.enabled:
            return False

        current_time = time.time() if now is None else float(now)
        try:
            tz = ZoneInfo(quiet_config.timezone)
            local_now = datetime.fromtimestamp(current_time, tz=tz)
        except (ValueError, ZoneInfoNotFoundError):
            local_now = datetime.fromtimestamp(current_time)
        return is_in_quiet_hours(
            local_now,
            enabled=quiet_config.enabled,
            start=quiet_config.start,
            end=quiet_config.end,
        )

    def _resolve_stream_id_from_message(self, message: dict[str, Any]) -> str:
        """从 Hook 消息或目标映射中解析真实 stream_id。"""

        stream_id = str(message.get("session_id") or message.get("stream_id") or "").strip()
        if stream_id in self._resolved_chats:
            return stream_id

        target_key = self._extract_target_key_from_message(message)
        if not target_key:
            return ""
        return self._target_key_to_stream_id.get(target_key, "")

    @staticmethod
    def _extract_stream_id_from_stream(stream_result: Any) -> str:
        """从 chat.open_session 返回结果中提取 stream_id。"""

        stream = stream_result
        if isinstance(stream_result, dict) and isinstance(stream_result.get("stream"), dict):
            stream = stream_result["stream"]
        if not isinstance(stream, dict):
            return ""
        return str(stream.get("stream_id") or stream.get("session_id") or "").strip()

    @staticmethod
    def _extract_display_name_from_stream(stream_result: Any, target: ChatTarget) -> str:
        """从聊天流结果中提取展示名称。"""

        stream = stream_result
        if isinstance(stream_result, dict) and isinstance(stream_result.get("stream"), dict):
            stream = stream_result["stream"]
        if not isinstance(stream, dict):
            return target.key
        if target.chat_type == "group":
            return str(stream.get("group_name") or target.key).strip()
        return str(stream.get("user_nickname") or stream.get("user_cardname") or target.key).strip()

    @staticmethod
    def _extract_messages(result: Any) -> list[dict[str, Any]]:
        """兼容 SDK 归一化前后的消息查询返回结构。"""

        if isinstance(result, list):
            return [message for message in result if isinstance(message, dict)]
        if isinstance(result, dict):
            messages = result.get("messages") or result.get("result") or []
            if isinstance(messages, list):
                return [message for message in messages if isinstance(message, dict)]
        return []

    @staticmethod
    def _extract_target_key_from_message(message: dict[str, Any]) -> str:
        """按平台、群号或用户号提取配置目标键。"""

        platform = str(message.get("platform") or "qq").strip() or "qq"
        message_info = message.get("message_info")
        if not isinstance(message_info, dict):
            return ""

        group_info = message_info.get("group_info")
        if isinstance(group_info, dict):
            group_id = str(group_info.get("group_id") or "").strip()
            if group_id:
                return f"{platform}:group:{group_id}"

        user_info = message_info.get("user_info")
        if isinstance(user_info, dict):
            user_id = str(user_info.get("user_id") or "").strip()
            if user_id:
                return f"{platform}:private:{user_id}"
        return ""

    def _get_state(self, stream_id: str, *, default_last_inbound_at: float = 0.0) -> ChatState:
        """获取或初始化聊天流状态。"""

        state = self._states.get(stream_id)
        if state is None:
            state = ChatState(last_inbound_at=default_last_inbound_at)
            self._states[stream_id] = state
        return state

    def _load_states(self) -> None:
        """从本地 JSON 读取持久状态。"""

        if not self._state_path.is_file():
            self._states = {}
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._get_logger().warning("读取静默主动发言状态失败: %s", exc)
            self._states = {}
            return

        raw_states = data.get("states") if isinstance(data, dict) else None
        if not isinstance(raw_states, dict):
            self._states = {}
            return
        self._states = {str(stream_id): ChatState.from_dict(raw_state) for stream_id, raw_state in raw_states.items()}

    def _save_states(self) -> None:
        """把状态写入插件本地 JSON 文件。"""

        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "updated_at": time.time(),
                "states": {stream_id: state.to_dict() for stream_id, state in self._states.items()},
            }
            self._state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._get_logger().error("保存静默主动发言状态失败: %s", exc, exc_info=True)


def _parse_hhmm(value: str) -> Optional[int]:
    """解析 HH:MM 为当天分钟数。"""

    parts = [part.strip() for part in str(value or "").split(":")]
    if len(parts) != 2:
        return None
    hour = _coerce_int(parts[0], -1)
    minute = _coerce_int(parts[1], -1)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour * 60 + minute


def _minutes_from_time_input(value: str | datetime) -> Optional[int]:
    """从字符串或 datetime 提取当天分钟数。"""

    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    return _parse_hhmm(str(value or ""))


def _extract_timestamp(message: dict[str, Any], *, default: float) -> float:
    """从消息字典中读取 Unix 时间戳。"""

    for key in ("timestamp", "time"):
        if key in message:
            return _coerce_float(message.get(key), default)
    return default


def _coerce_float(value: Any, default: float) -> float:
    """安全转换 float。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    """安全转换 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_duration(duration_seconds: int) -> str:
    """把秒数格式化为中文时长。"""

    seconds_total = max(0, int(duration_seconds))
    days, remainder = divmod(seconds_total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分钟")
    if seconds or not parts:
        parts.append(f"{seconds}秒")
    return "".join(parts)


def create_plugin() -> IdleProactiveChatPlugin:
    """创建插件实例。"""

    return IdleProactiveChatPlugin()