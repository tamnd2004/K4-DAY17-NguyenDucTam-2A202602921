from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import apply_profile_updates, estimate_tokens, extract_profile_updates, is_question
from model_provider import build_chat_model, has_live_credentials

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI trả lời bằng tiếng Việt. Bạn chỉ biết những gì người dùng nói trong cuộc "
    "trò chuyện hiện tại; nếu thông tin chưa được nhắc ở đây, hãy nói là bạn chưa biết."
)

FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "style": "Style trả lời",
    "interests": "Mối quan tâm kỹ thuật",
}
# Cue phrases that tell which facts a recall question asks for.
_FACT_QUERIES = (
    ("name", ("tên", "là ai", "về mình")),
    ("location", ("ở đâu", "nơi ở", "còn ở")),
    ("profession", ("nghề", "làm gì")),
    ("drink", ("đồ uống", "uống gì")),
    ("food", ("món ăn", "ăn gì")),
    ("pet", ("nuôi", "con gì", "thú cưng")),
    ("style", ("style", "kiểu trả lời", "phong cách")),
    ("interests", ("quan tâm", "là ai", "sở thích")),
)


def requested_fact_keys(question: str) -> list[str]:
    lowered = question.lower()
    return [key for key, cues in _FACT_QUERIES if any(cue in lowered for cue in cues)]


def _fit_bullet_style(lines: list[str], style: str) -> list[str]:
    """Honor a remembered "N bullet" preference by grouping the answer into at most N bullets."""

    match = re.search(r"(\d+) bullet", style)
    if not match or len(lines) <= int(match.group(1)):
        return lines
    count = int(match.group(1))
    bounds = [index * len(lines) // count for index in range(count + 1)]
    return ["- " + "; ".join(line[2:] for line in lines[start:end]) for start, end in zip(bounds, bounds[1:])]


def compose_offline_reply(message: str, facts: dict[str, str], scope: str, min_confidence: float) -> str:
    """Deterministic responder shared by both agents, so only their memory differs.

    Facts come from whatever memory the caller has (`scope` names it). Answers never echo the
    question, so a recall score cannot be earned by repeating the user's words.
    """

    if not is_question(message):
        noted = [FACT_LABELS[key].lower() for key in extract_profile_updates(message, min_confidence)]
        return f"Đã ghi nhận {', '.join(noted)} trong {scope}." if noted else "Đã nắm ý, bạn nói tiếp nhé."

    keys = requested_fact_keys(message) or list(facts)
    if not keys:
        return f"Mình chưa có thông tin nào về bạn trong {scope}."
    lines = [
        f"- {FACT_LABELS[key]}: {facts[key]}" if key in facts else f"- {FACT_LABELS[key]}: chưa có trong {scope}"
        for key in keys
    ]
    return "\n".join(_fit_bullet_style(lines, facts.get("style", "")))


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: short-term memory only.

    Sessions are keyed by `thread_id` (never `user_id`), there is no `User.md` and no compaction,
    so every new thread starts blank and the whole thread is re-sent as prompt on every turn.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        if not force_offline and has_live_credentials(self.config.model):
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Answer one turn. `user_id` is accepted for API parity but deliberately unused."""

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = self._context_tokens(session)

        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                updates = extract_profile_updates(item["content"], self.config.profile_min_confidence)
                facts = apply_profile_updates(facts, updates)
        response = compose_offline_reply(message, facts, "cuộc trò chuyện này", self.config.profile_min_confidence)

        return self._record_turn(session, response, prompt_tokens, estimate_tokens(response), mode="offline")

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = result["messages"][-1]
        response = answer.text.strip()
        usage = getattr(answer, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or self._context_tokens(session)
        agent_tokens = usage.get("output_tokens") or estimate_tokens(response)
        return self._record_turn(session, response, prompt_tokens, agent_tokens, mode="live")

    def _context_tokens(self, session: SessionState) -> int:
        """Prompt load of one turn: system prompt + the full, never-compacted thread."""

        return estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)

    def _record_turn(
        self, session: SessionState, response: str, prompt_tokens: int, agent_tokens: int, mode: str
    ) -> dict[str, Any]:
        # Accumulate per turn, so intermediate turns count toward the benchmark columns.
        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += agent_tokens
        return {"response": response, "mode": mode, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        """Live agent: chat model for the configured provider + `InMemorySaver` (per-thread state only)."""

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        return create_agent(
            model=build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
