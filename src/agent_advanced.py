from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_baseline import compose_offline_reply
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, has_live_credentials

# Live mode only. Module-level because `@tool` resolves the (string) annotations against module globals.
try:
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
    from langchain.tools import ToolRuntime, tool
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # offline mode works without LangChain
    create_agent = None

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI trả lời bằng tiếng Việt. Dùng hồ sơ User.md (fact ổn định về người dùng) và "
    "tóm tắt hội thoại trước để trả lời; luôn theo style trả lời ghi trong hồ sơ."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str
    thread_id: str = ""


class AdvancedAgent:
    """Agent B: short-term memory + persistent `User.md` + compact memory.

    Stable facts go to `User.md` (survives new threads); the conversation itself goes to compact
    memory (per thread, summarized when too long). The two paths are independent, so cross-session
    recall never depends on whether a summary happened to keep a fact.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.live_compactions: dict[str, int] = {}

        self.langchain_agent = None
        if not force_offline and has_live_credentials(self.config.model):
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id) + self.live_compactions.get(thread_id, 0)

    def _persist_profile(self, user_id: str, message: str) -> list[str]:
        """Steps 1-2: extract stable facts and upsert them; a correction replaces the old line."""

        updates = extract_profile_updates(message, self.config.profile_min_confidence)
        return [key for key, value in updates.items() if self.profile_store.upsert_fact(user_id, key, value)]

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changed = self._persist_profile(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)
        return self._record_turn(thread_id, response, prompt_tokens, estimate_tokens(response), "offline", changed)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Prompt load of one turn: system prompt + User.md + compact summary + kept recent messages."""

        context = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(context["summary"])
            + sum(estimate_tokens(item["content"]) for item in context["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Deterministic answer from persisted `User.md` facts (same responder as the baseline)."""

        facts = self.profile_store.facts(user_id)
        return compose_offline_reply(message, facts, "User.md", self.config.profile_min_confidence)

    def _record_turn(
        self, thread_id: str, response: str, prompt_tokens: int, agent_tokens: int, mode: str, changed: list[str]
    ) -> dict[str, Any]:
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + agent_tokens
        return {
            "response": response,
            "mode": mode,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_updates": changed,
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Persist deterministically first; the model can still read/correct User.md through its tools.
        changed = self._persist_profile(user_id, message)
        context = AgentContext(
            user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)), thread_id=thread_id
        )
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        messages = result["messages"]
        last_user = max(index for index, item in enumerate(messages) if item.type == "human")
        answers = [item for item in messages[last_user + 1:] if item.type == "ai"]
        response = answers[-1].text.strip()
        usages = [item.usage_metadata or {} for item in answers]
        prompt_tokens = sum(u.get("input_tokens", 0) for u in usages) or self._estimate_prompt_context_tokens(
            user_id, thread_id
        )
        agent_tokens = sum(u.get("output_tokens", 0) for u in usages) or estimate_tokens(response)
        return self._record_turn(thread_id, response, prompt_tokens, agent_tokens, "live", changed)

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + `InMemorySaver` + User.md tools + profile prompt + summarization."""

        if create_agent is None:
            return None

        store = self.profile_store
        live_compactions = self.live_compactions

        @tool
        def read_user_memory(runtime: ToolRuntime[AgentContext]) -> str:
            """Read the persistent User.md profile of the current user."""

            return store.read_text(runtime.context.user_id) or "(User.md đang trống)"

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Save or correct one stable user fact in User.md.

            Use keys such as name, location, profession, style, interests, drink, food, pet.
            A correction replaces the old value. Never save jokes, questions, or temporary details.
            """

            if not re.fullmatch(r"[a-z_]{2,30}", key):
                return "Key không hợp lệ: chỉ dùng chữ thường và dấu gạch dưới."
            changed = store.upsert_fact(runtime.context.user_id, key, value)
            return "Đã lưu." if changed else "Không đổi."

        @dynamic_prompt
        def profile_prompt(request: ModelRequest) -> str:
            profile = store.read_text(request.runtime.context.user_id) or "(trống)"
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n## User.md\n{profile}"

        class CountingSummarization(SummarizationMiddleware):
            """Summarization middleware that also feeds the Compactions column in live mode."""

            def before_model(self, state, runtime):
                update = super().before_model(state, runtime)
                if update is not None:
                    thread_id = runtime.context.thread_id
                    live_compactions[thread_id] = live_compactions.get(thread_id, 0) + 1
                return update

        model = build_chat_model(self.config.model)
        return create_agent(
            model=model,
            tools=[read_user_memory, save_user_fact],
            middleware=[
                profile_prompt,
                CountingSummarization(
                    model=model,
                    trigger=("tokens", self.config.compact_threshold_tokens),
                    keep=("messages", self.config.compact_keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
