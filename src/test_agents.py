from __future__ import annotations

import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path) -> LabConfig:
    """Isolated config: state in tmp_path, tiny compact threshold, no API key (offline only)."""

    offline_model = ProviderConfig(provider="openai", model_name="offline", temperature=0.0)
    return LabConfig(
        base_dir=REPO_ROOT,
        data_dir=REPO_ROOT / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=200,
        compact_keep_messages=2,
        model=offline_model,
        judge_model=offline_model,
    )


def make_agents(tmp_path: Path) -> tuple[BaselineAgent, AdvancedAgent]:
    config = make_config(tmp_path)
    return BaselineAgent(config, force_offline=True), AdvancedAgent(config, force_offline=True)


def stress_turns() -> list[str]:
    data = json.loads((REPO_ROOT / "data" / "advanced_long_context.json").read_text(encoding="utf-8"))
    return data[0]["turns"]


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    assert store.read_text("dungct") == ""
    assert store.file_size("dungct") == 0

    content = "# User profile: dungct\n\n## Facts\n- name: DũngCT\n- location: Đà Nẵng\n"
    path = store.write_text("dungct", content)
    assert path.is_file() and tmp_path in path.parents
    assert store.read_text("dungct") == content
    assert store.file_size("dungct") == len(content.encode("utf-8"))

    assert store.edit_text("dungct", "- location: Đà Nẵng", "- location: Huế") is True
    edited = store.read_text("dungct")
    assert "- location: Huế" in edited and "Đà Nẵng" not in edited
    assert store.edit_text("dungct", "- location: Hà Nội", "- location: Huế") is False

    # A hostile user id must not escape the profiles directory.
    assert (config.state_dir / "profiles").resolve() in store.path_for("../../outside").parents


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)

    memory.append("short", "user", "Chào bạn, mình tên là DũngCT.")
    assert memory.compaction_count("short") == 0

    for turn in stress_turns()[:6]:
        for role, content in (("user", turn), ("assistant", "Đã nắm ý, bạn nói tiếp nhé.")):
            before = memory.compaction_count("long")
            memory.append("long", role, content)
            if memory.compaction_count("long") > before:
                # Right after each compaction only the most recent messages stay verbatim.
                assert len(memory.context("long")["messages"]) == config.compact_keep_messages
    assert memory.compaction_count("long") > 0
    assert "NASA" in memory.context("long")["summary"]

    # The agent wires the same layer: a long thread must report compactions too.
    _, advanced = make_agents(tmp_path)
    for turn in stress_turns()[:6]:
        advanced.reply("dungct_stress", "long", turn)
    assert advanced.compaction_count("long") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    baseline, advanced = make_agents(tmp_path)
    facts = "Chào bạn, mình tên là DũngCT và đang làm MLOps engineer."
    question = "Mình tên gì và hiện tại mình làm nghề gì?"

    advanced.reply("dungct", "thread-1", facts)
    recalled = advanced.reply("dungct", "thread-2", question)
    assert recalled["mode"] == "offline"
    assert "DũngCT" in recalled["response"] and "MLOps engineer" in recalled["response"]
    assert advanced.memory_file_size("dungct") > 0

    baseline.reply("dungct", "thread-1", facts)
    assert "DũngCT" in baseline.reply("dungct", "thread-1", question)["response"]  # within-session memory works
    forgotten = baseline.reply("dungct", "thread-2", question)["response"]
    assert "DũngCT" not in forgotten and "MLOps engineer" not in forgotten


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    baseline, advanced = make_agents(tmp_path)
    last_turn: dict[str, int] = {}
    for turn in stress_turns():
        last_turn["baseline"] = baseline.reply("dungct_stress", "long", turn)["prompt_tokens"]
        last_turn["advanced"] = advanced.reply("dungct_stress", "long", turn)["prompt_tokens"]

    assert advanced.compaction_count("long") > 0
    assert baseline.compaction_count("long") == 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    # Baseline re-sends the whole thread; Advanced carries User.md + summary + a few recent messages.
    assert last_turn["advanced"] < last_turn["baseline"] / 2


def test_correction_replaces_old_fact(tmp_path: Path) -> None:
    _, advanced = make_agents(tmp_path)
    advanced.reply("dungct", "thread-1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    advanced.reply("dungct", "thread-2", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.")
    advanced.reply("dungct", "thread-2", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    profile = advanced.profile_store.read_text("dungct")
    assert profile.count("- location:") == 1 and profile.count("- profession:") == 1
    answer = advanced.reply("dungct", "thread-3", "Hiện tại mình đang ở đâu và làm nghề gì?")["response"]
    assert "Huế" in answer and "MLOps engineer" in answer
    assert "Đà Nẵng" not in answer and "backend" not in answer


def test_noise_and_questions_are_not_saved(tmp_path: Path) -> None:
    _, advanced = make_agents(tmp_path)
    advanced.reply("dungct", "thread-1", "Mình tên gì nhỉ?")
    assert advanced.memory_file_size("dungct") == 0

    advanced.reply("dungct", "thread-1", "Mình đang làm MLOps engineer và ở Đà Nẵng.")
    advanced.reply("dungct", "thread-1", "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại.")
    advanced.reply("dungct", "thread-1", "Có lúc mình đùa là chuyển sang product manager, nhưng đó chỉ là câu đùa.")

    facts = advanced.profile_store.facts("dungct")
    assert facts["location"] == "Đà Nẵng"
    assert facts["profession"] == "MLOps engineer"


def test_confidence_threshold_skips_one_off_preferences(tmp_path: Path) -> None:
    one_off = "Bạn thử trả lời một câu ngắn thôi."
    lasting = "Mình muốn bạn luôn trả lời ngắn gọn thành 3 bullet."

    _, advanced = make_agents(tmp_path)
    advanced.reply("dungct", "thread-1", one_off)
    assert "style" not in advanced.profile_store.facts("dungct")
    advanced.reply("dungct", "thread-1", lasting)
    assert "3 bullet" in advanced.reply("dungct", "thread-2", "Nhắc lại style trả lời mình thích.")["response"]

    # Same input with the threshold disabled: the one-off instruction leaks into User.md.
    config = make_config(tmp_path / "no-threshold")
    config.profile_min_confidence = 0.0
    permissive = AdvancedAgent(config, force_offline=True)
    permissive.reply("dungct", "thread-1", one_off)
    assert "style" in permissive.profile_store.facts("dungct")
