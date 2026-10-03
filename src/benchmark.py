from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, estimate_tokens
from model_provider import build_chat_model, has_live_credentials

SUITES = (
    ("Standard Benchmark", "conversations.json"),
    ("Long-Context Stress Benchmark", "advanced_long_context.json"),
)
COLUMNS = (
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
)
JUDGE_PROMPT = (
    "Bạn là giám khảo. Chấm câu trả lời của trợ lý cho một câu hỏi kiểm tra trí nhớ, thang 0-10: "
    "nêu đúng và đủ các fact mong đợi, không bịa fact cũ hoặc sai, ngắn gọn. Chỉ trả về một con số.\n"
    "Câu hỏi: {question}\nFact mong đợi: {expected}\nCâu trả lời: {answer}"
)


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    conversations = json.loads(Path(path).read_text(encoding="utf-8"))
    for conversation in conversations:
        missing = {"id", "user_id", "turns", "recall_questions"} - conversation.keys()
        if missing:
            raise ValueError(f"{path}: conversation {conversation.get('id')!r} is missing {sorted(missing)}")
    return conversations


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _found(answer: str, expected: list[str]) -> int:
    normalized = _normalize(answer)
    return sum(_normalize(item) in normalized for item in expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 when every expected fact appears, 0.5 when only some do, 0 when none do."""

    found = _found(answer, expected)
    if expected and found == len(expected):
        return 1.0
    return 0.5 if found else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality in [0, 1], same formula for every agent.

    70% fact coverage (finer than the 3-level recall), 15% brevity (the users ask for short
    answers), 15% bullet structure. An answer with none of the expected facts scores 0: being
    short and well formatted does not make a non-answer useful.
    """

    if not expected:
        return 0.0
    coverage = _found(answer, expected) / len(expected)
    if coverage == 0:
        return 0.0
    tokens = estimate_tokens(answer)
    brevity = 1.0 if tokens <= 60 else max(0.0, 1 - (tokens - 60) / 120)
    structure = 1.0 if re.search(r"^\s*[-*•]\s", answer, re.MULTILINE) else 0.5
    return round(0.7 * coverage + 0.15 * brevity + 0.15 * structure, 3)


def judge_quality(judge, question: str, answer: str, expected: list[str]) -> float:
    """LLM-as-judge score in [0, 1]; falls back to the heuristic if the reply has no number."""

    reply = judge.invoke(JUDGE_PROMPT.format(question=question, expected=", ".join(expected), answer=answer))
    match = re.search(r"\d+(?:[.,]\d+)?", reply.text)
    if not match:
        return heuristic_quality(answer, expected)
    return round(min(max(float(match.group(0).replace(",", ".")) / 10, 0.0), 1.0), 3)


def _memory_bytes(agent, user_ids: set[str]) -> int:
    size_of = getattr(agent, "memory_file_size", None)  # the baseline has no persistent memory
    return sum(size_of(user_id) for user_id in user_ids) if size_of else 0


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config, judge=None, details: list | None = None
) -> BenchmarkRow:
    """Feed every turn in order, then ask each conversation's recall questions in a fresh thread.

    Token columns cover the conversation threads; recall threads are only used for scoring.
    """

    user_ids = {conversation["user_id"] for conversation in conversations}
    memory_before = _memory_bytes(agent, user_ids)
    agent_tokens = prompt_tokens = compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id, thread_id = conversation["user_id"], conversation["id"]
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, turn)
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        recall_thread = f"{thread_id}::recall"  # same new-thread naming for every agent
        for item in conversation["recall_questions"]:
            answer = agent.reply(user_id, recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(
                judge_quality(judge, item["question"], answer, expected) if judge else heuristic_quality(answer, expected)
            )
            if details is not None:
                details.append((agent_name, item["question"], answer, recall_scores[-1]))

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
        memory_growth_bytes=_memory_bytes(agent, user_ids) - memory_before,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(cell) for cell in line) + " |" for line in table]
        return "\n".join(lines)
    return tabulate(table, headers=COLUMNS, tablefmt="github", disable_numparse=True)


def _ratio_line(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    def ratio(new: int, old: int) -> str:
        return f"{new / old:.2f}x" if old else "n/a"

    return (
        f"Advanced / Baseline: prompt tokens {ratio(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
        f"agent tokens {ratio(advanced.agent_tokens_only, baseline.agent_tokens_only)}"
    )


def main() -> None:
    """Run the Standard and Long-Context Stress suites for Baseline vs Advanced on identical input."""

    parser = argparse.ArgumentParser(description="Day 17 memory benchmark: Baseline vs Advanced.")
    parser.add_argument("--live", action="store_true", help="use the real LLM from .env (default: offline, reproducible)")
    parser.add_argument("--judge", action="store_true", help="score Response quality with the judge model (needs a key)")
    parser.add_argument("--details", action="store_true", help="print every recall question and answer")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(Path(__file__).resolve().parent.parent)
    judge = None
    if args.judge:
        if not has_live_credentials(config.judge_model):
            parser.error("--judge needs credentials for the judge model (see .env.example)")
        judge = build_chat_model(config.judge_model)

    mode = f"live ({config.model.provider}/{config.model.model_name})" if args.live else "offline (deterministic)"
    print(f"Mode: {mode} | quality: {'LLM judge' if judge else 'heuristic'} | "
          f"compact threshold {config.compact_threshold_tokens} tokens, keep {config.compact_keep_messages} messages | "
          f"profile min confidence {config.profile_min_confidence}")

    for title, filename in SUITES:
        conversations = load_conversations(config.data_dir / filename)
        # Fresh User.md per run, so Memory growth and recall never depend on a previous run.
        store = UserProfileStore(config.state_dir / "profiles")
        for user_id in {conversation["user_id"] for conversation in conversations}:
            store.path_for(user_id).unlink(missing_ok=True)

        details: list | None = [] if args.details else None
        rows = [
            run_agent_benchmark(name, agent_cls(config, force_offline=not args.live), conversations, config, judge, details)
            for name, agent_cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent))
        ]
        turns = sum(len(conversation["turns"]) for conversation in conversations)
        questions = sum(len(conversation["recall_questions"]) for conversation in conversations)
        print(f"\n## {title} ({len(conversations)} conversations, {turns} turns, {questions} recall questions)\n")
        print(format_rows(rows))
        print(f"\n{_ratio_line(*rows)}")
        for agent_name, question, answer, score in details or []:
            print(f"\n[{agent_name}] recall={score} | {question}\n{answer}")


if __name__ == "__main__":
    main()
