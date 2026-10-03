from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Deterministic heuristic: ~4 characters per token after collapsing whitespace."""

    normalized = " ".join(unicodedata.normalize("NFC", text or "").split())
    if not normalized:
        return 0
    return math.ceil(len(normalized) / 4)


# Facts whose values are comma-separated lists: new items are merged instead of replacing the line.
LIST_FACT_KEYS = {"style", "interests"}
_FACT_LINE = re.compile(r"^- ([a-z_]+): (.*)$", re.MULTILINE)


def _merge_items(existing: str, incoming: str) -> str:
    """Union two comma lists, keeping order; a more specific item replaces a generic one ("bullet" -> "3 bullet")."""

    items: list[str] = []
    for item in [*existing.split(", "), *incoming.split(", ")]:
        item = item.strip()
        if not item or any(item.lower() in kept.lower() for kept in items):
            continue
        items = [kept for kept in items if kept.lower() not in item.lower()]
        items.append(item)
    return ", ".join(items)


def apply_profile_updates(facts: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """Return `facts` with `updates` applied: corrections replace, list facts (style, interests) merge."""

    merged = dict(facts)
    for key, value in updates.items():
        value = " ".join(value.split())
        if value:
            merged[key] = _merge_items(merged[key], value) if key in LIST_FACT_KEYS and key in merged else value
    return merged


@dataclass
class UserProfileStore:
    """Persistent memory: one `User.md` per user at `<root_dir>/<slug>/User.md`.

    Facts are stored as `- key: value` lines so a correction edits the old line in place
    instead of appending a contradicting one.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        ascii_id = unicodedata.normalize("NFKD", user_id or "").encode("ascii", "ignore").decode()
        slug = re.sub(r"[^a-z0-9_-]+", "_", ascii_id.lower()).strip("_-") or "anonymous"
        root = self.root_dir.resolve()
        path = (root / slug / "User.md").resolve()
        if root not in path.parents:  # defense in depth: never write outside root_dir
            raise ValueError(f"Unsafe user id: {user_id!r}")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Fixed "\n" newlines keep file_size() identical across operating systems.
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return {key: value.strip() for key, value in _FACT_LINE.findall(self.read_text(user_id))}

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or replace one fact; returns whether `User.md` changed."""

        current = self.facts(user_id)
        value = apply_profile_updates(current, {key: value}).get(key)
        if not value or current.get(key) == value:
            return False
        if key in current:
            return self.edit_text(user_id, f"- {key}: {current[key]}\n", f"- {key}: {value}\n")

        content = self.read_text(user_id) or f"# User profile: {user_id}\n\n## Facts\n"
        if not content.endswith("\n"):
            content += "\n"
        self.write_text(user_id, f"{content}- {key}: {value}\n")
        return True


# --- Fact extraction -------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"[,;:]|\s+nhưng\s+")

_QUESTION_MARKERS = (" là gì", " tên gì", " ở đâu", " nghề gì", " là ai", " như thế nào", " con gì")
_REQUEST_MARKERS = ("nhắc lại giúp", "hãy nhắc", "tóm tắt", "nhớ lại xem", "bạn có biết", "có thể nhắc lại")
# A recall request opens with "Nhắc lại ..."; "Nhắc lại lần cuối cho chắc: tên X" is the user restating facts.
_REQUEST_PREFIX = re.compile(r"^(?:hãy\s+|bạn\s+)?nhắc lại(?! lần cuối)", re.IGNORECASE)
# Whole sentence is not a stable fact: jokes, temporary instructions, references to outdated info.
_SKIP_SENTENCE_MARKERS = ("đùa", "tạm thời", "thông tin cũ", "ví dụ cũ")
# Clause is hypothetical / reported / dismissive ("Hà Nội chỉ là nơi đi họp", "Lúc đầu mình nói ở Huế").
_SKIP_CLAUSE_MARKERS = ("chỉ là", "lúc đầu", "ban đầu", "đừng")
_SKIP_CLAUSE_PREFIXES = ("nếu",)
# Everything after a negation is the old/false value ("ở Huế chứ không còn ở Đà Nẵng").
_NEGATION_CUT = re.compile(r"\s(?:chứ không|không còn|không phải|mặc dù|dù|thay vì)\s", re.IGNORECASE)

_NAME_PATTERNS = (
    re.compile(r"(?:mình|tôi|em)\s+tên\s+(?:là\s+)?", re.IGNORECASE),
    re.compile(r"tên\s+(?:của\s+)?(?:mình|tôi|em)\s+là\s+", re.IGNORECASE),
    re.compile(r"^tên\s+(?:là\s+)?", re.IGNORECASE),
)
_LOCATION_PATTERNS = (
    re.compile(r"nơi ở\b[^,.;]*?\b(?:là|sang)\s+", re.IGNORECASE),
    re.compile(r"(?:sống|chuyển\s+(?:về|đến|tới|ra|vào))\s+(?:ở\s+|tại\s+)?", re.IGNORECASE),
    re.compile(r"(?<!\w)ở\s+", re.IGNORECASE),
)
_PROFESSION_TRIGGER = re.compile(r"(?<!\w)(?:làm|là|sang|nghề)(?!\w)", re.IGNORECASE)
_ROLE = re.compile(
    r"\b([A-Za-z][\w-]*\s+(?:engineer|developer|scientist|manager|designer|analyst|architect|researcher))\b"
)
_STOP_AFTER_VALUE = r"(?=\s+(?:như|nhưng|mỗi|vào|lúc|và|vì|để)\b|[,.;!?]|$)"
_DRINK = re.compile(
    r"(?:đồ uống(?: yêu thích)?(?: của mình)?\s+(?:là|:)|mình\s+(?:vẫn\s+|hay\s+|thường\s+)?uống)\s+(.+?)"
    + _STOP_AFTER_VALUE,
    re.IGNORECASE,
)
_FOOD = re.compile(
    r"món(?: ăn)?\s+(?:yêu thích|ruột)(?: của mình)?\s+(?:là|:)\s+(.+?)" + _STOP_AFTER_VALUE, re.IGNORECASE
)
_PET = re.compile(r"nuôi\s+(?:một\s+|1\s+)?(?:bé\s+|con\s+|chú\s+)?(.+?)" + _STOP_AFTER_VALUE, re.IGNORECASE)

_STYLE_TRIGGER = re.compile(r"trả lời|giải thích|style|trình bày", re.IGNORECASE)
# (pattern, normalized descriptor); descriptors are expanded with the match so "3 bullet" keeps its number.
_STYLE_DESCRIPTORS = (
    (re.compile(r"(\d+)\s*bullet", re.IGNORECASE), r"\1 bullet"),
    (re.compile(r"\bbullet", re.IGNORECASE), "bullet"),
    (re.compile(r"\bngắn\b|\bgọn\b", re.IGNORECASE), "ngắn gọn"),
    (re.compile(r"rõ ý", re.IGNORECASE), "rõ ý"),
    (re.compile(r"có cấu trúc", re.IGNORECASE), "có cấu trúc"),
    (re.compile(r"ví dụ thực (tế|chiến)", re.IGNORECASE), r"có ví dụ thực \1"),
    (re.compile(r"số liệu", re.IGNORECASE), "có số liệu minh họa"),
    (re.compile(r"trade-off", re.IGNORECASE), "nhấn trade-off"),
)
# Confidence threshold (bonus): a fact is written to User.md only when fact_confidence() reaches this.
PROFILE_MIN_CONFIDENCE = 0.7
_PREFERENCE_CUE = re.compile(
    r"mình\s+(?:vẫn\s+|rất\s+|cũng\s+)?(?:muốn|thích)|\bhãy\b|ưu tiên|\bstyle\b|quan tâm|dài hạn|\bluôn\b|\bnhớ\b",
    re.IGNORECASE,
)
_HEDGE_CUE = re.compile(r"\bthử\b|có lẽ|chắc là|hình như|lần này|câu này|một câu", re.IGNORECASE)
_INTEREST_TRIGGER = re.compile(r"(?<!không )(?:thích|quan tâm|đam mê|đang học)", re.IGNORECASE)
_INTEREST_TERMS = re.compile(
    r"\bPython\b|\bAI (?:ứng dụng|agent)\b|\bAI\b|\bMLOps\b|\bRAG\b|\bevaluation\b|\bLangChain\b"
    r"|\bLangGraph\b|\bLLM\b|\bmemory architecture\b|\bbenchmark memory\b"
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def _is_question_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return (
        sentence.rstrip().endswith("?")
        or any(marker in lowered for marker in _QUESTION_MARKERS)
        or any(marker in lowered for marker in _REQUEST_MARKERS)
        or bool(_REQUEST_PREFIX.match(sentence))
    )


def is_question(message: str) -> bool:
    """True when every sentence is a question or recall request, i.e. the turn carries no new fact."""

    sentences = _sentences(message)
    return bool(sentences) and all(_is_question_sentence(sentence) for sentence in sentences)


def _capitalized_run(text: str, max_words: int = 4) -> str | None:
    """Leading run of Capitalized words (a proper noun such as "Đà Nẵng" or "DũngCT Stress")."""

    words: list[str] = []
    for raw in text.split()[:max_words]:
        word = raw.strip(".,;:!?()\"'“”")
        if not word or not word[0].isupper():
            break
        words.append(word)
        if word != raw:  # punctuation ends the noun phrase
            break
    return " ".join(words) or None


def _fact_clauses(sentence: str) -> list[str]:
    clauses = []
    for clause in _CLAUSE_SPLIT.split(sentence):
        clause = clause.strip()
        lowered = clause.lower()
        if not clause or lowered.startswith(_SKIP_CLAUSE_PREFIXES) or any(m in lowered for m in _SKIP_CLAUSE_MARKERS):
            continue
        # Padded so a clause that *starts* with a negation ("không còn ở Huế") is dropped entirely.
        clause = _NEGATION_CUT.split(f" {clause} ", maxsplit=1)[0].strip()
        if clause:
            clauses.append(clause)
    return clauses


def _match_proper_noun(patterns: tuple[re.Pattern[str], ...], clause: str) -> str | None:
    for pattern in patterns:
        for match in pattern.finditer(clause):
            value = _capitalized_run(clause[match.end():])
            if value:
                return value
    return None


def fact_confidence(key: str, sentence: str) -> float:
    """How sure we are that `sentence` states a *lasting* fact for `key` (bonus: confidence threshold).

    Identity facts come from explicit self-statements, so they start high (0.8). Preferences and
    interests are cheap to mention in passing, so they start at 0.5 and need a preference cue
    ("mình muốn", "hãy", "dài hạn", ...) to pass. Hedged or one-off wording ("thử", "một câu",
    "lần này") lowers the score for every key.
    """

    score = 0.5 if key in LIST_FACT_KEYS else 0.8
    if key in LIST_FACT_KEYS and _PREFERENCE_CUE.search(sentence):
        score += 0.3
    if _HEDGE_CUE.search(sentence):
        score -= 0.4
    return round(min(max(score, 0.0), 1.0), 2)


def extract_profile_updates(message: str, min_confidence: float = PROFILE_MIN_CONFIDENCE) -> dict[str, str]:
    """Turn one user message into stable profile facts.

    Keys: name, location, profession, style, interests, drink, food, pet. Question-only turns,
    jokes, conditionals, temporary instructions, and negated (old) values yield nothing; facts
    scoring below `min_confidence` (see `fact_confidence`) are dropped.
    """

    updates: dict[str, str] = {}
    if is_question(message):
        return updates

    for sentence in _sentences(message):
        lowered = sentence.lower()
        if _is_question_sentence(sentence) or any(marker in lowered for marker in _SKIP_SENTENCE_MARKERS):
            continue

        found: dict[str, str] = {}
        if _STYLE_TRIGGER.search(sentence):
            for pattern, template in _STYLE_DESCRIPTORS:
                match = pattern.search(sentence)
                if match:
                    found["style"] = _merge_items(found.get("style", ""), match.expand(template))
        if _INTEREST_TRIGGER.search(sentence):
            for match in _INTEREST_TERMS.finditer(sentence):
                found["interests"] = _merge_items(found.get("interests", ""), match.group(0))

        for clause in _fact_clauses(sentence):
            name = _match_proper_noun(_NAME_PATTERNS, clause)
            if name:
                found["name"] = name
            location = _match_proper_noun(_LOCATION_PATTERNS, clause)
            if location:
                found["location"] = location
            if _PROFESSION_TRIGGER.search(clause):
                roles = _ROLE.findall(clause)
                if roles:
                    found["profession"] = roles[-1]
            for key, pattern in (("drink", _DRINK), ("food", _FOOD), ("pet", _PET)):
                match = pattern.search(clause)
                if match:
                    found[key] = match.group(1).strip()

        confident = {key: value for key, value in found.items() if fact_confidence(key, sentence) >= min_confidence}
        updates = apply_profile_updates(updates, confident)

    return updates


# --- Compact memory --------------------------------------------------------------------------

_TERMS_PREFIX = "Chủ đề đã nhắc: "
_MAX_SUMMARY_TERMS = 20
_GIST_CHARS = 110


def _key_terms(text: str) -> list[str]:
    """Proper nouns and codes (NASA, Artemis III, X-59, Đà Nẵng) in order of appearance."""

    terms: list[str] = []
    for sentence in _sentences(text):
        run: list[str] = []
        words = sentence.split()
        for index, raw in enumerate(words):
            token = raw.strip(".,;:!?()\"'“”")
            # A capitalized first word is only a name when the next word is capitalized too ("Artemis III").
            starts_name = index + 1 < len(words) and words[index + 1][:1].isupper()
            is_term = bool(token) and (
                (token[0].isupper() and (index > 0 or starts_name))
                or any(ch.isupper() for ch in token[1:])
                or (any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token))
            )
            if is_term:
                run.append(token)
            if run and (not is_term or token != raw):
                terms.append(" ".join(run))
                run = []
        if run:
            terms.append(" ".join(run))
    return terms


def _gist(text: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return ""
    best = max(sentences, key=lambda s: (len(_key_terms(s)), -sentences.index(s)))
    if len(best) <= _GIST_CHARS:
        return best
    return best[:_GIST_CHARS].rsplit(" ", 1)[0] + "…"


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: a bounded list of key terms + the gist of the most recent user turns.

    A message with role "summary" (the previous summary) is folded in, so repeated compactions
    keep the summary bounded instead of nesting summaries. Assistant turns are skipped: in
    offline mode they only echo memory that already lives in `User.md` or the summary.
    """

    terms: list[str] = []
    points: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if message.get("role") == "summary":
            for line in content.splitlines():
                if line.startswith(_TERMS_PREFIX):
                    terms.extend(t for t in line[len(_TERMS_PREFIX):].split(", ") if t)
                elif line.startswith("- "):
                    points.append(line[2:])
            continue
        if message.get("role") != "user":
            continue
        terms.extend(_key_terms(content))
        gist = _gist(content)
        if gist:
            points.append(f"user: {gist}")

    unique_terms = list(dict.fromkeys(terms))[:_MAX_SUMMARY_TERMS]
    lines = [_TERMS_PREFIX + ", ".join(unique_terms)] if unique_terms else []
    lines.extend(f"- {point}" for point in points[-max_items:])
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory: when a thread exceeds `threshold_tokens`, everything except the last
    `keep_messages` messages is folded into a summary, and the compaction is counted."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        thread["messages"].append({"role": role, "content": content})
        if self.token_count(thread_id) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self._compact(thread)

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})
        return {
            "messages": list(thread["messages"]),
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))

    def token_count(self, thread_id: str) -> int:
        """Tokens this thread would carry into the prompt: summary + kept messages."""

        thread = self.state.get(thread_id)
        if not thread:
            return 0
        return estimate_tokens(thread["summary"]) + sum(estimate_tokens(m["content"]) for m in thread["messages"])

    def _compact(self, thread: dict[str, object]) -> None:
        messages = thread["messages"]
        cut = len(messages) - self.keep_messages
        older, recent = messages[:cut], messages[cut:]
        previous = [{"role": "summary", "content": thread["summary"]}] if thread["summary"] else []
        thread["summary"] = summarize_messages(previous + older)
        thread["messages"] = recent
        thread["compactions"] += 1
