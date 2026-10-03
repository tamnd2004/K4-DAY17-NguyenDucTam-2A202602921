from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from memory_store import PROFILE_MIN_CONFIDENCE
from model_provider import ProviderConfig, normalize_provider

# Benchmark defaults. A standard conversation (~10 short turns) stays far below the
# threshold, while the 16-turn stress thread (~150 tokens per user turn) crosses it
# every few turns, so compaction fires several times there and never on short threads.
DEFAULT_COMPACT_THRESHOLD_TOKENS = 1000
DEFAULT_COMPACT_KEEP_MESSAGES = 4

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}

# Env var holding the API key / base URL for each provider.
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}
BASE_URL_ENV = {
    "custom": "CUSTOM_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}


REPO_ROOT = Path(__file__).resolve().parent.parent


def _offline_provider() -> ProviderConfig:
    # No API key -> agents stay on the deterministic offline path.
    return ProviderConfig(provider="openai", model_name=DEFAULT_MODELS["openai"], temperature=0.0)


@dataclass
class LabConfig:
    """Shared configuration: the contract between agents, memory store, and benchmark."""

    base_dir: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    state_dir: Path = REPO_ROOT / "state"
    compact_threshold_tokens: int = DEFAULT_COMPACT_THRESHOLD_TOKENS
    compact_keep_messages: int = DEFAULT_COMPACT_KEEP_MESSAGES
    model: ProviderConfig = field(default_factory=_offline_provider)
    judge_model: ProviderConfig = field(default_factory=_offline_provider)
    # Bonus knob: facts below this confidence are not written to User.md (0 disables the filter).
    profile_min_confidence: float = PROFILE_MIN_CONFIDENCE


def _env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _provider_from_env(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Read `<prefix>_PROVIDER`, `<prefix>_MODEL`, ... ; unset judge knobs inherit from the main model."""

    provider = normalize_provider(_env(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai"))
    same_provider = fallback is not None and fallback.provider == provider

    model_name = _env(f"{prefix}_MODEL") or (fallback.model_name if same_provider else DEFAULT_MODELS[provider])
    temperature = _env(f"{prefix}_TEMPERATURE") or (str(fallback.temperature) if same_provider else "0")
    rpm = _env(f"{prefix}_REQUESTS_PER_MINUTE") or (
        str(fallback.requests_per_minute) if same_provider and fallback.requests_per_minute else None
    )

    key_env = API_KEY_ENV[provider]
    api_key = _env(key_env) if key_env else None
    if provider == "gemini" and not api_key:
        api_key = _env("GOOGLE_API_KEY")
    base_url = _env(BASE_URL_ENV[provider]) if provider in BASE_URL_ENV else None

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(temperature),
        api_key=api_key,
        base_url=base_url,
        requests_per_minute=float(rpm) if rpm else None,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` and return a fully populated LabConfig; creates `state/` if missing.

    Env convention (all optional — offline mode needs none of them):
    - LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_REQUESTS_PER_MINUTE
    - JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_REQUESTS_PER_MINUTE (default: same as LLM_*)
    - OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), ANTHROPIC_API_KEY, OPENROUTER_API_KEY
    - CUSTOM_BASE_URL + CUSTOM_API_KEY, OLLAMA_BASE_URL, OPENROUTER_BASE_URL
    - COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES, PROFILE_MIN_CONFIDENCE
    """

    root = (base_dir or REPO_ROOT).resolve()

    try:
        from dotenv import load_dotenv
    except ImportError:  # offline mode must still work without python-dotenv
        pass
    else:
        load_dotenv(root / ".env", override=False)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    threshold = int(_env("COMPACT_THRESHOLD_TOKENS") or DEFAULT_COMPACT_THRESHOLD_TOKENS)
    keep = int(_env("COMPACT_KEEP_MESSAGES") or DEFAULT_COMPACT_KEEP_MESSAGES)
    if threshold <= 0 or keep <= 0:
        raise ValueError("COMPACT_THRESHOLD_TOKENS and COMPACT_KEEP_MESSAGES must be positive integers.")

    model = _provider_from_env("LLM")
    judge_model = _provider_from_env("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        model=model,
        judge_model=judge_model,
        profile_min_confidence=float(_env("PROFILE_MIN_CONFIDENCE") or PROFILE_MIN_CONFIDENCE),
    )
