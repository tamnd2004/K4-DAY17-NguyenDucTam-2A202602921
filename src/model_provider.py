from __future__ import annotations

import re
from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_PROVIDER_ALIASES = {
    "openai": "openai",
    "open-ai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "gemeni": "gemini",
    "gemnini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google-gemini": "gemini",
    "googleai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "anthrophic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "olama": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "openroute": "openrouter",
}

# One limiter per (provider, model, rpm): the main model and the judge share quota when they hit the same model.
_RATE_LIMITERS: dict[tuple[str, str, float], object] = {}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None
    requests_per_minute: float | None = None


def normalize_provider(value: str) -> str:
    """Map aliases and common typos (e.g. `anthorpic`) to one of SUPPORTED_PROVIDERS."""

    key = (value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if key in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[key]
    raise ValueError(
        f"Unsupported LLM provider {value!r}. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def has_live_credentials(config: ProviderConfig) -> bool:
    """Whether a real model can be built; agents stay on the deterministic offline path otherwise.

    Ollama and custom endpoints opt in through their base URL; hosted providers need an API key.
    """

    if normalize_provider(config.provider) in ("ollama", "custom"):
        return bool(config.base_url)
    return bool(config.api_key)


def _rate_limiter(config: ProviderConfig):
    if not config.requests_per_minute:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    key = (config.provider, config.model_name, config.requests_per_minute)
    if key not in _RATE_LIMITERS:
        _RATE_LIMITERS[key] = InMemoryRateLimiter(
            requests_per_second=config.requests_per_minute / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return _RATE_LIMITERS[key]


def build_chat_model(config: ProviderConfig):
    """Instantiate the real LangChain chat model for the selected provider.

    Imports are lazy so a missing SDK for an unused provider never breaks offline mode.
    """

    provider = normalize_provider(config.provider)
    common = {"temperature": config.temperature, "rate_limiter": _rate_limiter(config)}

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        if provider == "custom" and not config.base_url:
            raise ValueError("Provider 'custom' requires a base_url (CUSTOM_BASE_URL).")
        return ChatOpenAI(model=config.model_name, api_key=config.api_key, base_url=config.base_url, **common)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Gemini 3+ runs on Google's sampling defaults (3.5 Flash-Lite ignores temperature
        # and warns on every call), so only legacy 1.x/2.x models receive it.
        if not re.match(r"(models/)?gemini-[12]\.", config.model_name.lower()):
            common.pop("temperature")
        return ChatGoogleGenerativeAI(model=config.model_name, google_api_key=config.api_key, **common)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model_name, api_key=config.api_key, **common)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.model_name, base_url=config.base_url, **common)

    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(model=config.model_name, api_key=config.api_key, base_url=config.base_url, **common)
