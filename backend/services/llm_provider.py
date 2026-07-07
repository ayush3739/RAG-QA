"""
backend/services/llm_provider.py

Single LLM abstraction for DocuMind.

Active provider is set in backend/core/config.py:
    settings.llm_provider = "groq"
    settings.llm_model   = "llama-3.3-70b-versatile"

On a 429 / rate-limit from the primary provider, invoke() automatically
falls through this chain until one succeeds:

    Gemini (gemini-2.0-flash)
        → Groq  (llama-3.3-70b-versatile)
        → OpenRouter (google/gemini-flash-1.5)
        → GitHub AI  (gpt-4o-mini)
        → Local Ollama (ollama_model)

All providers use OpenAI-compatible REST except Gemini which uses the
google-genai SDK.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from backend.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_client(api_key: str, base_url: str):
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "too many requests" in s or "quota" in s


# ---------------------------------------------------------------------------
# Fallback chain definition
# Each entry: (label, callable that returns the answer string or raises)
# Built lazily so we don't fail at import time if a key is missing.
# ---------------------------------------------------------------------------

async def _try_gemini(messages: list[dict]) -> str:
    """Call Gemini via google-genai SDK (OpenAI-compat v1beta endpoint)."""
    key = settings.gemini_api_key
    if not key:
        raise RuntimeError("No Gemini API key configured.")
    client = _openai_client(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    resp = await client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_groq(messages: list[dict]) -> str:
    key = settings.groq_api_key
    if not key:
        raise RuntimeError("No Groq API key configured.")
    client = _openai_client(api_key=key, base_url="https://api.groq.com/openai/v1")
    resp = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_openrouter(messages: list[dict]) -> str:
    key = settings.open_router_key
    if not key:
        raise RuntimeError("No OpenRouter API key configured.")
    client = _openai_client(api_key=key, base_url="https://openrouter.ai/api/v1")
    resp = await client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_github(messages: list[dict]) -> str:
    key = settings.github_token
    if not key:
        raise RuntimeError("No GitHub token configured.")
    client = _openai_client(api_key=key, base_url="https://models.github.ai/inference")
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_ollama(messages: list[dict]) -> str:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=settings.ollama_model, temperature=0)
    result = await llm.ainvoke(messages)
    return result.content


# Ordered fallback chain: label → async callable
_FALLBACK_CHAIN = [
    ("Gemini",     _try_gemini),
    ("Groq",       _try_groq),
    ("OpenRouter", _try_openrouter),
    ("GitHub AI",  _try_github),
    ("Ollama",     _try_ollama),
]

# Map provider name → its position so the primary always starts at index 0
_PROVIDER_INDEX = {
    "gemini":     0,
    "groq":       1,
    "openrouter": 2,
    "github":     3,
    "ollama":     4,
}


# ---------------------------------------------------------------------------
# LLMProvider
# ---------------------------------------------------------------------------

class LLMProvider:
    """
    Thin abstraction over multiple LLM back-ends with automatic fallback.

    Usage
    -----
    llm = LLMProvider()
    response: str = await llm.invoke(messages)
    async for token in llm.stream(messages): yield token
    """

    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower()
        self._start_idx = _PROVIDER_INDEX.get(self.provider, 1)

        # Build a primary client for streaming (stream() only uses primary)
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama
            self._ollama = ChatOllama(model=settings.llm_model, temperature=0)
            self._client = None
        else:
            endpoints = {
                "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", settings.gemini_api_key),
                "groq":       ("https://api.groq.com/openai/v1",                          settings.groq_api_key),
                "openrouter": ("https://openrouter.ai/api/v1",                            settings.open_router_key),
                "github":     ("https://models.github.ai/inference",                      settings.github_token),
            }
            url, key = endpoints.get(self.provider, (None, None))
            self._client = _openai_client(api_key=key, base_url=url) if key and url else None
            self._ollama = None

    # ------------------------------------------------------------------
    # invoke  –  with automatic fallback on 429
    # ------------------------------------------------------------------

    async def invoke(self, messages: list[dict]) -> str:
        last_exc: Exception | None = None

        for label, fn in _FALLBACK_CHAIN[self._start_idx:] + _FALLBACK_CHAIN[:self._start_idx]:
            try:
                result = await fn(messages)
                if label != _FALLBACK_CHAIN[self._start_idx][0]:
                    logger.warning("[LLMProvider] Used fallback: %s", label)
                return result
            except Exception as exc:
                if _is_rate_limit(exc):
                    logger.warning(
                        "[LLMProvider] %s rate-limited, trying next fallback.", label
                    )
                    last_exc = exc
                    continue
                # Non-rate-limit error on primary → still try fallbacks
                logger.warning("[LLMProvider] %s failed (%s), trying next.", label, exc)
                last_exc = exc
                continue

        raise RuntimeError(
            f"All LLM providers exhausted. Last error: {last_exc}"
        )

    def supports_native_tool_calls(self) -> bool:
        return self.provider in {"groq", "github", "openrouter"}

    async def tool_call(self, messages: list[dict], tools: list[dict]) -> dict:
        if not self.supports_native_tool_calls() or self._client is None:
            return {"content": await self.invoke(messages), "tool_calls": []}

        response = await self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )
        message = response.choices[0].message
        tool_calls = []

        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": call.id, "name": call.function.name, "args": args})

        return {"content": message.content or "", "tool_calls": tool_calls}

    # ------------------------------------------------------------------
    # stream  –  primary only (no fallback for streaming)
    # ------------------------------------------------------------------

    async def stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        if self.provider == "ollama":
            async for chunk in self._ollama.astream(messages):
                yield chunk.content
            return

        if self._client is None:
            yield await self.invoke(messages)
            return

        response = await self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            yield chunk.choices[0].delta.content or ""
