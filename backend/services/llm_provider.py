"""
backend/services/llm_provider.py

Multi-Provider LLM abstraction with Task-Specific Routing Chains,
In-Memory 429 Cooldown Tracking, and Real-Time Telemetry.

Task-Specific Fallback Chains:
1. ROUTER_CHAIN: Ultra-low latency 8B/3B models (<100ms) for JSON routing.
2. RAG_AND_RESEARCH_CHAIN: 100k-1M context + citation grounding (Gemini, Cohere, Gonka MiniMax).
3. GENERAL_CHAT_CHAIN: Full conversational cascade (Groq 70B, Gemini, Gonka, Cohere, Cloudflare, Kilo).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import AsyncGenerator

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing Configuration Loader
# ---------------------------------------------------------------------------

_CORE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "routing_config.json"
_ROOT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "routing_config.json"
_CONFIG_PATH = _CORE_CONFIG_PATH if _CORE_CONFIG_PATH.exists() else _ROOT_CONFIG_PATH


def load_routing_config() -> dict:
    """Load configuration from routing_config.json with graceful fallback."""
    config_file = _CORE_CONFIG_PATH if _CORE_CONFIG_PATH.exists() else _ROOT_CONFIG_PATH
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[LLMProvider] Failed to parse routing_config.json: %s", e)
    return {}


_ROUTING_CONFIG = load_routing_config()


def _get_model_slug(provider_id: str, default_slug: str) -> str:
    """Return configured model slug from routing_config.json or default."""
    return _ROUTING_CONFIG.get("models", {}).get(provider_id, {}).get("model", default_slug)


# ---------------------------------------------------------------------------
# In-Memory 429 Cooldown Tracking
# ---------------------------------------------------------------------------

_COOLDOWNS: dict[str, float] = {}


def is_healthy(provider_name: str) -> bool:
    """Return True if the provider is not currently in a rate-limit cooldown window."""
    return time.time() >= _COOLDOWNS.get(provider_name, 0.0)


def mark_rate_limited(provider_name: str, cooldown_seconds: float = 60.0) -> None:
    """Mark a provider in cooldown to skip network calls during rate-limit bursts in 0ms."""
    _COOLDOWNS[provider_name] = time.time() + cooldown_seconds
    expiry = time.strftime("%H:%M:%S", time.localtime(_COOLDOWNS[provider_name]))
    logger.warning(
        "[LLMProvider] %s entered 429 cooldown for %.1fs (will skip in 0ms until %s)",
        provider_name,
        cooldown_seconds,
        expiry,
    )


def get_cooldown_status() -> dict[str, float]:
    """Return remaining cooldown seconds for active rate-limited providers."""
    now = time.time()
    return {k: round(v - now, 1) for k, v in _COOLDOWNS.items() if v > now}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_client(api_key: str, base_url: str):
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "too many requests" in s or "quota" in s or "504" in s or "resource_exhausted" in s


# ---------------------------------------------------------------------------
# Specialized Provider Implementations
# ---------------------------------------------------------------------------

async def _try_groq_20b(messages: list[dict]) -> str:
    """Groq GPT-OSS 20B: Ultra-fast (1,000 t/s, 250k TPM, 1k RPM) for JSON routing."""
    key = settings.groq_api_key or settings.groq_api_third
    if not key:
        raise RuntimeError("No Groq API key configured.")
    client = _openai_client(api_key=key, base_url="https://api.groq.com/openai/v1")
    model = _get_model_slug("groq_20b", "openai/gpt-oss-20b")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_groq_120b(messages: list[dict]) -> str:
    """Groq GPT-OSS 120B: High-fidelity reasoning (500 t/s, 250k TPM, 1k RPM, 131k context)."""
    key = settings.groq_api_key
    if not key:
        raise RuntimeError("No primary Groq API key configured.")
    client = _openai_client(api_key=key, base_url="https://api.groq.com/openai/v1")
    model = _get_model_slug("groq_120b", "openai/gpt-oss-120b")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_groq_qwen27b(messages: list[dict]) -> str:
    """Groq Qwen 27B: Fast deep reasoning (450 t/s, 250k TPM, 1k RPM, 131k context)."""
    key = settings.groq_api_key or settings.groq_api_secondary
    if not key:
        raise RuntimeError("No Groq API key configured.")
    client = _openai_client(api_key=key, base_url="https://api.groq.com/openai/v1")
    model = _get_model_slug("groq_qwen27b", "qwen/qwen3.8-27b")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_groq_secondary(messages: list[dict]) -> str:
    """Groq Secondary key fallback (uses GPT-OSS 120B)."""
    key = settings.groq_api_secondary
    if not key:
        raise RuntimeError("No Groq Secondary API key configured.")
    client = _openai_client(api_key=key, base_url="https://api.groq.com/openai/v1")
    model = _get_model_slug("groq_120b", "openai/gpt-oss-120b")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_groq_third(messages: list[dict]) -> str:
    """Groq Third key fallback (uses GPT-OSS 120B)."""
    key = settings.groq_api_third
    if not key:
        raise RuntimeError("No Groq Third API key configured.")
    client = _openai_client(api_key=key, base_url="https://api.groq.com/openai/v1")
    model = _get_model_slug("groq_120b", "openai/gpt-oss-120b")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


# Backward-compatibility aliases
_try_groq_8b = _try_groq_20b
_try_groq_70b = _try_groq_120b


async def _try_gemini(messages: list[dict]) -> str:
    """Google Gemini: 1M token context window, 1,000,000 TPM limit for RAG & burst absorbing."""
    key = settings.gemini_api_key
    if not key:
        raise RuntimeError("No Gemini API key configured.")
    import httpx
    gemini_contents = []
    for m in messages:
        role = "user" if m.get("role") in ("user", "system") else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": m.get("content") or ""}],
        })
    model = _get_model_slug("gemini_flash", "gemini-flash-lite-latest")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json={"contents": gemini_contents})
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def _try_cohere(messages: list[dict]) -> str:
    """Cohere Command R+: Purpose-built for RAG citation grounding and factuality."""
    key = settings.cohere_api_key
    if not key:
        raise RuntimeError("No Cohere API key configured.")
    import httpx
    url = "https://api.cohere.com/v2/chat"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    cohere_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "assistant", "system"):
            cohere_messages.append({"role": role, "content": content})
        else:
            cohere_messages.append({"role": "user", "content": content})

    model = _get_model_slug("cohere_r", "command-r-08-2024")
    payload = {
        "model": model,
        "messages": cohere_messages,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Cohere API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["message"]["content"][0]["text"]


async def _try_cloudflare(messages: list[dict]) -> str:
    """Cloudflare Workers AI: Low-latency edge inference with 10k free Neurons/day."""
    token = settings.cloudflare_ai_api or settings.cloudflare_api_token
    account_id = settings.cloudflare_account_id
    if not token:
        raise RuntimeError("No Cloudflare API token configured.")
    if not account_id:
        raise RuntimeError("No Cloudflare Account ID configured.")
    client = _openai_client(
        api_key=token,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
    )
    model = _get_model_slug("cloudflare_3b", "@cf/meta/llama-3.2-3b-instruct")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_gonka(messages: list[dict]) -> str:
    """Gonka Broker: MiniMax-M2.7 (200k context window) for heavy documents."""
    key = settings.gonka_api_key
    if not key:
        raise RuntimeError("No Gonka Broker API key configured.")
    client = _openai_client(
        api_key=key,
        base_url="https://proxy.gonkabroker.com/v1",
    )
    model = _get_model_slug("gonka_minimax", "MiniMaxAI/MiniMax-M2.7")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_gonka_deepseek(messages: list[dict]) -> str:
    """Gonka Broker: DeepSeek-V4-Flash-0731 (400k context window, FP8) for complex reasoning & research."""
    key = settings.gonka_api_key
    if not key:
        raise RuntimeError("No Gonka Broker API key configured.")
    client = _openai_client(
        api_key=key,
        base_url="https://proxy.gonkabroker.com/v1",
    )
    model = _get_model_slug("gonka_deepseek", "deepseek-ai/DeepSeek-V4-Flash-0731")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_gonka_glm(messages: list[dict]) -> str:
    """Gonka Broker: GLM-5.3-Flash (400k context window, FP8) for fast multi-lingual synthesis."""
    key = settings.gonka_api_key
    if not key:
        raise RuntimeError("No Gonka Broker API key configured.")
    client = _openai_client(
        api_key=key,
        base_url="https://proxy.gonkabroker.com/v1",
    )
    model = _get_model_slug("gonka_glm", "zai-org/GLM-5.3-Flash")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def get_gonka_bge_embeddings(texts: list[str]) -> list[list[float]]:
    """Gonka Broker: BAAI/bge-m3 dense/sparse embedding generator (8k context window)."""
    key = settings.gonka_api_key
    if not key:
        raise RuntimeError("No Gonka Broker API key configured.")
    client = _openai_client(
        api_key=key,
        base_url="https://proxy.gonkabroker.com/v1",
    )
    resp = await client.embeddings.create(
        model="BAAI/bge-m3",
        input=texts,
    )
    return [item.embedding for item in resp.data]


async def _try_kilo(messages: list[dict]) -> str:
    """Kilo Code AI Gateway: Dynamic autonomous free model routing."""
    key = settings.kilo_api_key
    if not key:
        raise RuntimeError("No Kilo API key configured.")
    client = _openai_client(
        api_key=key,
        base_url="https://ai-gateway.kilo.ai/api/v1",
    )
    model = _get_model_slug("kilo_free", "kilo-auto/free")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    msg = resp.choices[0].message
    return msg.content or getattr(msg, "reasoning", "") or ""


async def _try_nvidia(messages: list[dict]) -> str:
    """NVIDIA NIM: Enterprise reasoning model endpoint."""
    key = settings.nvidia_nim_api_key or settings.nvidia_nim
    if not key:
        raise RuntimeError("No Nvidia NIM API key configured.")
    client = _openai_client(
        api_key=key,
        base_url="https://integrate.api.nvidia.com/v1",
    )
    model = _get_model_slug("nvidia_nim", "mistralai/mistral-7b-instruct-v0.3")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_openrouter(messages: list[dict]) -> str:
    """OpenRouter: Free-tier multi-model aggregator."""
    key = settings.open_router_key
    if not key:
        raise RuntimeError("No OpenRouter API key configured.")
    client = _openai_client(api_key=key, base_url="https://openrouter.ai/api/v1")
    model = _get_model_slug("openrouter", "liquid/lfm-3-1.7b:free")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    msg = resp.choices[0].message
    return msg.content or getattr(msg, "reasoning", "") or ""


async def _try_ollama(messages: list[dict]) -> str:
    """Local Ollama: 100% offline, zero-cost, unmetered fallback."""
    from langchain_ollama import ChatOllama
    model = _get_model_slug("ollama", settings.ollama_model)
    llm = ChatOllama(model=model, temperature=0)
    result = await llm.ainvoke(messages)
    return result.content


# ---------------------------------------------------------------------------
# Task-Specific Fallback Chains & Dynamic Configuration
# ---------------------------------------------------------------------------

_DISPATCH_REGISTRY = {
    "groq_20b":       _try_groq_20b,
    "groq_120b":      _try_groq_120b,
    "groq_qwen27b":   _try_groq_qwen27b,
    "groq_8b":        _try_groq_20b,
    "groq_70b":       _try_groq_120b,
    "groq_secondary": _try_groq_secondary,
    "groq_third":     _try_groq_third,
    "gemini_flash":   _try_gemini,
    "cohere_r":       _try_cohere,
    "cloudflare_3b":  _try_cloudflare,
    "gonka_deepseek": _try_gonka_deepseek,
    "gonka_minimax":  _try_gonka,
    "gonka_glm":      _try_gonka_glm,
    "kilo_free":      _try_kilo,
    "nvidia_nim":     _try_nvidia,
    "openrouter":     _try_openrouter,
    "ollama":         _try_ollama,
}

_DEFAULT_ROUTER_CHAIN = [
    ("Groq GPT-OSS 20B", _try_groq_20b),
    ("Cloudflare 3B",    _try_cloudflare),
    ("Gemini Flash",     _try_gemini),
    ("Kilo Free",        _try_kilo),
    ("Ollama",           _try_ollama),
]

_DEFAULT_RAG_AND_RESEARCH_CHAIN = [
    ("Gemini Flash",      _try_gemini),
    ("Cohere R+",         _try_cohere),
    ("Gonka DeepSeek",    _try_gonka_deepseek),
    ("Gonka MiniMax",     _try_gonka),
    ("Gonka GLM",         _try_gonka_glm),
    ("Groq GPT-OSS 120B", _try_groq_120b),
    ("Groq Qwen 27B",     _try_groq_qwen27b),
    ("Kilo Free",         _try_kilo),
    ("Nvidia NIM",        _try_nvidia),
    ("Ollama",            _try_ollama),
]

_DEFAULT_GENERAL_CHAT_CHAIN = [
    ("Groq GPT-OSS 120B",     _try_groq_120b),
    ("Gemini Flash",          _try_gemini),
    ("Gonka DeepSeek",        _try_gonka_deepseek),
    ("Gonka MiniMax",         _try_gonka),
    ("Gonka GLM",             _try_gonka_glm),
    ("Cohere R+",             _try_cohere),
    ("Groq GPT-OSS 20B",      _try_groq_20b),
    ("Groq Qwen 27B",         _try_groq_qwen27b),
    ("Cloudflare 3B",         _try_cloudflare),
    ("Kilo Free",             _try_kilo),
    ("Groq Secondary (120B)", _try_groq_secondary),
    ("Nvidia NIM",            _try_nvidia),
    ("OpenRouter",            _try_openrouter),
    ("Groq Third (120B)",     _try_groq_third),
    ("Ollama",                _try_ollama),
]


def _build_chain(chain_key: str, default: list[tuple[str, any]]) -> list[tuple[str, any]]:
    configured = _ROUTING_CONFIG.get("chains", {}).get(chain_key)
    if not configured:
        return default
    built = []
    for item in configured:
        if not item.get("enabled", True):
            continue
        pid = item.get("id")
        label = item.get("label", pid)
        fn = _DISPATCH_REGISTRY.get(pid)
        if fn:
            built.append((label, fn))
    return built or default


_ROUTER_CHAIN = _build_chain("router", _DEFAULT_ROUTER_CHAIN)
_RAG_AND_RESEARCH_CHAIN = _build_chain("rag_and_research", _DEFAULT_RAG_AND_RESEARCH_CHAIN)
_GENERAL_CHAT_CHAIN = _build_chain("general_chat", _DEFAULT_GENERAL_CHAT_CHAIN)


# ---------------------------------------------------------------------------
# LLMProvider Class
# ---------------------------------------------------------------------------

class LLMProvider:
    """
    Unified LLM interface with Task-Specific Routing Chains,
    In-Memory 429 Cooldown Tracking, and Live Telemetry.
    """

    _global_last_telemetry: dict = {
        "provider": "Groq 70B",
        "task": "general",
        "failover_count": 0,
        "latency_ms": 0,
    }

    @classmethod
    def reload_routing_config(cls) -> dict:
        """Hot-reload routing_config.json into the active chains without restarting the server."""
        global _ROUTING_CONFIG, _ROUTER_CHAIN, _RAG_AND_RESEARCH_CHAIN, _GENERAL_CHAT_CHAIN
        _ROUTING_CONFIG = load_routing_config()
        _ROUTER_CHAIN = _build_chain("router", _DEFAULT_ROUTER_CHAIN)
        _RAG_AND_RESEARCH_CHAIN = _build_chain("rag_and_research", _DEFAULT_RAG_AND_RESEARCH_CHAIN)
        _GENERAL_CHAT_CHAIN = _build_chain("general_chat", _DEFAULT_GENERAL_CHAT_CHAIN)
        logger.info(
            "[LLMProvider] routing_config.json reloaded: router=%d, rag=%d, general=%d",
            len(_ROUTER_CHAIN),
            len(_RAG_AND_RESEARCH_CHAIN),
            len(_GENERAL_CHAT_CHAIN),
        )
        return {
            "router_providers": [l for l, _ in _ROUTER_CHAIN],
            "rag_providers": [l for l, _ in _RAG_AND_RESEARCH_CHAIN],
            "general_providers": [l for l, _ in _GENERAL_CHAT_CHAIN],
        }

    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower()
        self.last_telemetry = dict(LLMProvider._global_last_telemetry)

        # Primary client for streaming
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama
            self._ollama = ChatOllama(model=settings.llm_model, temperature=0)
            self._client = None
        else:
            endpoints = {
                "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", settings.gemini_api_key),
                "nvidia":     ("https://integrate.api.nvidia.com/v1",                      settings.nvidia_nim_api_key or settings.nvidia_nim),
                "groq":       ("https://api.groq.com/openai/v1",                          settings.groq_api_key or settings.groq_api_third),
                "openrouter": ("https://openrouter.ai/api/v1",                            settings.open_router_key),
                "gonka":      ("https://proxy.gonkabroker.com/v1",                        settings.gonka_api_key),
                "kilo":       ("https://ai-gateway.kilo.ai/api/v1",                       settings.kilo_api_key),
                "cloudflare": (
                    f"https://api.cloudflare.com/client/v4/accounts/{settings.cloudflare_account_id}/ai/v1"
                    if settings.cloudflare_account_id else None,
                    settings.cloudflare_ai_api or settings.cloudflare_api_token,
                ),
            }
            url, key = endpoints.get(self.provider, (None, None))
            self._client = _openai_client(api_key=key, base_url=url) if key and url else None
            self._ollama = None

    @classmethod
    def get_last_telemetry(cls) -> dict:
        """Return the latest LLM execution telemetry for SSE event emitting."""
        return cls._global_last_telemetry

    # ------------------------------------------------------------------
    # invoke  –  Task-Specific Chains + 0ms 429 Cooldown Skip
    # ------------------------------------------------------------------

    async def invoke(self, messages: list[dict], task: str = "general") -> str:
        """
        Execute prompt against the appropriate task chain with 0ms cooldown skipping.

        Tasks:
        - 'router': uses _ROUTER_CHAIN (Groq 8B, Cloudflare 3B, Gemini Flash)
        - 'rag' or 'research': uses _RAG_AND_RESEARCH_CHAIN (Gemini Flash, Cohere, Gonka MiniMax 200k)
        - 'general': uses _GENERAL_CHAT_CHAIN (Groq 70B, Gemini, Gonka, Cohere, Cloudflare, Kilo)
        """
        import asyncio

        t_start = time.perf_counter()
        last_exc: Exception | None = None

        # 1. Pick appropriate specialized chain
        if task in ("router", "classification"):
            chain = _ROUTER_CHAIN
        elif task in ("rag", "document", "research", "evidence_fusion"):
            chain = _RAG_AND_RESEARCH_CHAIN
        else:
            chain = _GENERAL_CHAT_CHAIN

        # If user explicitly requested non-default provider in config
        start_idx = 0
        if self.provider != "groq":
            for i, (label, _) in enumerate(chain):
                if self.provider in label.lower():
                    start_idx = i
                    break

        ordered_chain = chain[start_idx:] + chain[:start_idx]
        failover_count = 0

        for label, fn in ordered_chain:
            # 2. 0ms Cooldown Skip: don't make network calls to providers recently hitting 429
            if not is_healthy(label):
                logger.debug("[LLMProvider] Skipping %s in 0ms (429 cooldown active)", label)
                continue

            max_retries = 2 if "Secondary" in label or "Cohere" in label else 1
            for attempt in range(1, max_retries + 1):
                try:
                    result = await fn(messages)
                    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

                    if label != ordered_chain[0][0]:
                        logger.warning(
                            "[LLMProvider] Auto-failover triggered for task='%s': used %s in %.1fms (failovers: %d)",
                            task, label, elapsed_ms, failover_count
                        )

                    # Record live telemetry
                    self.last_telemetry = {
                        "provider": label,
                        "task": task,
                        "failover_count": failover_count,
                        "latency_ms": elapsed_ms,
                    }
                    LLMProvider._global_last_telemetry = self.last_telemetry
                    return result

                except Exception as exc:
                    last_exc = exc
                    failover_count += 1

                    if _is_rate_limit(exc):
                        # Mark 429 cooldown for 60 seconds
                        mark_rate_limited(label, 60.0)
                        logger.warning(
                            "[LLMProvider] %s hit 429 rate limit; marked in 60s cooldown. Failing over instantly.",
                            label
                        )
                        break  # Immediately hop to next provider!

                    if attempt < max_retries:
                        await asyncio.sleep(1)
                        continue

                    logger.warning("[LLMProvider] %s failed (%s), trying next provider in %s chain.", label, exc, task)
                    break

        raise RuntimeError(
            f"All LLM providers in task '{task}' chain exhausted. Active cooldowns: {get_cooldown_status()}. Last error: {last_exc}"
        )

    def supports_native_tool_calls(self) -> bool:
        return self.provider in {"groq", "openrouter", "nvidia"}

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
    # stream  –  Primary streaming with graceful invoke() fallback
    # ------------------------------------------------------------------

    async def stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        if self.provider == "ollama":
            async for chunk in self._ollama.astream(messages):
                yield chunk.content
            return

        if self._client is None:
            yield await self.invoke(messages)
            return

        try:
            response = await self._client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                stream=True,
            )
            async for chunk in response:
                yield chunk.choices[0].delta.content or ""
        except Exception as exc:
            if _is_rate_limit(exc):
                mark_rate_limited(self.provider.capitalize(), 60.0)
                logger.warning("[LLMProvider] Stream hit 429 on %s, invoking fallback cascade...", self.provider)
                yield await self.invoke(messages, task="general")
            else:
                raise
