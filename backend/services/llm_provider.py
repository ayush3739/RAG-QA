"""
backend/services/llm_provider.py

Single LLM abstraction for DocuMind.

Active provider is set in backend/core/config.py:
    settings.llm_provider = "bedrock"
    settings.llm_model   = "meta.llama3-3-70b-instruct-v1:0"

On a 429 / rate-limit from the primary provider, invoke() automatically
falls through this chain until one succeeds:

    AWS Bedrock (meta.llama3-3-70b-instruct-v1:0)
        → Gemini (gemini-2.0-flash)
        → Nvidia NIM (meta/llama-3.3-70b-instruct)
        → Groq (llama-3.3-70b-versatile)
        → OpenRouter (meta-llama/llama-3.3-70b-instruct:free)
        → GitHub AI (gpt-4o-mini)
        → Local Ollama (ollama_model)

All cloud providers use OpenAI-compatible REST endpoints, except Bedrock which
uses the boto3 Converse API or direct Bedrock Bearer Token REST endpoints.
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
    return "429" in s or "rate limit" in s or "too many requests" in s or "quota" in s or "504" in s


# ---------------------------------------------------------------------------
# Fallback chain definition
# Each entry: (label, callable that returns the answer string or raises)
# Built lazily so we don't fail at import time if a key is missing.
# ---------------------------------------------------------------------------

async def _try_gemini(messages: list[dict]) -> str:
    """Call Gemini via Google's OpenAI-compatible beta endpoint."""
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


async def _try_bedrock(messages: list[dict]) -> str:
    """Call AWS Bedrock using either a single API Key (Bearer Token) or standard boto3 (IAM keys)."""
    key_id = settings.aws_access_key_id
    secret = settings.aws_secret_access_key
    
    if not key_id:
        raise RuntimeError("No AWS Bedrock credentials or API key configured.")

    model_id = settings.llm_model
    region = settings.aws_region_name or "us-east-1"

    # MODE B: Single API Key / Bearer Token mode
    if not secret:
        import httpx
        url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse"
        
        system_prompts = []
        bedrock_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "system":
                system_prompts.append({"text": content})
            elif role in ["user", "assistant"]:
                bedrock_messages.append({
                    "role": role,
                    "content": [{"text": content}]
                })
        
        payload = {
            "messages": bedrock_messages,
        }
        if system_prompts:
            payload["system"] = system_prompts

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_id}"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Bedrock REST API error {resp.status_code}: {resp.text}")
            data = resp.json()
            return data["output"]["message"]["content"][0]["text"]

    # MODE A: Standard boto3 SDK / IAM role mode
    import boto3
    import asyncio

    def _call_bedrock():
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            aws_session_token=settings.aws_session_token,
        )
        
        system_prompts = []
        bedrock_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "system":
                system_prompts.append({"text": content})
            elif role in ["user", "assistant"]:
                bedrock_messages.append({
                    "role": role,
                    "content": [{"text": content}]
                })
        
        params = {
            "modelId": model_id,
            "messages": bedrock_messages,
        }
        if system_prompts:
            params["system"] = system_prompts
            
        response = client.converse(**params)
        return response["output"]["message"]["content"][0]["text"]

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_bedrock)


async def _try_nvidia(messages: list[dict]) -> str:
    """Call Nvidia NIM API endpoint."""
    key = settings.nvidia_nim
    if not key:
        raise RuntimeError("No Nvidia NIM API key configured.")
    client = _openai_client(
        api_key=key,
        base_url="https://integrate.api.nvidia.com/v1",
    )
    resp = await client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
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
        model=settings.llm_model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_groq_secondary(messages: list[dict]) -> str:
    key = settings.groq_api_secondary
    if not key:
        raise RuntimeError("No Groq Secondary API key configured.")
    client = _openai_client(api_key=key, base_url="https://api.groq.com/openai/v1")
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def _try_groq_third(messages: list[dict]) -> str:
    key = settings.groq_api_third
    if not key:
        raise RuntimeError("No Groq Third API key configured.")
    client = _openai_client(api_key=key, base_url="https://api.groq.com/openai/v1")
    resp = await client.chat.completions.create(
        model=settings.llm_model,
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
    # ("Bedrock",        _try_bedrock),
    ("Groq Third",     _try_groq_third),
    ("Groq Secondary", _try_groq_secondary),
    ("Groq",           _try_groq),
]

# Map provider name → its position so the primary always starts at index 0
_PROVIDER_INDEX = {
    # "bedrock": 0,
    "groq":    0,
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
        self._start_idx = _PROVIDER_INDEX.get(self.provider, 0)

        # Build a primary client for streaming (stream() only uses primary)
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama
            self._ollama = ChatOllama(model=settings.llm_model, temperature=0)
            self._client = None
        elif self.provider == "bedrock":
            self._ollama = None
            self._client = None
        else:
            endpoints = {
                "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", settings.gemini_api_key),
                "nvidia":     ("https://integrate.api.nvidia.com/v1",                      settings.nvidia_nim),
                "groq":       ("https://api.groq.com/openai/v1",                          settings.groq_api_third),
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
        import asyncio
        last_exc: Exception | None = None

        for label, fn in _FALLBACK_CHAIN[self._start_idx:] + _FALLBACK_CHAIN[:self._start_idx]:
            max_retries = 3 if label in ("Bedrock", "Groq Secondary") else 1
            for attempt in range(1, max_retries + 1):
                try:
                    result = await fn(messages)
                    if label != _FALLBACK_CHAIN[self._start_idx][0]:
                        logger.warning("[LLMProvider] Used fallback: %s", label)
                    return result
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        logger.warning(
                            "[LLMProvider] %s failed (attempt %d/%d): %s. Retrying in 2s...", 
                            label, attempt, max_retries, exc
                        )
                        await asyncio.sleep(2)
                        continue
                    
                    if _is_rate_limit(exc):
                        logger.warning(
                            "[LLMProvider] %s rate-limited/timed out, trying next fallback.", label
                        )
                    else:
                        # Non-rate-limit error on primary → still try fallbacks
                        logger.warning("[LLMProvider] %s failed (%s), trying next.", label, exc)
                    
                    break # Break inner retry loop, continue to next provider

        raise RuntimeError(
            f"All LLM providers exhausted. Last error: {last_exc}"
        )

    def supports_native_tool_calls(self) -> bool:
        return self.provider in {"groq", "github", "openrouter", "nvidia"}

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

        if self.provider == "bedrock":
            yield await self.invoke(messages)
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
