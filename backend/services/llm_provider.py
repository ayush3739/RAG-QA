"""
backend/services/llm_provider.py

Single LLM abstraction for DocuMind.
Provider and model are driven entirely by environment variables:

    LLM_PROVIDER=ollama   LLM_MODEL=qwen3:4b
    LLM_PROVIDER=groq     LLM_MODEL=openai/gpt-oss-20b
    LLM_PROVIDER=github   LLM_MODEL=openai/gpt-4.1-mini

Messages must follow OpenAI format everywhere:

    [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "..."},
    ]
"""

from __future__ import annotations

from typing import AsyncGenerator

from backend.core.config import settings


def _openai_client(api_key: str, base_url: str):
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


class LLMProvider:
    """
    Thin abstraction over multiple LLM back-ends.

    The underlying client/model is instantiated once in __init__
    and reused across all calls.

    Usage
    -----
    llm = LLMProvider()

    # Non-streaming (intent classifier, agent nodes, …)
    response: str = await llm.invoke(messages)

    # Streaming (chat, research report, …)
    async for token in llm.stream(messages):
        yield token
    """

    def __init__(self) -> None:
        self.provider = settings.llm_provider

        if self.provider == "ollama":
            from langchain_ollama import ChatOllama

            self._ollama = ChatOllama(
                model=settings.llm_model,
                temperature=0,
            )

        elif self.provider == "groq":
            self._client = _openai_client(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )

        elif self.provider == "github":
            self._client = _openai_client(
                api_key=settings.github_token,
                base_url="https://models.github.ai/inference",
            )

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    # ------------------------------------------------------------------
    # invoke  –  single-shot, returns the full response as a string
    # ------------------------------------------------------------------

    async def invoke(self, messages: list[dict]) -> str:
        if self.provider == "ollama":
            result = await self._ollama.ainvoke(messages)
            return result.content

        # groq + github share the same OpenAI-compatible path
        response = await self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # stream  –  async-generator, yields string tokens
    # ------------------------------------------------------------------

    async def stream(
        self, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        if self.provider == "ollama":
            async for chunk in self._ollama.astream(messages):
                yield chunk.content
            return

        # groq + github
        response = await self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            yield chunk.choices[0].delta.content or ""