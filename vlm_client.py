"""Asynchronous client for the Ollama VLM endpoint configured in .env."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv
from ollama import AsyncClient, ResponseError

load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:e4b").strip() or "gemma4:e4b"
OLLAMA_HOST = (
    os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    or "http://localhost:11434"
)
SYSTEM_PROMPT = (
    "You are a concise voice assistant. Respond in 1 to 2 short, conversational "
    "sentences. Do not use Markdown, bullet points, or special formatting."
)


async def generate_response(
    prompt: str, image_bytes: bytes | None = None
) -> str:
    """Stream a concise response from the local Ollama model."""
    if not prompt.strip():
        return ""

    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_bytes is not None:
        user_message["images"] = [image_bytes]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        user_message,
    ]
    response_parts: list[str] = []
    try:
        async with AsyncClient(host=OLLAMA_HOST) as client:
            stream = await client.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )
            print("\nAssistant: ", end="", flush=True)
            async for chunk in stream:
                token = chunk.message.content
                if token:
                    print(token, end="", flush=True)
                    response_parts.append(token)
            print(flush=True)
    except ConnectionError:
        print(
            f"\nOllama server at {OLLAMA_HOST} is not reachable. "
            "Check that the server is running and the endpoint is reachable.",
            file=sys.stderr,
            flush=True,
        )
        return ""
    except ResponseError as error:
        if error.status_code == 404 or "not found" in str(error).lower():
            message = (
                f"Ollama model '{MODEL_NAME}' is not available. "
                f"Pull it on the Ollama server with `ollama pull {MODEL_NAME}`."
            )
        else:
            message = f"Ollama request failed: {error}"
        print(f"\n{message}", file=sys.stderr, flush=True)
        return ""

    return "".join(response_parts)


async def _run_typed_prompt() -> None:
    prompt = input("Enter a prompt to test the local Ollama model: ")
    response = await generate_response(prompt)
    if not response:
        print("No response was returned.")


if __name__ == "__main__":
    asyncio.run(_run_typed_prompt())
