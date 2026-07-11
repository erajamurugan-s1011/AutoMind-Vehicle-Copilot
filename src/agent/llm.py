"""
AutoMind — LLM wrapper with provider switch.

Supports two backends behind the same interface:
  - "ollama" (default): local Qwen2.5-7B via Ollama, zero cost, needs your
    machine's GPU running.
  - "groq": free cloud API (Llama 3.3 70B), no local hardware needed — this
    is what makes AutoMind deployable as a real public demo instead of only
    running on your laptop.

Switch via the LLM_PROVIDER env var (see src/config.py). Everything else in
the codebase (nodes.py, etc.) just calls chat()/simple_prompt() and doesn't
know or care which provider is active.

Groq setup (free): create an account at https://console.groq.com, generate
an API key, set GROQ_API_KEY as an environment variable.
"""

import requests
from src.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
)


def _chat_ollama(messages: list, temperature: float) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _chat_groq(messages: list, temperature: float) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and set it as an env var."
        )
    response = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def chat(messages: list, temperature: float = 0.3) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns the model's reply text. Routes to Ollama or Groq based on
    LLM_PROVIDER — same call signature either way.
    """
    if LLM_PROVIDER == "groq":
        return _chat_groq(messages, temperature)
    return _chat_ollama(messages, temperature)


def simple_prompt(prompt: str, system: str = None, temperature: float = 0.3) -> str:
    """Convenience wrapper for a single-turn prompt without manual message building."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, temperature=temperature)


if __name__ == "__main__":
    # smoke test — confirms whichever provider is active is reachable
    print(f"🤖 Testing LLM connection (provider: {LLM_PROVIDER}) ...")
    reply = simple_prompt("Say 'AutoMind agent is online' and nothing else.")
    print(f"Response: {reply}")