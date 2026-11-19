import os
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Keys shipped in .env.example / left unconfigured. When the key is one of these
# (or obviously a stand-in), we skip the network call entirely and let callers
# fall back to local fuzzy matching — no wasted round-trip to the LLM provider.
_PLACEHOLDER_KEYS = {"", "your_key_here", "your-key-here", "changeme", "none", "null", "test", "sk-xxx"}


def llm_enabled() -> bool:
    """True only when a real LLM key is configured, not a placeholder."""
    key = (LLM_API_KEY or "").strip()
    low = key.lower()
    if low in _PLACEHOLDER_KEYS or low.startswith("your_") or low.endswith("_here"):
        return False
    return len(key) >= 16


async def chat_completion(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    # Fast path: no real key configured -> don't attempt a doomed network call.
    if not llm_enabled():
        return None

    url = f"{LLM_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        logger.error("LLM request timed out")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return None
