"""OpenRouter adapter for the Liar's Bar code.

Replaces the missing src/llm/llm_client.py and the local GPU client.
Every player call goes to OpenRouter instead of a local model.
"""
import os
import time
import requests

# Short names used in the game files -> OpenRouter model IDs.
# Every short name in the game's player list MUST appear here.
MODEL_MAP = {
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3-8b": "sao10k/l3-lunaris-8b",
    "qwen2.5-7b": "qwen/qwen-2.5-7b-instruct",
    "mistral-7b": "meta-llama/llama-3.1-8b-instruct",  # Mistral 7B is not on OpenRouter; stand-in
}

URL = "https://openrouter.ai/api/v1/chat/completions"
CALLS = {"n": 0}


class LLMClient:
    def __init__(self, *args, **kwargs):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

    def chat(self, messages, model="llama-3.1-8b", max_new_tokens=1024):
        if model not in MODEL_MAP:
            raise RuntimeError(f"Model '{model}' is not in MODEL_MAP in llm_client.py")
        model_id = MODEL_MAP[model]
        body = {"model": model_id, "messages": messages,
                "max_tokens": max_new_tokens, "temperature": 0.7}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(3):
            try:
                r = requests.post(URL, json=body, headers=headers, timeout=120)
            except requests.RequestException as e:  # network problem: retry
                print(f"Network error ({attempt + 1}/3) for {model_id}: {e}")
                time.sleep(2 * (attempt + 1))
                continue
            if 400 <= r.status_code < 500 and r.status_code != 429:
                # Bad model name, bad key, no credit: retrying will not help. Stop the game.
                raise RuntimeError(f"OpenRouter {r.status_code} for {model_id}: {r.text[:300]}")
            if r.status_code != 200:  # 429 rate limit or 5xx server error: retry
                print(f"OpenRouter {r.status_code} ({attempt + 1}/3) for {model_id}, retrying")
                time.sleep(2 * (attempt + 1))
                continue
            content = r.json()["choices"][0]["message"].get("content") or ""
            CALLS["n"] += 1
            if CALLS["n"] % 20 == 0:
                print(f"[adapter] {CALLS['n']} API calls so far")
            return content.strip(), ""
        raise RuntimeError(f"OpenRouter failed 3 times for {model_id}")
