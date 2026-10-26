"""OpenRouter adapter for the Liar's Bar code.

Replaces the missing src/llm/llm_client.py and the local GPU client.
Every player call goes to OpenRouter instead of a local model.
"""
import ast
import json
import os
import re
import time
import requests

# Short names used in the game files -> OpenRouter model IDs.
# Every short name in the game's player list MUST appear here.
MODEL_MAP = {
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3-8b": "sao10k/l3-lunaris-8b",
    "qwen2.5-7b": "qwen/qwen-2.5-7b-instruct",
    "mistral-7b": "meta-llama/llama-3.1-8b-instruct",  # Mistral 7B is not on OpenRouter; stand-in
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",  # larger monitor, same family as the players
    # 2026 models (prices per million tokens in / out on OpenRouter, Oct 2026)
    "claude-haiku-5.5": "anthropic/claude-haiku-5.5",        # 0.10 / 0.50
    "gpt-6-luna": "openai/gpt-6-luna",                       # 0.10 / 0.50
    "qwen3.8-flash": "qwen/qwen3.8-flash",                   # 0.15 / 0.47
    "deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",   # 0.02 / 1.00
    "gemini-3.8-flash": "google/gemini-3.8-flash",           # 0.825 / 4.125
    "claude-sonnet-5.5": "anthropic/claude-sonnet-5.5",      # 2 / 10 (monitor)
    "gpt-6.1-sol": "openai/gpt-6.1-sol",                     # 2 / 10 (monitor)
}

URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TRIES = 6           # waits 5, 10, 20, 40, 60 s between tries (rate limits of busy models)
CALLS = {"n": 0}
# Usage of this process, written into the game log meta at the end of a game.
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}
# Reasoning for models that support it: "off", "low" or "default" (send nothing, the model decides).
# Set with the LLM_REASONING environment variable or --reasoning in game_experiment.py.
REASONING_PARAM = {"off": {"enabled": False}, "low": {"effort": "low"}, "default": None}
REASONING_EXTRA_TOKENS = 4000   # added to max_tokens when reasoning is on (1024 was used up by thinking alone)
REASONING_MODELS = {"claude-haiku-5.5", "gpt-6-luna", "qwen3.8-flash", "deepseek-v4.1-flash",
                    "gemini-3.8-flash", "claude-sonnet-5.5", "gpt-6.1-sol"}


def reasoning_mode() -> str:
    mode = os.environ.get("LLM_REASONING", "off")
    if mode not in REASONING_PARAM:
        raise RuntimeError(f"LLM_REASONING must be one of {list(REASONING_PARAM)}, not '{mode}'")
    return mode


def clean_json(text):
    """Turn a small model's almost-JSON answer into real JSON.
    Handles: ```json fences, single quotes ['A'], missing quotes [A, Q],
    Python-style True/False/None."""
    text = re.sub(r"```json\s*|```", "", text)
    m = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text)
    if not m:
        return text.strip()
    s = m.group(1)
    # 1. Already valid JSON
    try:
        return json.dumps(json.loads(s))
    except json.JSONDecodeError:
        pass
    # 2. Python-style dict: single quotes, True/False/None
    try:
        return json.dumps(ast.literal_eval(s))
    except Exception:
        pass
    # 3. Missing quotes around card names: [A, Joker] -> ["A", "Joker"]
    t = re.sub(r'(?<=[\[,])\s*(?!true\b|false\b|null\b)([A-Za-z][A-Za-z0-9]*)\s*(?=[,\]])', r' "\1"', s)
    t = re.sub(r':\s*(?!true\b|false\b|null\b)([A-Za-z][A-Za-z0-9]*)(?=\s*[,}])', r': "\1"', t)
    try:
        return json.dumps(json.loads(t))
    except json.JSONDecodeError:
        pass
    try:
        return json.dumps(ast.literal_eval(t))
    except Exception:
        return s


class LLMClient:
    def __init__(self, *args, **kwargs):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

    def chat(self, messages, model="llama-3.1-8b", max_new_tokens=1024, temperature=0.7):
        if model not in MODEL_MAP:
            raise RuntimeError(f"Model '{model}' is not in MODEL_MAP in llm_client.py")
        model_id = MODEL_MAP[model]
        body = {"model": model_id, "messages": messages,
                "max_tokens": max_new_tokens, "temperature": temperature,
                "usage": {"include": True}}          # the response then carries the real cost
        if model in REASONING_MODELS and REASONING_PARAM[reasoning_mode()] is not None:
            body["reasoning"] = REASONING_PARAM[reasoning_mode()]
            if reasoning_mode() != "off":    # reasoning tokens count toward max_tokens: leave room for the answer
                body["max_tokens"] = max_new_tokens + REASONING_EXTRA_TOKENS
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(MAX_TRIES):
            try:
                r = requests.post(URL, json=body, headers=headers, timeout=120)
            except requests.RequestException as e:  # network problem: retry
                print(f"Network error ({attempt + 1}/{MAX_TRIES}) for {model_id}: {e}")
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if 400 <= r.status_code < 500 and r.status_code != 429:
                # Bad model name, bad key, no credit: retrying will not help. Stop the game.
                raise RuntimeError(f"OpenRouter {r.status_code} for {model_id}: {r.text[:300]}")
            if r.status_code != 200:  # 429 rate limit or 5xx server error: wait longer each time
                print(f"OpenRouter {r.status_code} ({attempt + 1}/{MAX_TRIES}) for {model_id}, retrying")
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            data = r.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or ""     # reasoning text, if the model returns it
            usage = data.get("usage") or {}
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += usage.get("prompt_tokens") or 0
            USAGE["completion_tokens"] += usage.get("completion_tokens") or 0
            USAGE["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            USAGE["cost_usd"] += usage.get("cost") or 0.0
            CALLS["n"] += 1
            if CALLS["n"] % 20 == 0:
                print(f"[adapter] {CALLS['n']} API calls so far")
            content = content.strip()
            if "{" in content:
                content = clean_json(content)
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    # Save answers we could not repair, so you can see what the model wrote
                    with open("bad_json.log", "a", encoding="utf-8") as f:
                        f.write(f"--- {model} ({model_id})\n{content}\n\n")
            return content, reasoning
        raise RuntimeError(f"OpenRouter failed {MAX_TRIES} times for {model_id}")
