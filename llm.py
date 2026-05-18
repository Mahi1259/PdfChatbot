"""Hugging Face Inference Providers client for answer generation.

Uses the OpenAI-compatible router at router.huggingface.co, which is the
modern replacement for the deprecated api-inference.huggingface.co/models/* API.
"""
from __future__ import annotations

import os

import requests

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_TIMEOUT = 90

SUPPORTED_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    # Originals from the spec (may 400 if no enabled provider hosts them):
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "tiiuae/falcon-7b-instruct",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a PDF document. "
    "Answer ONLY based on the context the user provides. "
    "If the answer is not in the context, say \"I don't know.\""
)


class LLMError(Exception):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass


def _build_user_message(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    return f"Context:\n{context_block}\n\nQuestion: {question}"


def generate_answer(
    model: str,
    question: str,
    contexts: list[str],
    api_token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_new_tokens: int = 512,
) -> str:
    token = api_token or os.getenv("HUGGING_FACE_API_TOKEN")
    if not token:
        raise LLMError("HUGGING_FACE_API_TOKEN is not set. Add it to your .env file.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(question, contexts)},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "stream": False,
    }

    try:
        resp = requests.post(HF_ROUTER_URL, headers=headers, json=body, timeout=timeout)
    except requests.Timeout as e:
        raise LLMTimeoutError(
            f"The model took longer than {timeout}s to respond. Try again or pick a smaller model."
        ) from e
    except requests.ConnectionError as e:
        raise LLMUnavailableError("Could not reach the Hugging Face Inference router.") from e

    if resp.status_code == 503:
        raise LLMUnavailableError(
            "The Hugging Face model is loading or temporarily unavailable. Try again shortly."
        )
    if resp.status_code == 401:
        raise LLMError("Authentication failed. Check your HUGGING_FACE_API_TOKEN.")
    if resp.status_code == 403:
        raise LLMError(
            f"Access denied for model '{model}' (HTTP 403). "
            f"HF said: {resp.text}\n\n"
            "Common causes:\n"
            "1. Your token is missing inference permissions — go to "
            "https://huggingface.co/settings/tokens, edit the token, and enable "
            "'Make calls to Inference Providers' (or generate a Write/Fine-grained token).\n"
            "2. The model requires accepting its license — visit the model page on huggingface.co.\n"
            "3. The model isn't available on the free tier; pick another from the dropdown."
        )
    if resp.status_code == 404:
        raise LLMError(
            f"Model '{model}' is not available through Hugging Face Inference Providers. "
            "Try a different model from the dropdown."
        )
    if resp.status_code >= 400:
        raise LLMError(f"Hugging Face API error {resp.status_code}: {resp.text}")

    payload = resp.json()
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape: {payload!r}") from e
