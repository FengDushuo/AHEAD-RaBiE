#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests


class MultiEndpointClient:
    """Small OpenAI-compatible client for local vLLM and external providers such as SCNet.

    Key features added for strict literature extraction:
    - OpenAI-compatible authentication via environment variables or explicit api_key.
    - Rotating multi-endpoint support for local vLLM, while still supporting a single HTTPS endpoint.
    - Chat-first extraction with optional JSON mode and robust fallback when providers reject response_format.
    - Conservative system prompt suitable for evidence-grounded scientific extraction.

    Environment variables recognized for API keys:
      SCNET_API_KEY, OPENAI_API_KEY, LLM_API_KEY, API_KEY
    """

    def __init__(
        self,
        api_bases: List[str],
        model_id: str,
        timeout: int = 180,
        max_retries: int = 3,
        retry_backoff: float = 1.6,
        prefer_chat: bool = True,
        json_mode: bool = True,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        provider: str = "auto",
    ):
        self.api_bases = [b.rstrip("/") for b in api_bases if str(b).strip()]
        if not self.api_bases:
            raise ValueError("api_bases is empty")
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.prefer_chat = prefer_chat
        self.json_mode = json_mode
        self.provider = provider
        self.api_key = api_key or os.environ.get("SCNET_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("API_KEY")
        self.extra_headers = extra_headers or {}
        self._rr = 0
        self._lock = threading.Lock()

    def _next_base(self) -> str:
        with self._lock:
            b = self.api_bases[self._rr % len(self.api_bases)]
            self._rr += 1
        return b

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _chat_payload(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative scientific information extraction engine. "
                        "Return exactly one valid JSON object only. Do not add Markdown, comments, or explanations. "
                        "Use null for unsupported fields. Copy only evidence-supported values. "
                        "Never infer numbers from trends, axes, page numbers, potentials, temperatures, or unrelated reference systems."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
        # OpenAI-compatible providers and newer vLLM versions support this. If rejected,
        # _try_chat automatically retries without it.
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _completion_payload(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        return {
            "model": self.model_id,
            "prompt": prompt + "\n\nReturn JSON only:",
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }

    def _extract_chat_text(self, j: Dict[str, Any]) -> Optional[str]:
        if "choices" in j and j["choices"]:
            ch0 = j["choices"][0]
            msg = ch0.get("message", {}) or {}
            text = msg.get("content", "")
            # Some reasoning APIs may place final content in other fields.
            if isinstance(text, list):
                parts = []
                for x in text:
                    if isinstance(x, dict):
                        parts.append(str(x.get("text") or x.get("content") or ""))
                    else:
                        parts.append(str(x))
                text = "\n".join(parts)
            text = (text or "").strip()
            return text or None
        return None

    def _try_chat(self, base: str, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        payload = self._chat_payload(prompt, temperature, max_tokens)
        headers = self._headers()
        try:
            r = requests.post(base + "/chat/completions", json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
        except Exception:
            # Some providers reject response_format for specific models. Retry once without it.
            if "response_format" in payload:
                payload.pop("response_format", None)
                r = requests.post(base + "/chat/completions", json=payload, headers=headers, timeout=self.timeout)
                r.raise_for_status()
            else:
                raise
        return self._extract_chat_text(r.json())

    def _try_completion(self, base: str, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        payload = self._completion_payload(prompt, temperature, max_tokens)
        r = requests.post(base + "/completions", json=payload, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if "choices" in j and j["choices"]:
            ch0 = j["choices"][0]
            text = (ch0.get("text") or "").strip()
            return text or None
        return None

    def completions(self, prompt: str, temperature: float = 0.0, max_tokens: int = 900) -> str:
        last_err = None
        for attempt in range(self.max_retries + 1):
            base = self._next_base()
            try:
                if self.prefer_chat:
                    text = self._try_chat(base, prompt, temperature, max_tokens)
                    if text:
                        return text
                    # SCNet/OpenAI-compatible chat models normally do not expose /completions;
                    # fallback remains useful for local vLLM legacy endpoints.
                    text = self._try_completion(base, prompt, temperature, max_tokens)
                    if text:
                        return text
                else:
                    text = self._try_completion(base, prompt, temperature, max_tokens)
                    if text:
                        return text
                    text = self._try_chat(base, prompt, temperature, max_tokens)
                    if text:
                        return text
            except Exception as e:
                last_err = e
            time.sleep(min(12.0, (self.retry_backoff ** attempt) * 0.45))
        raise last_err if last_err is not None else RuntimeError("LLM request failed without explicit error")


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def safe_json_extract(s: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from a model response.

    Handles common deviations: Markdown fences, leading text, trailing commas,
    and extra text after the JSON object. Returns dict only.
    """
    s = _strip_code_fences(s)
    if not s:
        return None

    candidates = [s, _remove_trailing_commas(s)]
    for cand in candidates:
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    start = None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            if start is not None:
                depth -= 1
                if depth == 0:
                    raw = s[start : i + 1]
                    for cand in (raw, _remove_trailing_commas(raw)):
                        try:
                            obj = json.loads(cand)
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            pass
                    start = None
                    depth = 0
    return None
