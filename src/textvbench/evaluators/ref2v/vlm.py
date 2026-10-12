"""Strict JSON OpenAI-compatible multimodal client."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Callable, Iterable

from openai import OpenAI, OpenAIError

from ..vlm_schema import request_kwargs, request_with_response_format
from ..vlm_validation import InvalidCandidateSelectionError


class VLMError(RuntimeError):
    """A VLM dependency or request failure that must stop evaluation."""


class VLMClient:
    def __init__(self, config: dict[str, Any]):
        llm = config.get("llm", {})
        profile_name = llm.get("active_profile")
        profile = llm.get("profiles", {}).get(profile_name, {})
        settings = {**llm, **profile}
        self.model = settings["model"]
        self.structured_output = bool(llm.get("structured_output"))
        self.request = request_kwargs(config)
        self.request["temperature"] = 0.0
        self.client = OpenAI(
            base_url=settings["base_url"],
            api_key=settings.get("api_key", "EMPTY"),
            timeout=float(llm.get("timeout_sec", 180)),
            max_retries=0,
        )
        self.record_retries = int(llm.get("record_retries", 2))

    def complete_json(
        self,
        prompt: str,
        payload: dict[str, Any],
        image_urls: Iterable[str],
        *,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
            **(request or self.request),
        )
        text = response.choices[0].message.content or ""
        if text.strip().startswith("```"):
            text = text.strip().split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("VLM response must be a JSON object")
        return value

    def complete_with_retries(
        self,
        prompt: str,
        payload: dict[str, Any],
        image_urls: Iterable[str],
        validator: Callable[[dict[str, Any]], None],
        *,
        response_format: dict[str, Any] | None = None,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, Any], int]:
        urls = list(image_urls)
        request = request_with_response_format(
            self.request, response_format if getattr(self, "structured_output", False) else None,
        )
        previous_error = ""
        last_error: Exception | None = None
        fatal_error: Exception | None = None
        for attempt in range(self.record_retries + 1):
            attempt_prompt = prompt + ("\n\nPrevious response error: " + previous_error if previous_error else "")
            result: dict[str, Any] | None = None
            try:
                if response_format is None:
                    result = self.complete_json(attempt_prompt, payload, urls)
                else:
                    result = self.complete_json(attempt_prompt, payload, urls, request=request)
                validator(result)
                trace = {"attempt": attempt + 1, "status": "ok", "model": self.model,
                         "request": request,
                         "system_prompt": attempt_prompt, "response": result}
                if trace_callback:
                    trace_callback(trace)
                return result, attempt + 1
            except Exception as exc:
                last_error = exc
                if isinstance(exc, (OpenAIError, VLMError, OSError, TimeoutError, socket.timeout)):
                    fatal_error = exc
                previous_error = str(exc)
                recoverable = (
                    attempt == self.record_retries
                    and fatal_error is None
                    and isinstance(exc, InvalidCandidateSelectionError)
                )
                if trace_callback:
                    trace = {"attempt": attempt + 1,
                             "status": "recovered_invalid_candidates" if recoverable else (
                                 "validation_error" if result is not None else "request_error"
                             ),
                             "model": self.model, "request": request, "system_prompt": attempt_prompt,
                             "response": result, "error": str(exc)}
                    if isinstance(exc, InvalidCandidateSelectionError):
                        trace["recovered_response"] = exc.recovered_response
                        trace["removed_ids"] = exc.removed_ids
                    trace_callback(trace)
                if attempt < self.record_retries:
                    time.sleep(2 ** attempt)
        if fatal_error is not None:
            raise VLMError(f"VLM request failed after {self.record_retries + 1} attempts: {fatal_error}") from fatal_error
        if isinstance(last_error, InvalidCandidateSelectionError):
            return last_error.recovered_response, self.record_retries + 1
        raise last_error  # type: ignore[misc]
