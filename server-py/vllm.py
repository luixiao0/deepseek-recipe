"""vLLM inference backend for deepseek-recipe's FastAPI example.

`main.py` converts Chat Completions, Responses, and Messages requests, renders
the DeepSeek V4.1 prompt, and parses model output back into the same protocols.
This module only completes that rendered prompt on vLLM's
``POST /v1/completions`` endpoint and streams the generated tokens back.

Environment:

===========================  ==================================================
``VLLM_BASE_URL``            vLLM OpenAI server root (e.g. ``http://127.0.0.1:8000``)
``VLLM_MODEL``               served model name
``TOKENIZER_PATH``           ``tokenizer.json`` used by the prompt encoding.
                             Enables token-level streaming (required for
                             ``completion_tokens_details.reasoning_tokens`` and
                             for thinking/content segmentation). Without it the
                             backend falls back to vLLM's text stream.
``VLLM_API_KEY``             optional ``Authorization: Bearer`` value
``VLLM_MAX_MODEL_LEN``       optional context length override; otherwise read
                             from ``GET /v1/models``
``SYSTEM_FINGERPRINT``       optional value reported as ``system_fingerprint``
``IMAGE_PLACEHOLDER``        image placeholder spelling (default
                             ``<｜deepseek_image｜>`` for DeepSeek-V4.1-Flash)
===========================  ==================================================

Images are forwarded as ``multi_modal_data`` on the completion request. That
field is an extension of this repository's vLLM deployment; see
``docs/vllm-backend.md`` for the required patch.
"""

import base64
import json
import os
from collections.abc import AsyncIterator

import httpx
from deepseek_recipe import (
    DeepseekV41Encoding,
    InferenceChunk,
    InferenceFinishReason,
    ModelVersion,
    PromptUsage,
    SpecialTokens,
    Tokenizer,
)

import main

VLLM_BASE_URL = os.environ["VLLM_BASE_URL"].rstrip("/")
VLLM_MODEL = os.environ["VLLM_MODEL"]
API_KEY = os.environ.get("VLLM_API_KEY")
SYSTEM_FINGERPRINT = os.environ.get("SYSTEM_FINGERPRINT")
IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<｜deepseek_image｜>")

TOKENIZER = None
ENCODING = DeepseekV41Encoding()
if tokenizer_path := os.environ.get("TOKENIZER_PATH"):
    TOKENIZER = Tokenizer.from_file(tokenizer_path)
    ENCODING = ENCODING.with_tokenizer(TOKENIZER).with_special_tokens(
        SpecialTokens(image=IMAGE_PLACEHOLDER)
    )

# Stop token that vLLM appends to the generated token ids; the text stream never
# contains it, so token chunks filter it out.
EOS_TOKEN_ID = TOKENIZER.encode("<｜end▁of▁sentence｜>")[0] if TOKENIZER else None

# vLLM prepends ``COMPRESS_PAD_TO - 1 - start % COMPRESS_PAD_TO`` alignment pad
# tokens before each image span. Without per-image token counts from the
# server example, budget the worst case when resolving ``max_tokens``.
COMPRESS_PAD_TO = 2

HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
CLIENT = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=None, headers=HEADERS)

_context_limit: int | None = None

FINISH_REASONS = {
    "stop": InferenceFinishReason.Stop,
    "length": InferenceFinishReason.Length,
    "tool_calls": InferenceFinishReason.ToolCalls,
    "content_filter": InferenceFinishReason.ContentFilter,
}


async def context_limit() -> int:
    """Return ``max_model_len`` from the backend's model listing."""
    global _context_limit
    if _context_limit is None:
        if override := os.environ.get("VLLM_MAX_MODEL_LEN"):
            _context_limit = int(override)
        else:
            response = await CLIENT.get("/v1/models")
            response.raise_for_status()
            models = response.json().get("data") or []
            if not models or not models[0].get("max_model_len"):
                raise RuntimeError("backend /v1/models does not expose max_model_len")
            _context_limit = int(models[0]["max_model_len"])
    return _context_limit


async def vllm_inference(request: main.PreparedRequest) -> AsyncIterator[InferenceChunk]:
    prompt = ENCODING.encode(request.conversation_request.conversation)
    # Image placeholders occupy one token each; the backend expands them into
    # the image span and may prepend alignment pads.
    prompt_tokens = len(prompt) + request.image_token_adjustment
    prompt_tokens += (COMPRESS_PAD_TO - 1) * len(request.images)
    options = request.inference_options
    body = {
        "model": VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": request.conversation_request.resolve_max_tokens(
            prompt_tokens, await context_limit()
        ),
        "temperature": options.temperature,
        "top_p": options.top_p,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    if TOKENIZER is not None:
        # Token chunks let the stream processor segment thinking vs content and
        # report completion_tokens_details.reasoning_tokens.
        body["return_token_ids"] = True
    if options.seed is not None:
        body["seed"] = options.seed
    if options.output_constraint is not None:
        body["structured_outputs"] = structured_output_constraint(options.output_constraint)
    if request.images:
        body["multi_modal_data"] = {
            "image": [
                {"data": base64.b64encode(image.data).decode("ascii")}
                for image in request.images
            ]
        }
    async with CLIENT.stream("POST", "/v1/completions", json=body) as response:
        if response.status_code != 200:
            detail = (await response.aread()).decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("message") or detail
            except (ValueError, AttributeError):
                pass
            raise main.RequestError(detail, response.status_code)
        ready = False
        counted = 0
        finish = None
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("object") == "error" or "error" in event:
                raise RuntimeError(f"vLLM stream error: {data}")
            usage = event.get("usage") or {}
            if not ready:
                yield InferenceChunk.ready(
                    prompt_usage=PromptUsage(prompt_tokens=prompt_tokens),
                    system_fingerprint=SYSTEM_FINGERPRINT,
                )
                ready = True
            if isinstance(details := usage.get("prompt_tokens_details"), dict):
                request.late_usage.prompt_cache_hit_tokens = (
                    details.get("cached_tokens") or 0
                )
            completion = usage.get("completion_tokens", counted)
            for choice in event.get("choices") or []:
                token_ids = choice.get("token_ids")
                if token_ids and TOKENIZER is not None:
                    token_ids = [
                        token for token in token_ids if token != EOS_TOKEN_ID
                    ]
                    if token_ids:
                        yield InferenceChunk.tokens(token_ids)
                else:
                    yield InferenceChunk.text(
                        choice.get("text") or "",
                        token_count=completion - counted,
                    )
                counted = completion
                if (reason := choice.get("finish_reason")) is not None:
                    finish = FINISH_REASONS.get(reason, InferenceFinishReason.Stop)
        if finish is not None:
            yield InferenceChunk.finish(finish)


def structured_output_constraint(constraint) -> dict | None:
    """Translate an output constraint into vLLM structured-output parameters."""
    if constraint.kind == "json":
        return {"json_object": True} if constraint.schema == "" else {"json": json.loads(constraint.schema)}
    if constraint.kind == "regex":
        return {"regex": constraint.schema}
    return None


OPTIONS = main.conversion_options().with_model_alias(
    VLLM_MODEL, ModelVersion.DeepseekV41Flash
)
# The DeepSeek-V4.1-Flash checkpoint spells its image placeholder
# <｜deepseek_image｜>; the library default is <｜image｜>.
if IMAGE_PLACEHOLDER:
    OPTIONS = OPTIONS.with_special_tokens(SpecialTokens(image=IMAGE_PLACEHOLDER))

app = main.create_app(vllm_inference, options=OPTIONS, tokenizer=TOKENIZER)
