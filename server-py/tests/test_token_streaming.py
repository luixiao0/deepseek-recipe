"""Token-level streaming through the example server.

The stream processor can decode `tokens` chunks when `create_app()` receives a
tokenizer; without it, `tokens` chunks fail (the documented default).
"""

import importlib.util
import json
from pathlib import Path

from deepseek_recipe import (
    InferenceChunk,
    InferenceFinishReason,
    PromptUsage,
    Tokenizer,
)
from fastapi.testclient import TestClient

SPEC = importlib.util.spec_from_file_location(
    "recipe_server_example_token", Path(__file__).resolve().parents[1] / "main.py"
)
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)

TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]": 0, "Hello": 1, "world": 2, "!": 3},
        "unk_token": "[UNK]",
    },
}


async def token_inference(request):
    yield InferenceChunk.ready(prompt_usage=PromptUsage(prompt_tokens=1))
    yield InferenceChunk.tokens([1, 2, 3])
    yield InferenceChunk.finish(InferenceFinishReason.Stop)


def test_tokens_chunks_need_a_tokenizer():
    app = SERVER.create_app(token_inference)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code >= 500


def test_tokens_chunks_stream_with_a_tokenizer():
    tokenizer = Tokenizer.from_str(json.dumps(TOKENIZER_JSON))
    app = SERVER.create_app(token_inference, tokenizer=tokenizer)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    message = body["choices"][0]["message"]
    text = (message.get("content") or "") + (message.get("reasoning_content") or "")
    assert "Hello" in text
    assert "world" in text
    # Token chunks let the stream processor count thinking tokens.
    usage = body["usage"]
    assert usage["completion_tokens"] == 3
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 3
