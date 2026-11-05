# vLLM backend for the Python example server

`server-py/vllm.py` completes the prompt that `server-py/main.py` renders on a
vLLM OpenAI server. It is the reference pattern for serving DeepSeek-V4.1-Flash
through this toolkit: conversion, prompt rendering, image preprocessing, and
response parsing stay in `deepseek_recipe`; vLLM only performs inference.

## Run

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install ./deepseek-recipe-python -r server-py/requirements.txt -r server-py/requirements-vllm.txt

VLLM_BASE_URL=http://127.0.0.1:8000 \
VLLM_MODEL=deepseek-flash \
TOKENIZER_PATH=/models/DeepSeek-V4.1-Flash/tokenizer.json \
SYSTEM_FINGERPRINT=my-backend \
python3 -m uvicorn vllm:app --app-dir server-py --host 127.0.0.1 --port 7777
```

Environment variables:

| variable | required | meaning |
|---|---|---|
| `VLLM_BASE_URL` | yes | vLLM OpenAI server root |
| `VLLM_MODEL` | yes | served model name |
| `TOKENIZER_PATH` | recommended | `tokenizer.json` used by the prompt encoding. Enables token-level streaming, thinking/content segmentation, and `completion_tokens_details.reasoning_tokens` |
| `VLLM_API_KEY` | no | bearer token for the backend |
| `VLLM_MAX_MODEL_LEN` | no | context length override; otherwise read from `GET /v1/models` |
| `SYSTEM_FINGERPRINT` | no | value echoed as `system_fingerprint` |
| `IMAGE_PLACEHOLDER` | no | image placeholder spelling, default `<｜deepseek_image｜>` |

Requests reach the backend as `POST /v1/completions` with a token-id prompt.
The backend streams `token_ids` when `TOKENIZER_PATH` is set (the example server
then reports `reasoning_tokens` and segments thinking from content), and falls
back to vLLM's text stream otherwise.

## Images

Images are sent as `multi_modal_data` on the completion request, using the bytes
of the preprocessed `ImageInfo` objects produced by `deepseek_recipe`, so the
token grid matches the reference image processor.

Stock vLLM does not accept `multi_modal_data` on `/v1/completions`; the DeepSeek
V4.1 vision deployments add it in the OpenAI frontend:

```python
# entrypoints/openai/completion/protocol.py
class CompletionRequest(OpenAIBaseModel):
    ...
    multi_modal_data: dict[str, Any] | None = None

# renderers/online_renderer.py, in render_completion():
if request.multi_modal_data:
    engine_input = await self.renderer.process_for_engine_async(
        TokensPrompt(
            prompt_token_ids=list(request.prompt),
            multi_modal_data=decoded_images,
        ),
        time.time(),
    )
    return [engine_input]
```

`decoded_images` is `{"image": [PIL.Image, ...]}` built from the base64 payload.
Without that patch, run the backend text-only.

## Limitations

- `image_token_adjustment` from the library does not include the backend's
  compressor-alignment pads, so this example budgets `(PAD - 1)` tokens per image
  when resolving `max_tokens`. A backend that can report the exact prompt token
  count after generation should override `PreparedRequest.late_usage.prompt_tokens`.
- Streaming token decoding relies on the same tokenizer used for encoding; keep
  them on the same checkpoint revision.
