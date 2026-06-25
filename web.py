"""
Веб-версия AI-чата. Бэкенд на FastAPI.
Groq (бесплатные модели) + Google Gemini (премиум, родной API, ключи AIza/AQ.).

Ключи живут на сервере. Gemini-ключ владельца (GEMINI_API_KEY) — необязательный:
если задан, премиум работает у всех посетителей по нему; иначе посетитель
вставляет свой ключ в настройках.
"""

import os
import json
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_NATIVE_BASE = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
).rstrip("/")

MAX_TOKENS = 2048
MAX_MESSAGES = 24
MAX_CONTENT = 8000

MODELS = {
    "llama-3.3-70b-versatile": {"provider": "groq", "label": "Llama 3.3 70B"},
    "llama-3.1-8b-instant":    {"provider": "groq", "label": "Llama 3.1 8B (быстро)"},
    "gemini-2.5-flash":        {"provider": "gemini", "label": "Gemini 2.5 Flash (премиум)"},
}

app = FastAPI(title="AI Chat")
HERE = Path(__file__).parent


class UpstreamError(Exception):
    def __init__(self, status: int, body: str = ""):
        self.status = status
        self.body = body


def sanitize(messages) -> list:
    out = []
    if isinstance(messages, list):
        for m in messages[-MAX_MESSAGES:]:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                out.append({"role": role, "content": content[:MAX_CONTENT]})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def to_gemini_contents(messages: list) -> list:
    return [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]


async def groq_stream(model: str, messages: list):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS, "stream": True}
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0)) as c:
        async with c.stream("POST", GROQ_URL, headers=headers, json=payload) as r:
            if r.status_code >= 400:
                raise UpstreamError(r.status_code, (await r.aread()).decode("utf-8", "ignore"))
            async for line in r.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    piece = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if piece:
                        yield piece
                except Exception:
                    continue


async def gemini_stream(model: str, messages: list, key: str):
    url = f"{GEMINI_NATIVE_BASE}/models/{model}:streamGenerateContent?alt=sse"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    payload = {
        "contents": to_gemini_contents(messages),
        "generationConfig": {"maxOutputTokens": MAX_TOKENS, "thinkingConfig": {"thinkingBudget": 0}},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0)) as c:
        async with c.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code >= 400:
                raise UpstreamError(r.status_code, (await r.aread()).decode("utf-8", "ignore"))
            async for line in r.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                    for part in obj["candidates"][0]["content"]["parts"]:
                        if part.get("text") and not part.get("thought"):
                            yield part["text"]
                except Exception:
                    continue


@app.get("/")
async def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/models")
async def models():
    return {"models": [{"id": k, "label": v["label"]} for k, v in MODELS.items()]}


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    model = body.get("model", "llama-3.3-70b-versatile")
    messages = sanitize(body.get("messages", []))
    gemini_key = (body.get("gemini_key") or body.get("om_key") or GEMINI_API_KEY or "").strip()

    info = MODELS.get(model)
    if not info:
        return JSONResponse({"error": "Неизвестная модель"}, status_code=400)
    if not messages:
        return JSONResponse({"error": "Пустой запрос"}, status_code=400)
    if info["provider"] == "gemini" and not gemini_key:
        return JSONResponse(
            {"error": "Для премиум-модели нужен ключ Gemini — вставь его в настройках (⚙️)."},
            status_code=400,
        )

    async def gen():
        try:
            if info["provider"] == "groq":
                async for t in groq_stream(model, messages):
                    yield t
            else:
                async for t in gemini_stream(model, messages, gemini_key):
                    yield t
        except UpstreamError as e:
            if e.status in (401, 403):
                yield "\n\n⚠️ Ключ недействителен."
            elif e.status == 429:
                yield "\n\n⏳ Превышен лимит Gemini, подожди минуту."
            else:
                yield f"\n\n⚠️ Ошибка модели ({e.status})."
        except Exception:
            yield "\n\n⚠️ Что-то пошло не так. Попробуй ещё раз."

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
