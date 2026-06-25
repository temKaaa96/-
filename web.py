"""
Веб-версия AI-чата. Бэкенд на FastAPI.

Зачем нужен бэкенд: API-ключи (Groq, OpenModel) должны храниться на сервере,
а не в коде страницы — иначе их украдут. Этот сервер держит ключи у себя и
проксирует запросы к моделям, отдавая ответ браузеру по мере генерации (стрим).

Запуск (локально):  uvicorn web:app --reload
Запуск (Railway):   uvicorn web:app --host 0.0.0.0 --port $PORT
"""

import os
import json
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

# ─── Ключи и эндпоинты ───────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Необязательный «ключ по умолчанию» владельца для DeepSeek. Если задать — DeepSeek
# на сайте будет работать у всех на твоём ключе (но это жжёт лимит 10 запросов/мин
# на анонимный трафик). Безопаснее оставить пустым: тогда DeepSeek работает только
# когда посетитель сам вставил свой om-ключ в настройках.
OPENMODEL_KEY_DEFAULT = os.environ.get("OPENMODEL_KEY", "").strip()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENMODEL_URL = "https://api.openmodel.ai/v1/messages"
MAX_TOKENS = 2048
MAX_MESSAGES = 24
MAX_CONTENT = 8000

# Какие модели доступны на сайте
MODELS = {
    "llama-3.3-70b-versatile": {"provider": "groq", "label": "Llama 3.3 70B"},
    "llama-3.1-8b-instant":    {"provider": "groq", "label": "Llama 3.1 8B (быстро)"},
    "deepseek-v4-flash":       {"provider": "openmodel", "label": "DeepSeek V4 Flash"},
}

app = FastAPI(title="AI Chat")
HERE = Path(__file__).parent


class UpstreamError(Exception):
    def __init__(self, status: int, body: str = ""):
        self.status = status
        self.body = body


def sanitize(messages) -> list:
    """Чистим вход: только корректные роли, обрезаем длину, диалог начинается с user."""
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


async def groq_stream(model: str, messages: list):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS, "stream": True}
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0)) as client:
        async with client.stream("POST", GROQ_URL, headers=headers, json=payload) as r:
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
                    piece = json.loads(data)["choices"][0]["delta"].get("content")
                    if piece:
                        yield piece
                except Exception:
                    continue


async def openmodel_stream(model: str, messages: list, key: str):
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {"model": model, "max_tokens": MAX_TOKENS, "stream": True, "messages": messages}
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0)) as client:
        async with client.stream("POST", OPENMODEL_URL, headers=headers, json=payload) as r:
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
                except Exception:
                    continue
                if obj.get("type") == "content_block_delta":
                    d = obj.get("delta", {})
                    if d.get("type") == "text_delta" and d.get("text"):
                        yield d["text"]
                elif obj.get("type") == "message_stop":
                    break


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
    om_key = (body.get("om_key") or OPENMODEL_KEY_DEFAULT or "").strip()

    info = MODELS.get(model)
    if not info:
        return JSONResponse({"error": "Неизвестная модель"}, status_code=400)
    if not messages:
        return JSONResponse({"error": "Пустой запрос"}, status_code=400)
    if info["provider"] == "openmodel" and not om_key:
        return JSONResponse(
            {"error": "Для DeepSeek нужен om-ключ OpenModel — вставь его в настройках (⚙️)."},
            status_code=400,
        )

    async def gen():
        try:
            if info["provider"] == "groq":
                async for t in groq_stream(model, messages):
                    yield t
            else:
                async for t in openmodel_stream(model, messages, om_key):
                    yield t
        except UpstreamError as e:
            if e.status in (401, 403):
                yield "\n\n⚠️ Ключ недействителен."
            elif e.status == 429:
                yield "\n\n⏳ Слишком много запросов, подожди минуту."
            else:
                yield f"\n\n⚠️ Ошибка модели ({e.status})."
        except Exception:
            yield "\n\n⚠️ Что-то пошло не так. Попробуй ещё раз."

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
