"""
FastAPI backend for nepalgov.datacortex.in.

Serves the static frontend and exposes /api endpoints that the JS calls.
Boots the RAG corpus in a background thread so the HTTP port opens
immediately — Render's port scanner needs to see a listening port within
~10 minutes, and indexing the corpus can take longer on free-tier CPU.

The frontend's /api/health poller handles the warming state.

Local:
    pip install -e ".[web]"
    python web/app.py
    # → http://127.0.0.1:8000

Render:
    Build:  pip install -e ".[web]"
    Start:  python web/app.py
    Port comes from $PORT.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

logger = logging.getLogger(__name__)

# Module-level state. Render workers are single-process so this is fine.
_state: dict[str, Any] = {"rag": None, "agent": None, "ready": False, "error": None}
_boot_lock = threading.Lock()
_boot_started = False


# ---------------------------------------------------------------------------
# Boot: runs in a background thread so HTTP port opens immediately
# ---------------------------------------------------------------------------

def _boot() -> None:
    """Build the RAG index. Can take 60-120s on Render free-tier CPU."""
    t0 = time.time()
    try:
        from nepal_gov_agent import GovAgent, GovRAG, GovRAGConfig, download_corpus

        corpus_dir = os.environ.get("NGA_CORPUS_DIR", "").strip()
        if not corpus_dir:
            logger.info("Downloading seed corpus...")
            corpus_dir = download_corpus(dest_dir="./nepal_gov_data/")

        cache_dir = os.environ.get("NGA_CACHE_DIR", ".nepal_gov_cache")
        config = GovRAGConfig(cache_dir=cache_dir)
        logger.info("Building RAG index from %s...", corpus_dir)
        rag = GovRAG(corpus_dir=corpus_dir, config=config)
        agent = GovAgent(rag=rag, session_id="web")

        _state["rag"] = rag
        _state["agent"] = agent
        _state["ready"] = True
        logger.info("Boot complete in %.1fs: %s", time.time() - t0, rag.stats)
    except Exception as exc:
        _state["error"] = repr(exc)
        logger.exception("Boot failed")


def _kick_boot_once() -> None:
    """Start _boot() in a background thread, only once per process."""
    global _boot_started
    with _boot_lock:
        if _boot_started:
            return
        _boot_started = True
    t = threading.Thread(target=_boot, name="nga-boot", daemon=True)
    t.start()
    logger.info("Boot thread started; HTTP server is ready to accept traffic.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nepal GovAgent",
    description="Reference RAG implementation over Nepal's policy and legal corpus.",
    docs_url=None,  # hide swagger; this isn't a public API
    redoc_url=None,
)


@app.on_event("startup")
async def _on_startup() -> None:
    # Critical: do NOT block here. Render's port scanner needs to see the
    # HTTP port open within ~10 minutes. The corpus build runs in a thread.
    _kick_boot_once()

WEB_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=12)


class Source(BaseModel):
    doc: str
    page: Any  # can be int or "?"
    heading: Optional[str] = None
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    confidence: str
    sources: list[Source]
    elapsed_ms: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health() -> dict[str, Any]:
    rag = _state["rag"]
    out: dict[str, Any] = {
        "ready": _state["ready"],
        "error": _state["error"],
    }
    if rag is not None:
        out["stats"] = rag.stats
    return out


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not _state["ready"]:
        # 503 is what front-end polls on
        raise HTTPException(status_code=503, detail="Service warming up")
    rag = _state["rag"]
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG unavailable")

    t0 = time.time()
    try:
        result = rag.ask(req.query, k_final=req.k)
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed_ms = int((time.time() - t0) * 1000)

    sources = [
        Source(
            doc=str(s.get("doc", "?")),
            page=s.get("page", "?"),
            heading=s.get("heading"),
            excerpt=(s.get("excerpt") or "").strip(),
        )
        for s in (result.sources or [])[:8]
    ]
    return AskResponse(
        answer=result.answer or "",
        confidence=result.confidence or "unknown",
        sources=sources,
        elapsed_ms=elapsed_ms,
    )


@app.get("/api/stats")
async def stats() -> JSONResponse:
    rag = _state["rag"]
    if rag is None:
        return JSONResponse({"ready": False}, status_code=503)
    return JSONResponse({"ready": True, **rag.stats})


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips="*",  # Render is behind a proxy
    )


if __name__ == "__main__":
    main()
