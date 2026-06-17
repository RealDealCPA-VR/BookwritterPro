"""FastAPI application: JSON API under /api + static frontend at /.

All routes match the authoritative HTTP contract. The SSE endpoint is
hand-rolled (no sse-starlette): a StreamingResponse over an async generator that
replays the broker ring buffer, then tails an asyncio.Queue fed thread-safely by
the background write job, emitting a periodic comment heartbeat.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .broker import EventBroker, TERMINAL_TYPES
from .schemas import (
    CreateBookRequest,
    KdpRequest,
    MarketingRequest,
    PricingRequest,
    WriteRequest,
)
from .service import BookService, ServiceError

# Heartbeat interval (seconds) for the SSE stream.
_HEARTBEAT = 15.0

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def _default_data_dir() -> str:
    env = os.environ.get("BOOKWRITER_DATA_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, ".bookwriter_data")


def create_app(data_dir: str | None = None) -> FastAPI:
    broker = EventBroker()
    service = BookService(data_dir or _default_data_dir(), broker=broker)

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Bind the running loop so worker threads can schedule queue puts.
        broker.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="BookwriterPro", version="0.1.0", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- error handling -------------------------------------------------
    @app.exception_handler(ServiceError)
    async def _service_error(_req: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    # ================================================================== #
    # API routes
    # ================================================================== #
    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        from ..provider import provider_name
        return {
            "status": "ok",
            "has_api_key": service.has_api_key(),
            "provider": provider_name(),
        }

    @app.get("/api/profiles")
    async def profiles() -> Dict[str, Any]:
        return service.profiles()

    @app.get("/api/providers")
    async def providers() -> Dict[str, Any]:
        from ..provider import provider_catalog
        from ..images import image_status
        cat = provider_catalog()
        cat["image"] = image_status()  # which image backend is active + usable
        return cat

    @app.get("/api/books")
    async def list_books() -> Dict[str, Any]:
        return service.list_books()

    @app.post("/api/books")
    async def create_book(req: CreateBookRequest) -> Dict[str, Any]:
        # Planning is synchronous and may call the model; run off the event loop.
        return await asyncio.to_thread(service.create_book, req)

    @app.get("/api/books/{book_id}")
    async def get_book(book_id: str) -> Dict[str, Any]:
        return service.get_book(book_id)

    @app.post("/api/books/{book_id}/write")
    async def write_book(book_id: str, req: WriteRequest) -> Dict[str, Any]:
        return service.start_write(book_id, req)

    @app.get("/api/books/{book_id}/chapters/{n}")
    async def get_chapter(book_id: str, n: int) -> Dict[str, Any]:
        return service.get_chapter(book_id, n)

    @app.get("/api/books/{book_id}/chapters/{n}/image")
    async def get_chapter_image(book_id: str, n: int) -> Response:
        path, media = service.get_chapter_image(book_id, n)
        return FileResponse(path, media_type=media)

    @app.get("/api/books/{book_id}/graph")
    async def get_graph(book_id: str) -> Dict[str, Any]:
        return service.get_graph(book_id)

    @app.get("/api/books/{book_id}/cost")
    async def get_cost(book_id: str) -> Dict[str, Any]:
        return service.get_cost(book_id)

    @app.get("/api/books/{book_id}/manuscript")
    async def get_manuscript(book_id: str, download: int = Query(0)) -> Any:
        data = service.get_manuscript(book_id)
        if download:
            try:
                meta = service._require_meta(book_id)  # noqa: SLF001 - internal helper
                fname = (meta.get("title") or book_id) or book_id
            except ServiceError:
                fname = book_id
            safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in fname).strip() or book_id
            return Response(
                content=data["markdown"],
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{safe}.md"'},
            )
        return data

    # -- KDP ------------------------------------------------------------
    @app.post("/api/books/{book_id}/kdp")
    async def prepare_kdp(book_id: str, req: KdpRequest) -> Dict[str, Any]:
        # Generation may call the model + writes the kit; run off the event loop.
        return await asyncio.to_thread(service.prepare_kdp, book_id, req)

    @app.get("/api/books/{book_id}/kdp")
    async def get_kdp(book_id: str) -> Dict[str, Any]:
        return {"metadata": service.get_kdp(book_id)}

    @app.get("/api/books/{book_id}/kdp/listing")
    async def get_kdp_listing(book_id: str) -> Response:
        return Response(
            content=service.kdp_listing(book_id),
            media_type="text/plain; charset=utf-8",
        )

    @app.get("/api/books/{book_id}/export/epub")
    async def export_epub(book_id: str) -> FileResponse:
        path = await asyncio.to_thread(service.epub_path, book_id)
        fname = service.epub_filename(book_id)
        return FileResponse(
            path,
            media_type="application/epub+zip",
            filename=fname,
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    # -- Print / pricing / marketing -----------------------------------
    @app.get("/api/books/{book_id}/export/docx")
    async def export_docx(book_id: str) -> FileResponse:
        path = await asyncio.to_thread(service.export_docx_path, book_id)
        fname = service.docx_filename(book_id)
        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=fname,
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    @app.get("/api/books/{book_id}/print")
    async def get_print(book_id: str) -> Dict[str, Any]:
        return {"spec": service.print_spec(book_id)}

    @app.api_route("/api/books/{book_id}/export/print-cover", methods=["GET", "HEAD"])
    async def export_print_cover(book_id: str) -> Response:
        svg = await asyncio.to_thread(service.print_cover_svg, book_id)
        fname = service.docx_filename(book_id).replace(".docx", "-print-cover.svg")
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Content-Disposition": f'inline; filename="{fname}"'},
        )

    @app.post("/api/books/{book_id}/pricing")
    async def pricing(book_id: str, req: PricingRequest) -> Dict[str, Any]:
        return {"pricing": service.estimate_pricing(book_id, req)}

    @app.post("/api/books/{book_id}/marketing")
    async def marketing(book_id: str, req: MarketingRequest) -> Dict[str, Any]:
        # Generation may call the model + writes marketing.json; run off the loop.
        result = await asyncio.to_thread(service.generate_marketing, book_id, req)
        return {"marketing": result}

    @app.delete("/api/books/{book_id}")
    async def delete_book(book_id: str) -> Dict[str, Any]:
        return service.delete_book(book_id)

    # -- SSE ------------------------------------------------------------
    @app.get("/api/books/{book_id}/events")
    async def events(book_id: str, request: Request) -> StreamingResponse:
        # 404 if the book does not exist.
        service._require_meta(book_id)  # noqa: SLF001 - raises ServiceError -> handled

        q, replay, finished = broker.subscribe(book_id)

        async def gen():
            try:
                # Replay everything emitted so far for the current/last job.
                saw_terminal = False
                for ev in replay:
                    yield _sse(ev)
                    if ev.get("type") in TERMINAL_TYPES:
                        saw_terminal = True
                # If the last job already finished and we replayed its terminal
                # event, we are done — close cleanly.
                if finished and saw_terminal:
                    return
                # Tail live events with a heartbeat so proxies keep the stream open.
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=_HEARTBEAT)
                    except asyncio.TimeoutError:
                        yield ":\n\n"  # comment heartbeat
                        continue
                    yield _sse(ev)
                    if ev.get("type") in TERMINAL_TYPES:
                        return
            finally:
                broker.unsubscribe(book_id, q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ================================================================== #
    # Static frontend
    # ================================================================== #
    @app.get("/")
    async def index() -> Any:
        idx = os.path.join(_WEB_DIR, "index.html")
        if os.path.isfile(idx):
            return FileResponse(idx)
        return JSONResponse(
            status_code=200,
            content={"detail": "Frontend not built yet. API is live under /api."},
        )

    if os.path.isdir(_WEB_DIR):
        # Serve styles.css, app.js, and any other assets from /.
        app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")

    return app


def _sse(event: Dict[str, Any]) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
