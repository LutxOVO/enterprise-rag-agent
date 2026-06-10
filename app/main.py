from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse

from app.api.routes import router
from app.core.config import settings
from app.storage.database import init_db


STATIC_DIR = Path(__file__).resolve().parent / "static"


app = FastAPI(
    title=settings.app_name,
    description="A beginner-friendly FastAPI + LangChain + LangGraph RAG Agent demo.",
    version="0.1.0",
    docs_url=None,
)


@app.on_event("startup")
def on_startup() -> None:
    """Create local data folders and SQLite tables before serving requests."""
    settings.ensure_dirs()
    init_db()


app.include_router(router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def web_console() -> FileResponse:
    """Serve the lightweight RAG console frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html() -> HTMLResponse:
    """Swagger UI with a high-visibility cursor for classroom demos."""
    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{settings.app_name} - API Docs",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )
    html = response.body.decode("utf-8")
    cursor_css = """
    <style>
      html, body, .swagger-ui, .swagger-ui * {
        cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 48 48'%3E%3Cpath d='M8 4 L38 27 L24 30 L18 44 L8 4 Z' fill='white' stroke='black' stroke-width='3'/%3E%3C/svg%3E") 2 2, auto !important;
      }

      a, button, input, select, textarea, label,
      .opblock-summary, .btn, .try-out__btn, .execute,
      .copy-to-clipboard, .download-contents {
        cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 48 48'%3E%3Cpath d='M17 4 C14 4 12 6 12 9 V25 L9 22 C7 20 4 20 3 22 C2 24 2 26 4 28 L16 42 C18 44 20 45 23 45 H32 C38 45 42 41 42 35 V21 C42 18 40 16 37 16 C36 16 35 16 34 17 C33 15 31 14 29 14 C28 14 27 14 26 15 C25 13 23 12 21 12 C20 12 19 12 18 13 V9 C18 6 20 4 17 4 Z' fill='%23ffe66d' stroke='black' stroke-width='3'/%3E%3C/svg%3E") 6 4, pointer !important;
      }
    </style>
    """
    html = html.replace("</head>", f"{cursor_css}</head>")
    return HTMLResponse(html)
