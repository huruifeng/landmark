"""
FastAPI application entry point.

Start (from landmark/backend/):
    uvicorn main:app --reload --port 8000
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # must be before any torch import

import asyncio
from collections import Counter
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from state import app_state
from services.loader import (
    get_db_embeddings,
    get_image_dir,
    get_transform,
    load_csv_lists,
    load_model,
)
from routers import images, retrieve, status


async def _initialize():
    """Run heavy startup work in a thread so the server is immediately reachable."""
    loop = asyncio.get_event_loop()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── 1. Load model ────────────────────────────────────────────────────────
    app_state.status   = "loading_model"
    app_state.progress = 0
    app_state.message  = "Loading model checkpoint…"
    model = await loop.run_in_executor(None, load_model, device)

    # ── 2. Load CSVs ─────────────────────────────────────────────────────────
    app_state.message = "Reading dataset files…"
    db_files, db_labels, val_files, val_labels = await loop.run_in_executor(None, load_csv_lists)

    # ── 3. Build / load embeddings ───────────────────────────────────────────
    app_state.status   = "building_embeddings"
    app_state.progress = 0
    app_state.message  = f"Building image embeddings… On device: {device}"
    transform = get_transform()

    def on_batch(current: int, total: int):
        app_state.progress = int(current / total * 100)
        app_state.message  = f"Building embeddings: {current} / {total} batches"

    db_emb = await loop.run_in_executor(
        None, get_db_embeddings, model, device, db_files, transform, on_batch
    )

    # ── 4. Populate shared state ─────────────────────────────────────────────
    app_state.model     = model
    app_state.device    = device
    app_state.transform = transform
    app_state.db_emb    = db_emb
    app_state.db_files  = db_files
    app_state.db_labels = db_labels
    app_state.val_files        = val_files
    app_state.val_labels       = val_labels
    app_state.val_file_to_label = dict(zip(val_files, val_labels))
    app_state.db_label_counts  = Counter(db_labels)
    app_state.image_dir = get_image_dir()

    app_state.status   = "ready"
    app_state.progress = 100
    app_state.message  = f"Ready — {len(val_files)} val images, {len(db_files)} db images."
    print(app_state.message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_initialize())   # non-blocking: server accepts requests immediately
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(get_image_dir())), name="images")

app.include_router(status.router)
app.include_router(images.router)
app.include_router(retrieve.router)
