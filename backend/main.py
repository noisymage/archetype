"""
FastAPI application entry point for Character Consistency Validator.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - initialize database on startup."""
    init_db()
    yield


app = FastAPI(
    title="Archetype - Character Consistency Validator",
    description="API for validating and curating datasets for LoRA training",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint for frontend connectivity verification."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Archetype API - Character Consistency Validator"}
