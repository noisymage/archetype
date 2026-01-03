# Architecture Pivot: Enriched Metadata + Assisted Curation

## Summary

Pivot from ML-based validation (pass/fail) to **enriched metadata** that powers assisted curation. ML scores become informational data points. Vision LLM generates rich descriptions and captions during batch processing.

---

## Changes Overview

### Database Schema

**New `ImageDescription` table** for rich textual descriptions from vision LLM:
- Shot type, pose, expression, clothing, lighting, background
- Full description and quality notes
- LLM provider/model metadata

**Updated `CaptionModelType` enum**: SDXL, Flux, Qwen-Image, Z-Image

### LLM Integration

**New `llm_engine.py`** with provider abstraction:
- `OllamaProvider` for local models
- `GeminiProvider` for cloud (with Files API caching)
- Prompts loaded from external `prompts.yaml`

### Settings

**New `settings.json`** (gitignored) for:
- LLM provider selection
- API keys and model configuration

**Settings UI** via gear icon in sidebar

### Processing Pipeline

Batch processing extended with enrichment phase:
1. ML extraction (face, pose) → metrics
2. LLM enrichment → descriptions + captions
3. Store all results

### Cleanup

Remove dead SMPL-X code from `vision_engine.py`

---

## Files to Modify/Create

| Action | File |
|--------|------|
| MODIFY | `backend/database.py` |
| MODIFY | `backend/batch_processor.py` |
| MODIFY | `backend/main.py` |
| MODIFY | `backend/vision_engine.py` |
| NEW | `backend/llm_engine.py` |
| NEW | `backend/prompts.yaml` |
| NEW | `backend/settings.json` (gitignored) |
| NEW | `frontend/src/components/SettingsModal.jsx` |
| MODIFY | `frontend/src/components/Sidebar.jsx` |
| MODIFY | `.gitignore` |

---

## Verification

1. Schema migration: new tables created
2. Settings UI: configure LLM via gear icon
3. Enrichment: process images, verify descriptions stored
4. Export: captions export correctly
