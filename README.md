# Fashion Recommendation - ML Only (Fresh Start)

This repository has been reset to keep only the machine learning part of the project.

## Current Structure

- `ml-service/` - FastAPI service, model loading, embedding generation, and similarity search
- `data/` - Dataset files used for recommendations
- `.venv/` - Python virtual environment

## What Was Removed

- Frontend code
- Backend/API gateway code
- Docker integration
- Full-stack integration guides and startup scripts

## Run ML Service

From the repository root:

```powershell
& ".\.venv\Scripts\Activate.ps1"
Set-Location .\ml-service
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

## Next Learning Path

1. Understand `ml-service/app/main.py` endpoints.
2. Study `embedding_generator.py` and `model_loader.py`.
3. Learn `search_engine.py` and FAISS index usage.
4. Rebuild backend integration from scratch when ready.
5. Rebuild frontend integration from scratch when ready.
