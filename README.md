# Fashion Recommendation System

An image-based fashion recommendation app built with three parts:
- Java Spring Boot API gateway
- Python FastAPI ML service
- Simple static frontend

## What it does
- Upload a fashion image in the browser.
- Send it through the gateway.
- Extract image embeddings with MobileNetV2.
- Search similar products with FAISS.
- Render recommendation cards with product metadata.

## Run the project

From the project root:

```powershell
docker compose up --build
```

Open:
- Frontend: http://localhost:3000
- API Gateway health: http://localhost:8080/api/health
- ML Service health: http://localhost:8000/health

## Project structure
- `api-gateway/` Spring Boot gateway and proxy logic
- `ml-service/` FastAPI service, model loading, and FAISS search
- `frontend/` clean upload-and-results UI
- `data/` images, metadata, and dataset files

## Current behavior
- The frontend is intentionally simple and keeps the same theme and fonts.
- The gateway retries briefly if the ML service is still warming up.
- The ML service returns enriched product metadata and image URLs.

## Interview prep
- See [INTERVIEW_PREP_GUIDE.md](INTERVIEW_PREP_GUIDE.md) for the development story, code walkthrough, and interview questions.

## Local development note
- For local non-Docker runs, set `ml.service.url` to `http://localhost:8000`.
- For Docker runs, keep it as `http://ml-service:8000`.
