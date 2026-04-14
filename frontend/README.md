# Frontend Build Guide

This frontend is intentionally small: plain HTML, CSS, and JavaScript with the same theme and fonts as before.

## What it does
- Uploads one fashion image.
- Shows a preview.
- Sends the file to the API gateway.
- Renders recommendation cards from the ML service.

## Build flow
1. `index.html` for structure.
2. `styles.css` for the theme.
3. `app.js` for upload, preview, request, and render logic.
4. `Dockerfile` and `nginx.conf` for static hosting.
5. `docker-compose.yml` to start it with the backend.

## Run it

From the project root:

```powershell
docker compose up --build
```

Then open:
- Frontend: http://localhost:3000
- Gateway health: http://localhost:8080/api/health
- ML health: http://localhost:8000/health

## UI flow
- Choose an image.
- Preview it.
- Click Analyze image.
- View the recommended product cards.

## Files
- `index.html` for layout.
- `styles.css` for the dark theme and cards.
- `app.js` for the interaction logic.
- `Dockerfile` for the container.
- `nginx.conf` for static serving.
