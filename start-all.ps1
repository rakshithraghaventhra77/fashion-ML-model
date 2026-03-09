# Fashion Recommendation System - Start All Services
# This script starts ML Service, API Gateway, and Frontend in parallel

Write-Host "🚀 Starting Fashion Recommendation System..." -ForegroundColor Green
Write-Host "This will open 3 new terminals for each service`n" -ForegroundColor Yellow

$projPath = Get-Location

# Terminal 1: ML Service (Python)
Write-Host "📍 Starting ML Service on port 8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
  Set-Location '$projPath\ml-service'
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt 2>$null
  Write-Host '⏳ ML Service starting...' -ForegroundColor Yellow
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
"

Start-Sleep -Seconds 2

# Terminal 2: API Gateway (Spring Boot)
Write-Host "📍 Starting API Gateway on port 8080..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
  Set-Location '$projPath\api-gateway'
  Write-Host '⏳ API Gateway starting (this takes ~15 seconds)...' -ForegroundColor Yellow
  .\mvnw.cmd spring-boot:run
"

Start-Sleep -Seconds 2

# Terminal 3: Frontend (React)
Write-Host "📍 Starting Frontend on port 5173..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
  Set-Location '$projPath\frontend'
  npm install 2>$null
  Write-Host '⏳ Frontend starting...' -ForegroundColor Yellow
  npm run dev
"

Write-Host "`n✅ All services are starting..." -ForegroundColor Green
Write-Host "   ML Service:   http://localhost:8000" -ForegroundColor Green
Write-Host "   API Gateway:  http://localhost:8080" -ForegroundColor Green
Write-Host "   Frontend:     http://localhost:5173" -ForegroundColor Green
Write-Host "`n⏳ Wait 20-30 seconds for all services to be ready..." -ForegroundColor Yellow
Write-Host "Then open http://localhost:5173 in your browser`n" -ForegroundColor Yellow
