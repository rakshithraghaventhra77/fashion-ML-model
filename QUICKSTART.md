# Quick Start Checklist

Complete checklist to get the Fashion Recommendation System running locally.

## 🚀 Quick Start (5 minutes)

### Terminal 1: ML Service (Python)
```bash
cd ml-service
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # First time only
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# Wait for: "Uvicorn running on http://0.0.0.0:8000"
```

### Terminal 2: API Gateway (Spring Boot)
```bash
cd api-gateway
.\mvnw.cmd spring-boot:run
# Wait for: "Started DemoApplication in X seconds"
```

### Terminal 3: Frontend (React)
```bash
cd frontend
npm install  # First time only
npm run dev
# Open http://localhost:5173 in browser
```

## ✅ System Ready!

**If you see:**
- ML Service: ✓ Listening on port 8000
- Spring Boot: ✓ Listening on port 8080
- React: ✓ Listening on port 5173

Then you're ready to test!

## 🧪 Test the System

1. **Open browser**: http://localhost:5173
2. **Drag/click** to upload an image
3. **Click** "Get Recommendations" button
4. **See** 5 recommended similar items

## 📋 Setup Checklist (First Time Only)

- [ ] Python virtual environment activated (ml-service)
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Node.js dependencies installed in frontend (`npm install`)
- [ ] Maven dependencies downloaded (Spring Boot)

## 🔧 Configuration

### ML Service Port
```python
# In ml-service terminal:
python -m uvicorn app.main:app --port 8001  # Change to 8001
```

### Spring Boot Port & ML Service URL
```properties
# Edit: api-gateway/src/main/resources/application.properties
server.port=8081  # If port 8080 is taken
ml.service.url=http://localhost:8001  # If ML service port changed
```

### Frontend Port
```bash
# In frontend terminal:
npm run dev -- --port 3000  # Change to 3000
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Port already in use** | Change port in command (see above) |
| **ML service says "Model not loaded"** | Wait 30 seconds, models are loading. Check logs. |
| **Frontend can't call API** | Ensure Spring Boot on 8080, restart frontend |
| **"File too large" error** | Image must be under 5MB |
| **"Invalid file type" error** | Must be JPEG, PNG, GIF, or WebP |

## 📚 Documentation

- **Frontend**: `frontend/README.md` - Component structure, styling
- **Backend**: `api-gateway/BACKEND_GUIDE.md` - Controller flow, validation
- **Integration**: `INTEGRATION_GUIDE.md` - Full system architecture
- **Setup Guide**: `frontend/SETUP.md` - Detailed frontend setup

## 💡 Key Files to Review

### Frontend Entry Points
```
frontend/src/
├── main.jsx          # React entry point
├── App.jsx           # Main component, API calls
├── components/
│   ├── UploadBox.jsx
│   ├── ImagePreview.jsx
│   └── RecommendationGrid.jsx
```

### Backend Entry Points
```
api-gateway/src/main/java/com/example/demo/
├── DemoApplication.java
├── controller/
│   └── ImageController.java    # /api/recommend endpoint
└── dto/
    └── RecommendationResponse.java  # Response mapping
```

## 🎓 Interview Prep

### Code You'll Explain
1. **App.jsx** - State management, API calls, user flow
2. **ImageController.java** - File validation, microservice call, error handling
3. **RecommendationResponse.java** - DTO pattern, JSON mapping

### Questions You'll Get
- "How do you handle file uploads?"
- "How do you validate user input?"
- "How does the API Gateway communicate with ML service?"
- "What happens if ML service is down?"
- "How do you make it production-ready?"

## 🚢 Production Checklist

- [ ] Environment variables for ML service URL (don't hardcode)
- [ ] HTTPS for frontend-to-backend communication
- [ ] Input validation on both frontend AND backend
- [ ] Structured logging with proper levels
- [ ] Error monitoring and alerting
- [ ] Rate limiting to prevent abuse
- [ ] Docker containerization (already has Dockerfiles)
- [ ] Load balancing if scaling
- [ ] Database for storing recommendations history
- [ ] API documentation (Swagger/OpenAPI)

## 📞 Support

If something doesn't work:

1. **Check logs** - Most issues visible in terminal output
2. **Check ports** - Ensure no port conflicts
3. **Restart service** - Kill process, start again
4. **Clear cache** - Browser: Ctrl+Shift+Delete, Frontend: Ctrl+C then `npm install`
5. **Read documentation** - `INTEGRATION_GUIDE.md` has troubleshooting section

## ✨ What's Implemented

### Frontend (New)
- React 18 with Vite
- Tailwind CSS styling
- 4 reusable components
- File upload with validation
- Error handling
- Responsive design

### Backend (Improved)
- RecommendationResponse DTO
- File validation (size, type)
- Comprehensive logging
- Better error handling
- Code comments for learning

### Documentation (New)
- This checklist
- Integration guide
- Backend guide
- Frontend README
- Setup instructions

---

**You're all set! Start the services and explore the code.** 🎉
