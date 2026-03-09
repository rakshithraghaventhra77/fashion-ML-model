# Full-Stack Integration Guide

Complete setup and running instructions for the Fashion Recommendation System.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ User Browser (http://localhost:5173)                             │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ React Frontend (Vite + Tailwind)                             │ │
│ │  - App.jsx (Main state & logic)                              │ │
│ │  - UploadBox.jsx (File upload)                               │ │
│ │  - ImagePreview.jsx (Show preview)                           │ │
│ │  - RecommendationGrid.jsx (Show results)                     │ │
│ └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
            POST /api/recommend (multipart/form-data)
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ API Gateway (http://localhost:8080)                              │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Spring Boot 4.0.3                                            │ │
│ │  - ImageController (handles /api/recommend)                  │ │
│ │  - RecommendationResponse DTO (maps response)                │ │
│ │  - File validation (size, type, empty)                       │ │
│ │  - Error handling & logging                                  │ │
│ └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
                POST /recommend (multipart/form-data)
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ ML Microservice (http://localhost:8000)                          │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ FastAPI Python                                               │ │
│ │  - main.py (FastAPI app)                                     │ │
│ │  - embedding_generator.py (Extract features)                 │ │
│ │  - search_engine.py (FAISS similarity search)                │ │
│ │ Returns:                                                      │ │
│ │  {"success": true, "recommendations": [...]}                 │ │
│ └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Setup

### Step 1: Start ML Microservice

First, start the Python ML service:

```bash
# Navigate to ml-service directory
cd ml-service

# Activate Python virtual environment (if not already activated)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Expected output:
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Verify it's running:**
```bash
curl http://localhost:8000/health
# Response: {"status": "Service is running", "model_loaded": true}
```

### Step 2: Start Spring Boot API Gateway

In a new terminal:

```bash
# Navigate to api-gateway directory
cd api-gateway

# Build the project (first time only)
mvn clean package

# Run Spring Boot
mvn spring-boot:run

# Expected output:
# Tomcat started on port(s): 8080 (http)
# Started DemoApplication in X seconds
```

**Verify it's running:**
```bash
curl http://localhost:8080/api/health
# Should forward to FastAPI health check
```

### Step 3: Start React Frontend

In a new terminal:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Expected output:
# VITE v4.3.9  ready in XXX ms
# ➜  Local:   http://localhost:5173/
```

**Open in browser:** http://localhost:5173

## Testing the System

### Test 1: Upload and Get Recommendations

1. Open http://localhost:5173 in browser
2. Drag an image or click to upload
3. See image preview
4. Click "Get Recommendations" button
5. View 5 recommended similar items in grid

### Test 2: API Test with cURL

```bash
# Upload image to backend
curl -X POST -F "file=@path/to/image.jpg" \
  http://localhost:8080/api/recommend

# Response should be:
# {
#   "success": true,
#   "recommendations": [
#     {"imagePath": "data/images/image1.jpg", "score": 0.95},
#     ...
#   ]
# }
```

### Test 3: Direct ML Service Test

```bash
# Call ML service directly
curl -X POST -F "file=@path/to/image.jpg" \
  http://localhost:8000/recommend

# Response:
# {
#   "success": true,
#   "recommendations": [...]
# }
```

## Common Issues & Solutions

### Issue 1: Frontend can't reach API

**Error**: CORS blocked or connection refused

**Solution**:
1. Ensure Spring Boot runs on `localhost:8080`
2. Check Vite proxy in `frontend/vite.config.js`
3. Restart frontend dev server

**Debug**:
```bash
# Check backend is running
curl http://localhost:8080/api/health
# Should succeed
```

### Issue 2: ML Service returns 503 (Model not loaded)

**Error**: `{"detail": "Model not loaded"}`

**Solution**:
1. Check ML service logs - see if model loaded successfully
2. Verify `models/` directory contains model files
3. Check `embedding_generator.py` and `model_loader.py`
4. Restart ML service

**Debug**:
```bash
# Check ML health
curl http://localhost:8000/health
# Should show "model_loaded": true
```

### Issue 3: File validation fails

**Error**: "File size exceeds maximum" or "Invalid file type"

**Solution**:
1. Ensure file is JPEG, PNG, GIF, or WebP
2. Ensure file is under 5MB
3. Check frontend file selection logic

**Debug**:
```bash
# Test with curl
curl -X POST -F "file=@small_image.jpg" \
  http://localhost:8080/api/recommend
```

### Issue 4: Port already in use

**Error**: "Address already in use" or "EADDRINUSE"

**Solution**:
- **For ML service on port 8000:**
  ```bash
  python -m uvicorn app.main:app --port 8001
  ```
  Then update ImageController with new URL in `application.properties`

- **For Spring Boot on port 8080:**
  ```bash
  # In api-gateway/src/main/resources/application.properties
  server.port=8081
  ```

- **For Frontend on port 5173:**
  ```bash
  npm run dev -- --port 3000
  ```

### Issue 5: ClassNotFoundException or Dependency Error

**Error**: Spring Boot complains about missing classes

**Solution**:
```bash
# Clean and rebuild
mvn clean install
mvn spring-boot:run
```

## Code Explanation for Interviews

### Frontend Flow (React)

```javascript
// 1. User selects file
handleFileSelect(file) {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))  // Create preview
}

// 2. User clicks recommend button
handleRecommend() {
    // Create FormData for multipart upload
    const formData = new FormData()
    formData.append('file', selectedFile)
    
    // Call backend API
    fetch('/api/recommend', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => setRecommendations(data.recommendations))
    .catch(e => setError(e.message))
}

// 3. Component displays recommendations
<RecommendationGrid recommendations={recommendations} />
```

**Key Points**:
- State management with `useState`
- FormData for multipart uploads
- Async/await pattern with fetch
- Error handling
- Component composition

### Backend Flow (Spring Boot)

```java
@PostMapping("/recommend")
public ResponseEntity<RecommendationResponse> recommend(
        @RequestParam("file") MultipartFile file) {
    
    try {
        // 1. Validate file
        validateFile(file);  // Checks: size, type, empty
        
        // 2. Create request to ML service
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(file.getBytes()) {...});
        HttpEntity<MultiValueMap<String, Object>> request = 
            new HttpEntity<>(body, headers);
        
        // 3. Call ML service
        ResponseEntity<RecommendationResponse> response =
            restTemplate.postForEntity(
                mlServiceUrl + "/recommend",
                request,
                RecommendationResponse.class
            );
        
        // 4. Return mapped response
        return ResponseEntity.ok(response.getBody());
        
    } catch (IllegalArgumentException e) {
        // Validation failed - return 400
        return ResponseEntity.badRequest().body(errorResponse);
    } catch (Exception e) {
        // System error - return 500
        return ResponseEntity.status(500).body(errorResponse);
    }
}
```

**Key Points**:
- Dependency injection (RestTemplate)
- Validation before processing
- Multipart data handling
- Exception handling strategy
- DTO mapping for type safety
- Structured error responses

## Running with Docker

If you have Docker installed:

```bash
# Run all services with docker-compose
docker-compose up

# Stops all services
docker-compose down
```

## Performance Considerations

1. **Image Upload Size Limit**: 5MB (frontend + backend validation)
2. **ML Model Loading**: Takes 10-30 seconds on first request
3. **FAISS Search**: Fast (< 100ms for similarity search)
4. **API Response Time**: ~500ms-2s (includes ML inference)

## Project Files Summary

```
fashion-ML-model/
├── frontend/                    # NEW: React Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadBox.jsx
│   │   │   ├── ImagePreview.jsx
│   │   │   └── RecommendationGrid.jsx
│   │   ├── App.jsx             # Main component
│   │   ├── main.jsx            # Entry point
│   │   └── index.css           # Tailwind
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── README.md
│
├── api-gateway/                # IMPROVED: Spring Boot
│   ├── src/main/java/com/example/demo/
│   │   ├── controller/
│   │   │   └── ImageController.java  # IMPROVED with validation
│   │   └── dto/
│   │       └── RecommendationResponse.java  # NEW DTO
│   ├── BACKEND_GUIDE.md        # NEW documentation
│   └── pom.xml
│
├── ml-service/                 # Existing FastAPI
│   ├── app/
│   │   ├── main.py
│   │   ├── embedding_generator.py
│   │   ├── search_engine.py
│   │   └── model_loader.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml          # Orchestrate all services
└── INTEGRATION_GUIDE.md         # This file

```

## Interview Talking Points

### Frontend
1. **Component Architecture**: Separation of concerns (Upload, Preview, Grid)
2. **State Management**: React hooks, lifting state to parent
3. **API Integration**: Fetch API, FormData for multipart
4. **User Experience**: Error messages, loading states, reset functionality
5. **Styling**: Tailwind CSS responsive classes

### Backend
1. **DTO Pattern**: Type safety, JSON mapping, extensibility
2. **Validation**: Input validation strategy, error codes
3. **Error Handling**: Distinguishing client vs server errors
4. **Logging**: Structured logging at different levels
5. **Microservice Communication**: REST API calls, timeout handling

### System Design
1. **API Gateway Pattern**: Centralized request routing
2. **Separation of Concerns**: Frontend / API / ML services
3. **Microservice Architecture**: Independent scalability
4. **Data Flow**: Multipart upload → Processing → Response mapping
5. **Error Propagation**: Frontend catches, displays user-friendly errors

## What's New

### Frontend (Created)
- ✅ React Vite project with all dependencies
- ✅ 4 reusable components with separation of concerns  
- ✅ Tailwind CSS styling (responsive, modern UI)
- ✅ File upload with validation
- ✅ Image preview and recommendation grid
- ✅ Error handling and loading states
- ✅ Extensive inline comments

### Backend (Improved)
- ✅ RecommendationResponse DTO for type-safe responses
- ✅ File validation: size, type, empty checks
- ✅ Comprehensive logging at info/debug/warn/error levels
- ✅ Better error handling: 400 vs 500 status codes
- ✅ Extensive code comments explaining each step
- ✅ Helper methods extracted for readability

## Next Day Tasks

After running successfully:

1. **Add Loading Spinner**: Show spinner while waiting for recommendations
2. **Add Image Optimization**: Resize before upload to reduce size
3. **Add Persistence**: Store previously recommended items
4. **Add Authentication**: Require login to use system
5. **Add Metrics**: Track usage, response times, error rates
6. **Add Retry Logic**: Automatically retry failed ML service calls
7. **Deploy to Production**: Docker, Kubernetes, or cloud platform

---

**Ready to demonstrate full-stack image recommendation system in interviews!**
