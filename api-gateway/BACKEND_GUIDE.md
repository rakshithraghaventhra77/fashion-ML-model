# Backend - Spring Boot API Gateway

## Overview

The API Gateway is a Spring Boot microservice that:
1. Receives image uploads from the React frontend
2. Validates the uploaded file
3. Forwards requests to the FastAPI ML microservice
4. Maps responses to structured DTOs
5. Returns recommendations to the frontend

## Architecture

```
Frontend (React)
    ↓
API Gateway (Spring Boot) ← You are here
    ↓
ML Service (FastAPI)
    ↓
FAISS Vector Index
```

## Key Components

### ImageController
**Location**: `src/main/java/com/example/demo/controller/ImageController.java`

Handles the `/api/recommend` endpoint:
1. **Validates** uploaded file (size, type)
2. **Forwards** to ML microservice via RestTemplate
3. **Maps** JSON response to RecommendationResponse DTO
4. **Returns** structured response to frontend

**Key Methods**:
```java
// Main endpoint
POST /api/recommend (MultipartFile file)
  ├─ validateFile()         // Check size, type
  ├─ Call ML Service        // HTTP POST
  ├─ Map Response to DTO    // JSON deserialization
  └─ Return DTO             // Type-safe response

// Helper methods
validateFile()              // File validation logic
isAllowedImageType()        // Content-type checking
```

### RecommendationResponse DTO
**Location**: `src/main/java/com/example/demo/dto/RecommendationResponse.java`

Maps the FastAPI JSON response to a Java object:

```java
RecommendationResponse {
    boolean success;
    List<Recommendation> recommendations;

    Recommendation {
        String imagePath;
        double score;        // 0.0 to 1.0
    }
}
```

**Why DTOs?**
- Type safety (no casting strings)
- Automatic JSON deserialization
- Easy to extend with validation annotations
- Self-documenting code

## Validation

### File Size
- **Limit**: 5MB
- **Check**: `file.getSize() > MAX_FILE_SIZE_BYTES`
- **Error**: Returns 400 Bad Request with message

### File Type
- **Allowed**: JPEG, PNG, GIF, WebP
- **Check**: Validates Content-Type header
- **Error**: Returns 400 Bad Request with message

### File Empty Check
- **Check**: `file.isEmpty()`
- **Error**: Returns 400 Bad Request

## Logging

**Log Levels Used**:
- `logger.info()` - Major operations (file validation passed, recommendations returned)
- `logger.debug()` - Detailed info (ML service URL, file size)
- `logger.warn()` - Validation failures
- `logger.error()` - Exception handling

**Example Logs**:
```
INFO - ImageController initialized with ML service URL: http://localhost:8000
INFO - File validation passed for: dress.jpg
DEBUG - Calling ML service at: http://localhost:8000/recommend
INFO - Successfully got 5 recommendations
```

## Interview Explanation Points

### 1. Why DTOs?
- Separate API contract from internal representation
- Type-safe than raw JSON strings
- Easy to add validation later
- Self-documenting with comments

### 2. How validation works?
- Size check prevents memory issues
- Type check prevents malicious uploads
- Empty check catches edge cases
- All happen before ML service call

### 3. Error Handling Strategy
- Validation errors → 400 Bad Request (client's fault)
- System errors → 500 Internal Server Error (server's fault)
- Return DTO with success=false for API consistency

### 4. RestTemplate Configuration
- Defined in `RestTemplateConfig.java`
- Used for synchronous HTTP calls
- Handles multipart/form-data automatically

### 5. CORS Configuration
- Defined in `CorsConfig.java`
- Allows requests from React frontend
- Required for cross-origin requests

## Configuration

**application.properties**:
```properties
# ML microservice URL
ml.service.url=http://localhost:8000
```

## API Endpoint

```
Endpoint: POST /api/recommend
Content-Type: multipart/form-data

Request:
- Parameter: "file" (image file)

Response:
{
  "success": true,
  "recommendations": [
    {
      "imagePath": "path/to/image1.jpg",
      "score": 0.95
    }
  ]
}

Error Response:
{
  "success": false
}
Status: 400 (validation) or 500 (system error)
```

## Development

### Build
```bash
mvn clean package
```

### Run
```bash
mvn spring-boot:run
```

Server runs on `http://localhost:8080`

### Test File Upload (cURL)
```bash
curl -X POST -F "file=@path/to/image.jpg" \
  http://localhost:8080/api/recommend
```

## Dependencies

- **Spring Boot 4.0.3**: Web framework
- **RestTemplate**: HTTP client (built-in)
- **Jackson**: JSON serialization (built-in)
- **SLF4J**: Logging (built-in)

## Code Walkthrough

### validateFile() Method

```java
private void validateFile(MultipartFile file) {
    // 1. Check if file exists
    if (file == null || file.isEmpty()) {
        throw new IllegalArgumentException("File is empty");
    }
    
    // 2. Check file size
    if (file.getSize() > MAX_FILE_SIZE_BYTES) {
        throw new IllegalArgumentException("File too large");
    }
    
    // 3. Check file type
    if (!isAllowedImageType(file.getContentType())) {
        throw new IllegalArgumentException("Invalid file type");
    }
}
```

**Why this order?**
- Empty check first (cheap operation)
- Size check second (prevents DoS)
- Type check last (more expensive)

### API Call Flow

```java
@PostMapping("/recommend")
public ResponseEntity<RecommendationResponse> recommend(MultipartFile file) {
    try {
        // 1. Validate input
        validateFile(file);
        
        // 2. Prepare HTTP request
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        
        // 3. Create multipart body
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(file.getBytes()) {...});
        
        // 4. Send to ML service
        ResponseEntity<RecommendationResponse> response =
            restTemplate.postForEntity(mlServiceUrl + "/recommend", 
                                      requestEntity, 
                                      RecommendationResponse.class);
        
        // 5. Return mapped response
        return ResponseEntity.ok(response.getBody());
        
    } catch (IllegalArgumentException e) {
        // Validation failed - client error
        return ResponseEntity.badRequest().body(errorResponse);
    } catch (Exception e) {
        // System error
        return ResponseEntity.status(500).body(errorResponse);
    }
}
```

## Interview Questions You'll Get

1. **"How do you handle errors?"**
   - Validation errors → 400, System errors → 500
   - DTO with success flag provides consistent API

2. **"Why use DTOs instead of raw JSON?"**
   - Type safety, easier to extend, self-documenting
   - JsonProperty annotation handles field name mismatches

3. **"How does file validation work?"**
   - Size, type, empty checks before processing
   - Prevents DoS, malicious files, memory issues

4. **"What if ML service is down?"**
   - RestTemplate throws exception, caught in catch block
   - Returns 500 error with success=false

5. **"How do you log important operations?"**
   - SLF4J with different levels
   - Log entry point, key decisions, errors

## Next Steps

1. Add request/response interceptors for monitoring
2. Add circuit breaker pattern for ML service resilience
3. Add metrics collection (response time, error rates)
4. Add unit tests for validation logic
5. Add retry logic for transient failures
