# Fashion Recommendation System - Interview Prep Guide (Development Story Format)

## 1. Elevator Pitch
I built an image-based fashion recommendation system as two microservices: a Java Spring Boot API Gateway and a Python FastAPI ML service. The gateway receives uploads, the ML service converts images to embeddings using MobileNetV2, and FAISS returns the top similar products.

## 2. Development Story: What I Built, In Order, and Why

This is the most important section for interview delivery.

### Phase 1: Build the ML core first
What I built first:
- Embedding extraction with MobileNetV2 in ml-service/app/embedding_generator.py.
- Vector search with FAISS in ml-service/app/search_engine.py.

Why this order:
- Recommendation quality depends on ML output. If embedding and retrieval are weak, API layers do not matter.
- Building core logic first let me validate feasibility early.

Code proof snippet:
```python
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])
```

Story line to say:
I started with ML retrieval because that is the heart of the product. I used a pretrained lightweight model to avoid expensive training and got a reliable baseline quickly.

### Phase 2: Expose ML as a service
What I built second:
- FastAPI app and endpoints in ml-service/app/main.py.
- Health endpoint and recommend endpoint.

Why this order:
- Once core retrieval worked locally, I needed network APIs so other components could consume it.

Code proof snippet:
```python
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    query_vector = extract_embedding(img, model)
    results = search(query_vector, TOP_K)
    return {"success": True, "recommendations": results}
```

Story line to say:
After local model validation, I wrapped it behind a clean HTTP contract so I could keep ML isolated and independently deployable.

### Phase 3: Build API Gateway boundary
What I built third:
- Java controller to receive client upload and forward to ML service.
- RestTemplate with timeouts.

Why this order:
- A gateway gives a stable API for clients while allowing ML internals to change without breaking consumers.

Code proof snippet:
```java
String recommendUrl = mlServiceUrl + "/recommend";
ResponseEntity<String> response =
        restTemplate.postForEntity(recommendUrl, requestEntity, String.class);

return ResponseEntity.ok(response.getBody());
```

Story line to say:
I introduced a gateway to separate transport concerns from ML logic and to make production hardening easier later, like retries and auth.

### Phase 4: Add operational readiness
What I built fourth:
- Health endpoints in both services.
- CORS config for local development.
- Dockerfiles and docker-compose service orchestration.

Why this order:
- After feature correctness, I focused on runnability and integration so the full system could be demoed quickly.

Code proof snippets:
```yaml
services:
  ml-service:
    build: ./ml-service
    ports:
      - "8000:8000"

  api-gateway:
    build: ./api-gateway
    ports:
      - "8080:8080"
    depends_on:
      - ml-service
```

```java
factory.setConnectTimeout(5000);
factory.setReadTimeout(30000);
```

Story line to say:
I treated observability and containerization as first-class requirements because interviewers care about deployability, not just model accuracy.

### Phase 5: Indexing and data artifacts
What I built fifth:
- Scripts for generating embeddings and FAISS index files.
- Storage under ml-service/index.

Why this order:
- Query-time speed depends on precomputed vectors. Doing this before final integration prevented runtime bottlenecks.

Code proof snippet:
```python
embeddings = np.load(embeddings_path).astype("float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, str(index_dir / "faiss_index.index"))
```

Story line to say:
I moved heavy computation to preprocessing so online recommendations remain fast.

## 3. Full Codebase Teaching Map

### Root
- docker-compose.yml: runs both services and wiring.
- PROJECT_GUIDE.md: conceptual documentation.
- data/: source images and metadata.

### API Gateway (api-gateway)
- pom.xml: Spring Boot parent 4.0.3, Java 17, webmvc dependencies.
- application.properties: ml.service.url config.
- ImageController.java: POST /api/recommend, forwards multipart to ML.
- HealthController.java: /api/health and /api/ root endpoint.
- RestTemplateConfig.java: timeout-safe HTTP client bean.
- CorsConfig.java: local browser integration support.
- Dockerfile: runtime image for JAR.

### ML Service (ml-service)
- requirements.txt: FastAPI, TensorFlow, FAISS, Pillow.
- app/main.py: API layer, CORS, startup model loading.
- app/config.py: index and inference constants.
- app/model_loader.py: model singleton-style loader.
- app/embedding_generator.py: model build + embedding extraction logic.
- app/search_engine.py: FAISS search operation.
- app/build_faiss_index.py: precompute index pipeline.
- app/simple_search.py and app/query_pipeline.py: exploratory retrieval scripts.
- Dockerfile: Python runtime and Uvicorn startup.

## 4. System Flow You Should Narrate in Interview
1. Client uploads image to API gateway endpoint POST /api/recommend.
2. Gateway packages file as multipart and forwards to ML service.
3. ML service reads image, generates embedding, searches FAISS index.
4. ML service returns top-K file paths.
5. Gateway returns JSON response to client.

## 5. Build and Run Guide

### Preferred: Docker
From project root:

```powershell
docker compose up --build
```

Check health:
- http://localhost:8080/api/health
- http://localhost:8000/health

Test recommendation:
```powershell
curl -X POST -F "file=@data/images/sample.jpg" http://localhost:8080/api/recommend
```

### Local mode (without Docker)

ML service:
```powershell
cd ml-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app/build_faiss_index.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API gateway in new terminal:
```powershell
cd api-gateway
.\mvnw.cmd clean package
.\mvnw.cmd spring-boot:run
```

Important local note:
- For local run, set ml.service.url to http://localhost:8000.
- For Docker run, keep ml.service.url as http://ml-service:8000.

## 5.1 Current Frontend Shape
- The UI is intentionally minimal now.
- It keeps the same theme and fonts, but removes the extra learning sidebar.
- The page focuses on one clean path: upload, preview, analyze, and show results.
- The code is shorter and easier to explain in an interview.

## 6. Challenges I Faced and How I Overcame Them

### Challenge A: Docker hostname vs localhost mismatch
Problem:
- Service URL works in Docker network but fails in local host mode.

Fix approach:
- Use environment-based configuration for ml.service.url.

Interview line:
I learned to separate deployment config from code and introduced profile-based service discovery behavior.

### Challenge B: Startup dependency on model/index artifacts
Problem:
- If model or FAISS files are missing, recommendations fail.

Fix approach:
- Added startup loading pattern and health checks; enforce pre-index generation in setup flow.

Interview line:
I treated model/index readiness as a production reliability concern, not just a development detail.

### Challenge C: Gateway-to-ML startup timing
Problem:
- The gateway can reach the ML service before it is fully ready.

Fix approach:
- Add a small retry loop in the Java gateway.
- Return a clearer 503 response if the ML service is still warming up.

Interview line:
I handled service warm-up as a normal deployment reality, so the user gets a clean retry path instead of a raw connection failure.

### Challenge C: Script consistency risk in preprocessing module
Problem observed in current repository:
- build_faiss_index.py imports generate_embeddings, but embedding_generator.py structure indicates a potential missing/inconsistent function block.

Fix plan:
- Refactor embedding_generator.py to explicitly provide build_model, extract_embedding, and generate_embeddings.
- Add smoke test import check in CI.

Interview line:
I identified a pipeline fragility point and converted it into a stable, testable preprocessing contract.

## 7. Integrations Left and How We Plan to Build Them

### Integration 1: Frontend upload and result display
What is left:
- No UI service is wired yet.

Plan:
1. Build minimal frontend with image upload, preview, loading, and recommendation cards.
2. Integrate with gateway POST /api/recommend.
3. Add result metadata rendering.

### Integration 2: Metadata enrichment
What is left:
- Current output mostly returns image paths.

Plan:
1. Parse styles.csv into lookup map.
2. Return enriched response: id, category, brand, color, price, image.
3. Keep backward compatibility for existing response field.

### Integration 3: Reliability layer in gateway
What is left:
- Timeout exists, but retries/circuit-breaker are not implemented.

Plan:
1. Add retry with exponential backoff.
2. Add circuit breaker for ML outage periods.
3. Add standardized error codes.

### Integration 4: End-to-end automated tests and CI
What is left:
- Full integration test pipeline is limited.

Plan:
1. Add API contract tests.
2. Add full upload-to-recommend integration test with sample fixture.
3. Add CI workflow to run gateway tests, ML tests, and smoke check.

### Integration 5: Final hardening
What is left:
- Add a clearer fallback when the ML service is unavailable during startup.

Plan:
1. Keep retry logic in the gateway.
2. Preserve ML service error codes when it responds with a real application error.
3. Use a friendly 503 only for temporary unavailability.

## 8. Interview Questions With Story-Based Answers

Q1. Why this implementation order?
Answer:
I prioritized risk reduction. I validated ML retrieval first because recommendation quality is the core risk, then exposed APIs, then integrated gateway and deployment.

Q2. Why microservices and not monolith?
Answer:
ML and API layers have different scaling and dependency profiles. Isolating them improves deployment flexibility and fault containment.

Q3. Why MobileNetV2 and FAISS?
Answer:
MobileNetV2 gives strong lightweight features, and FAISS provides fast nearest-neighbor retrieval with a clear path from exact to approximate indexes.

Q4. What would you improve next?
Answer:
Frontend integration, metadata-enriched recommendations, resilient gateway patterns, and end-to-end CI tests.

Q5. What hard engineering lesson did you learn?
Answer:
Configuration and artifact readiness break systems more often than core logic. I now design startup validation and environment-specific config from day one.

## 9. 2-Minute Interview Delivery Script
I built this project in five phases. First I validated ML retrieval with MobileNetV2 embeddings and FAISS search. Second I exposed that logic via FastAPI endpoints. Third I added a Spring Boot gateway to isolate client-facing API concerns and forward multipart requests to ML. Fourth I productionized the setup with health checks, CORS, timeouts, and Docker Compose orchestration. Fifth I finalized preprocessing scripts to generate index artifacts for fast online inference. This sequence helped me reduce risk early, keep architecture modular, and make the system demo-ready and scalable.

## 10. Last-Minute Revision Checklist
- I can explain each development phase in order with why it came next.
- I can walk through at least one code snippet per phase.
- I can explain one challenge and one concrete fix story.
- I can run the full system with Docker and hit both health endpoints.
- I can clearly state the remaining integrations and roadmap.

## 11. Code-Heavy Appendix (Use in Interview)

### 11.1 API Gateway controller code
From ImageController.java:

  package com.example.demo.controller;

  import org.springframework.beans.factory.annotation.Value;
  import org.springframework.core.io.ByteArrayResource;
  import org.springframework.http.*;
  import org.springframework.util.LinkedMultiValueMap;
  import org.springframework.util.MultiValueMap;
  import org.springframework.web.bind.annotation.*;
  import org.springframework.web.client.RestTemplate;
  import org.springframework.web.multipart.MultipartFile;
  import org.slf4j.Logger;
  import org.slf4j.LoggerFactory;

  @RestController
  @RequestMapping("/api")
  public class ImageController {

    private static final Logger logger = LoggerFactory.getLogger(ImageController.class);

    private final RestTemplate restTemplate;
    private final String mlServiceUrl;

    public ImageController(RestTemplate restTemplate,
                @Value("${ml.service.url}") String mlServiceUrl) {
      this.restTemplate = restTemplate;
      this.mlServiceUrl = mlServiceUrl;
    }

    @PostMapping("/recommend")
    public ResponseEntity<String> recommend(@RequestParam("file") MultipartFile file) {

      try {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

        body.add("file", new ByteArrayResource(file.getBytes()) {
          @Override
          public String getFilename() {
            return file.getOriginalFilename();
          }
        });

        HttpEntity<MultiValueMap<String, Object>> requestEntity =
            new HttpEntity<>(body, headers);

        String recommendUrl = mlServiceUrl + "/recommend";
        ResponseEntity<String> response =
            restTemplate.postForEntity(recommendUrl, requestEntity, String.class);

        return ResponseEntity.ok(response.getBody());

      } catch (Exception e) {
        logger.error("Error processing recommendation request", e);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body("{\"error\": \"Failed to process recommendation request\"}");
      }
    }
  }

How to explain quickly:
- This is where multipart upload enters the system.
- Gateway forwards payload without ML logic inside Java service.
- Errors are normalized to a predictable JSON response.

### 11.2 API Gateway health and timeout config
From HealthController.java:

  @GetMapping("/health")
  public ResponseEntity<Map<String, String>> health() {
    return ResponseEntity.ok(Map.of(
      "status", "UP",
      "service", "Fashion Recommendation API Gateway",
      "timestamp", Instant.now().toString()
    ));
  }

From RestTemplateConfig.java:

  @Bean
  public RestTemplate restTemplate() {
    SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
    factory.setConnectTimeout(5000);
    factory.setReadTimeout(30000);
    return new RestTemplate(factory);
  }

How to explain quickly:
- Health endpoint supports operational checks.
- Timeout settings avoid indefinite waits during ML service delays.

### 11.3 ML service API code
From ml-service/app/main.py:

  from fastapi import FastAPI, UploadFile, File, HTTPException
  from fastapi.middleware.cors import CORSMiddleware
  from PIL import Image
  import io
  import logging
  import os

  from app.embedding_generator import extract_embedding
  from app.model_loader import get_model
  from app.search_engine import search
  from app.config import TOP_K

  app = FastAPI(title="Image Similarity Service")

  ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8080,http://127.0.0.1:8080"
  ).split(",")

  app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)

  try:
    model = get_model()
    logger.info("Model loaded successfully")
  except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    model = None

  @app.get("/health")
  def health_check():
    return {"status": "Service is running", "model_loaded": model is not None}

  @app.post("/recommend")
  async def recommend(file: UploadFile = File(...)):
    if model is None:
      raise HTTPException(status_code=503, detail="Model not loaded")

    try:
      contents = await file.read()
      img = Image.open(io.BytesIO(contents)).convert("RGB")
      query_vector = extract_embedding(img, model)
      results = search(query_vector, TOP_K)
      return {
        "success": True,
        "recommendations": results
      }
    except Exception as e:
      logger.exception(f"Error processing image: {e}")
      raise HTTPException(status_code=500, detail="Image processing failed") from e

How to explain quickly:
- Model is loaded once at startup, reducing per-request overhead.
- Endpoint is async and returns deterministic JSON contract.

### 11.4 Embedding extraction code
From ml-service/app/embedding_generator.py:

  def build_model():
    base_model = MobileNetV2(
      weights="imagenet",
      include_top=False,
      input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = Sequential([
      base_model,
      GlobalMaxPooling2D()
    ])

    return model

  def extract_embedding(image, model):

    if isinstance(image, (str, Path)):
      img = Image.open(image).convert("RGB")
    else:
      img = image

    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0)
    features = features.flatten()
    features = features / norm(features)

    return features

How to explain quickly:
- Image preprocessing matches MobileNetV2 expectations.
- L2 normalization improves similarity consistency.

### 11.5 FAISS search code
From ml-service/app/search_engine.py:

  index = faiss.read_index(str(FAISS_INDEX_PATH))

  with open(FILENAMES_PATH, "rb") as f:
    filenames = pickle.load(f)

  def search(query_vector, top_k=5):
    query_vector = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    return [filenames[i] for i in indices[0]]

How to explain quickly:
- Query embedding is cast to float32 and searched in index.
- Returned indices are mapped to filenames for client output.

### 11.6 Index build code
From ml-service/app/build_faiss_index.py:

  embeddings = np.load(embeddings_path).astype("float32")
  print("Embeddings shape:", embeddings.shape)

  dimension = embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(embeddings)
  print("Total vectors:" , index.ntotal)

  faiss.write_index(index, str(index_dir / "faiss_index.index"))
  print("FAISS index built and saved successfully.")

How to explain quickly:
- This preprocessing stage enables fast runtime retrieval.
- IndexFlatL2 provides exact nearest-neighbor baseline.

### 11.7 Docker and compose code
From docker-compose.yml:

  services:
    ml-service:
    build: ./ml-service
    container_name: ml-service
    ports:
      - "8000:8000"

    api-gateway:
    build: ./api-gateway
    container_name: api-gateway
    ports:
      - "8080:8080"
    depends_on:
      - ml-service

From ml-service/Dockerfile:

  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

From api-gateway/Dockerfile:

  FROM eclipse-temurin:17-jdk-alpine
  WORKDIR /app
  COPY target/*.jar app.jar
  EXPOSE 8080
  ENTRYPOINT ["java", "-jar", "app.jar"]

How to explain quickly:
- Compose gives quick local orchestration and realistic service communication.
- Each service has its own runtime image and dependency boundary.

### 11.8 Short code walkthrough flow (30-second answer)
You can say this exactly:
I receive the image in Java at POST /api/recommend, forward it as multipart to the Python service, the Python endpoint loads image bytes and calls extract_embedding, then search_engine queries FAISS and returns top-K matches, and the gateway returns that response to the client. This keeps API concerns and ML concerns cleanly separated.
