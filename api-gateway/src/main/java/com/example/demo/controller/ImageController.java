package com.example.demo.controller;

import com.example.demo.dto.RecommendationResponse;
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

/**
 * ImageController
 * 
 * Handles image upload and recommendation requests
 * 
 * Flow:
 * 1. Receives image file from frontend (multipart/form-data)
 * 2. Validates the file (checks size, type)
 * 3. Forwards to ML microservice (FastAPI)
 * 4. Receives JSON response with recommendations
 * 5. Maps JSON to RecommendationResponse DTO
 * 6. Returns structured response to frontend
 */
@RestController
@RequestMapping("/api")
public class ImageController {

    private static final Logger logger = LoggerFactory.getLogger(ImageController.class);
    
    // Maximum file size allowed: 5MB
    private static final long MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024;
    
    // Allowed image content types
    private static final String[] ALLOWED_IMAGE_TYPES = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/webp"
    };

    private final RestTemplate restTemplate;
    private final String mlServiceUrl;

    /**
     * Constructor - dependency injection for RestTemplate and ML service URL
     */
    public ImageController(RestTemplate restTemplate, 
                          @Value("${ml.service.url}") String mlServiceUrl) {
        this.restTemplate = restTemplate;
        this.mlServiceUrl = mlServiceUrl;
        logger.info("ImageController initialized with ML service URL: {}", mlServiceUrl);
    }

    /**
     * POST /api/recommend
     * 
     * Accepts an image file and returns recommended similar items
     * 
     * Request: multipart/form-data with 'file' parameter
     * Response: JSON with recommendations array
     * 
     * @param file - The uploaded image file
     * @return ResponseEntity with RecommendationResponse DTO
     */
    @PostMapping("/recommend")
    public ResponseEntity<RecommendationResponse> recommend(
            @RequestParam("file") MultipartFile file) {

        try {
            // Step 1: Validate the uploaded file
            validateFile(file);
            logger.info("File validation passed for: {}", file.getOriginalFilename());

            // Step 2: Create request to send to ML service
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // Create multipart form body with file
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });

            HttpEntity<MultiValueMap<String, Object>> requestEntity =
                    new HttpEntity<>(body, headers);

            // Step 3: Call ML microservice
            String recommendUrl = mlServiceUrl + "/recommend";
            logger.debug("Calling ML service at: {}", recommendUrl);
            
            ResponseEntity<RecommendationResponse> response =
                    restTemplate.postForEntity(recommendUrl, requestEntity, 
                            RecommendationResponse.class);

            // Step 4: Log the result
            if (response.getBody() != null && response.getBody().isSuccess()) {
                logger.info("Successfully got {} recommendations", 
                        response.getBody().getRecommendations().size());
            }

            // Step 5: Return the mapped DTO response
            return ResponseEntity.ok(response.getBody());

        } catch (IllegalArgumentException e) {
            // File validation failed
            logger.warn("File validation failed: {}", e.getMessage());
            RecommendationResponse errorResponse = new RecommendationResponse();
            errorResponse.setSuccess(false);
            return ResponseEntity.badRequest().body(errorResponse);

        } catch (Exception e) {
            // Any other error (network, parsing, etc)
            logger.error("Error processing recommendation request: {}", e.getMessage(), e);
            RecommendationResponse errorResponse = new RecommendationResponse();
            errorResponse.setSuccess(false);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(errorResponse);
        }
    }

    /**
     * Validates the uploaded file
     * 
     * Checks:
     * - File is not empty
     * - File size is within limit
     * - File type is an allowed image format
     * 
     * @param file - The file to validate
     * @throws IllegalArgumentException if validation fails
     */
    private void validateFile(MultipartFile file) throws IllegalArgumentException {
        // Check if file is empty
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("File is empty or not provided");
        }

        // Check file size
        if (file.getSize() > MAX_FILE_SIZE_BYTES) {
            throw new IllegalArgumentException(
                    "File size exceeds maximum allowed size of 5MB. Got: " + 
                    (file.getSize() / (1024 * 1024)) + "MB");
        }

        // Check file type
        String contentType = file.getContentType();
        if (contentType == null || !isAllowedImageType(contentType)) {
            throw new IllegalArgumentException(
                    "Invalid file type. Allowed types: JPEG, PNG, GIF, WebP. Got: " + contentType);
        }

        logger.debug("File validation successful - Size: {} bytes, Type: {}", 
                file.getSize(), contentType);
    }

    /**
     * Checks if the provided content type is an allowed image format
     * 
     * @param contentType - The MIME type to check
     * @return true if the type is allowed, false otherwise
     */
    private boolean isAllowedImageType(String contentType) {
        for (String allowedType : ALLOWED_IMAGE_TYPES) {
            if (contentType.equalsIgnoreCase(allowedType)) {
                return true;
            }
        }
        return false;
    }
}
