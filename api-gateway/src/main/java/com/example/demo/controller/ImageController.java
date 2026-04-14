package com.example.demo.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.HttpStatusCodeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

@RestController
@RequestMapping("/api")
public class ImageController {

    private static final Logger logger = LoggerFactory.getLogger(ImageController.class);
    private static final int MAX_ATTEMPTS = 3;

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

            ResponseEntity<String> response = postWithRetry(requestEntity);

            return ResponseEntity.ok(response.getBody());

        } catch (HttpStatusCodeException e) {
            logger.error("ML service returned an error response", e);
            return ResponseEntity.status(e.getStatusCode())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(e.getResponseBodyAsString());
        } catch (Exception e) {
            logger.error("Error processing recommendation request", e);
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body("{\"error\":\"ML service is starting or unavailable. Please try again in a moment.\"}");
        }
    }

    private ResponseEntity<String> postWithRetry(HttpEntity<MultiValueMap<String, Object>> requestEntity)
            throws InterruptedException {
        String recommendUrl = mlServiceUrl + "/recommend";
        RestClientException lastError = null;

        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            try {
                return restTemplate.postForEntity(recommendUrl, requestEntity, String.class);
            } catch (RestClientException e) {
                lastError = e;

                if (attempt == MAX_ATTEMPTS || !isRetryable(e)) {
                    break;
                }

                logger.warn("ML service call failed on attempt {} of {}, retrying...", attempt, MAX_ATTEMPTS);
                TimeUnit.MILLISECONDS.sleep(600L * attempt);
            }
        }

        if (lastError instanceof HttpStatusCodeException statusException) {
            throw statusException;
        }

        throw new RestClientException("ML service is unavailable after retries");
    }

    private boolean isRetryable(RestClientException exception) {
        if (exception instanceof HttpStatusCodeException statusException) {
            HttpStatusCode statusCode = statusException.getStatusCode();
            return statusCode.value() == 503 || statusCode.value() == 502 || statusCode.value() == 504;
        }

        return true;
    }
}
