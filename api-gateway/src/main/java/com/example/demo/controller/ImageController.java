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
