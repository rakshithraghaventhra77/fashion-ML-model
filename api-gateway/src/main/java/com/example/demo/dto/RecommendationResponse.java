package com.example.demo.dto;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * RecommendationResponse DTO
 * 
 * Maps the Python/FastAPI response to a structured Java object
 * Makes it easier to work with the API response in a type-safe way
 * 
 * Example JSON from FastAPI:
 * {
 *   "success": true,
 *   "recommendations": [
 *     {"image_path": "path/to/image1.jpg", "score": 0.95},
 *     {"image_path": "path/to/image2.jpg", "score": 0.92}
 *   ]
 * }
 */
public class RecommendationResponse {
    
    // Whether the recommendation API call was successful
    private boolean success;
    
    // List of recommended items with image paths and similarity scores
    private List<Recommendation> recommendations;

    // Default constructor (required for JSON deserialization)
    public RecommendationResponse() {
    }

    // Constructor with parameters
    public RecommendationResponse(boolean success, List<Recommendation> recommendations) {
        this.success = success;
        this.recommendations = recommendations;
    }

    // Getter for success flag
    public boolean isSuccess() {
        return success;
    }

    // Setter for success flag
    public void setSuccess(boolean success) {
        this.success = success;
    }

    // Getter for recommendations list
    public List<Recommendation> getRecommendations() {
        return recommendations;
    }

    // Setter for recommendations list
    public void setRecommendations(List<Recommendation> recommendations) {
        this.recommendations = recommendations;
    }

    /**
     * Inner class representing a single recommendation item
     * Contains the image path and similarity score
     */
    public static class Recommendation {
        
        // Path to the recommended image file
        @JsonProperty("image_path")
        private String imagePath;
        
        // Similarity score (0.0 to 1.0, where 1.0 is identical)
        private double score;

        // Default constructor
        public Recommendation() {
        }

        // Constructor with parameters
        public Recommendation(String imagePath, double score) {
            this.imagePath = imagePath;
            this.score = score;
        }

        // Getter for image path
        public String getImagePath() {
            return imagePath;
        }

        // Setter for image path
        public void setImagePath(String imagePath) {
            this.imagePath = imagePath;
        }

        // Getter for similarity score
        public double getScore() {
            return score;
        }

        // Setter for similarity score
        public void setScore(double score) {
            this.score = score;
        }

        @Override
        public String toString() {
            return "Recommendation{" +
                    "imagePath='" + imagePath + '\'' +
                    ", score=" + score +
                    '}';
        }
    }

    @Override
    public String toString() {
        return "RecommendationResponse{" +
                "success=" + success +
                ", recommendations=" + recommendations +
                '}';
    }
}
