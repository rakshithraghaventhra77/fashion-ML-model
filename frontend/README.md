# Frontend - Fashion Recommendation System

A minimal, interview-ready React frontend for the Fashion Recommendation system built with Vite and Tailwind CSS.

## Features

✨ **Image Upload**
- Drag-and-drop file upload
- Click to select from computer
- Image format validation (JPEG, PNG, GIF, WebP)

🖼️ **Image Preview**
- Shows selected image with filename
- Responsive layout

🎯 **Recommendations Display**
- Grid layout showing 5 similar items
- Displays similarity score for each item
- Responsive to mobile/tablet/desktop

🎨 **UI/UX**
- Clean, modern design with Tailwind CSS
- Gradient backgrounds
- Smooth transitions and hover effects
- Error messages with clear feedback

## Quick Start

```bash
# Install dependencies
npm install

# Run dev server (http://localhost:5173)
npm run dev

# Build for production
npm run build
```

## Code Structure

### App.jsx - Main Component

Handles:
- State management (file, preview, recommendations, loading, errors)
- API calls to `/api/recommend`
- User interactions (upload, recommend, reset)
- Layout and basic styling

```javascript
// Key states:
- selectedFile       // The File object from upload
- previewUrl         // Temporary URL for image preview
- recommendations    // Array from API response
- loading            // Boolean for loading state
- error              // Error message string
```

### Components

**UploadBox.jsx** - File Upload Component
- Handles drag-and-drop and click-to-upload
- Validates file is an image
- Calls `onFileSelect` callback with File object

**ImagePreview.jsx** - Preview Component  
- Shows selected image
- Displays filename
- Simple, focused presentation

**RecommendationGrid.jsx** - Recommendations Display
- Maps recommendation array to cards
- Shows image and similarity score
- Responsive grid (2-3 columns)
- Fallback for missing images

## API Integration

### Endpoint
```
POST /api/recommend
Content-Type: multipart/form-data
```

### Request
```
File parameter: "file" (FormData)
```

### Response
```json
{
  "success": true,
  "recommendations": [
    {
      "image_path": "path/to/image1.jpg",
      "score": 0.95
    },
    {
      "image_path": "path/to/image2.jpg", 
      "score": 0.92
    }
  ]
}
```

## Interview Points to Explain

1. **Component Abstraction**: Why each component has single responsibility
2. **State Management**: How to lift state up to parent (App.jsx)
3. **API Integration**: RestTemplate proxy in Vite, FormData for multipart
4. **Error Handling**: User-friendly error messages, try/catch blocks
5. **File Validation**: Check type on frontend + backend validation
6. **Responsive Design**: Tailwind CSS responsive classes (grid-cols-2 md:grid-cols-3)

## Key Concepts

### Drag-and-Drop
```javascript
// Event listeners: onDragOver, onDragLeave, onDrop
// Prevents default behavior, extracts file from event.dataTransfer
```

### File Preview
```javascript
// Creates temporary URL using URL.createObjectURL()
// Displays with <img> tag
// Cleans up memory via callback
```

### API Calls
```javascript
// FormData for multipart upload
// Fetch API for HTTP requests
// JSON response mapping
```

## Styling with Tailwind

- **Layout**: `max-w-6xl`, `grid grid-cols-1 lg:grid-cols-2`
- **Colors**: Gradient backgrounds, shadowing
- **Responsive**: Mobile-first breakpoints (md:, lg:)
- **Interactions**: `hover:`, `transition-`, `disabled:`

## Performance Considerations

- Lazy loading of images in recommendations grid
- Image object URL cleanup on component unmount
- Tailwind CSS purging unused styles in production build
- Vite fast refresh for development

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Modern browsers with ES6+ support

## Development

Built with:
- React 18.2 (Hooks-based)
- Vite 4.3.9 (Fast build tool)
- Tailwind CSS 3.3.0 (Utility-first CSS)
- PostCSS for CSS processing
