# Frontend Setup Guide

## Prerequisites
- Node.js 16+ 
- npm or yarn

## Installation Steps

### 1. Install Dependencies
```bash
cd frontend
npm install
```

This installs:
- React 18.2
- Vite (build tool)
- Tailwind CSS (styling)
- Other dev dependencies

### 2. Run Development Server
```bash
npm run dev
```

This starts the Vite dev server at `http://localhost:5173`

The frontend includes a proxy configuration that forwards `/api` requests to the Spring Boot backend at `http://localhost:8080`

### 3. Build for Production
```bash
npm run build
```

Generates optimized production build in `dist/` folder

### 4. Preview Production Build
```bash
npm run preview
```

## Architecture

### Component Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── UploadBox.jsx          # Drag-drop file upload
│   │   ├── ImagePreview.jsx       # Shows selected image
│   │   └── RecommendationGrid.jsx # Displays recommendations
│   ├── App.jsx                     # Main component, state & API logic
│   ├── main.jsx                    # React entry point
│   └── index.css                   # Tailwind CSS imports
├── index.html                      # HTML template
├── vite.config.js                  # Vite configuration
├── tailwind.config.js              # Tailwind CSS config
└── package.json                    # Dependencies
```

### How It Works

1. **Upload**: User drags/clicks to select image
2. **Preview**: Image appears with file name
3. **Recommend**: Click button to call `/api/recommend` API
4. **Display**: Grid shows similar items with similarity scores
5. **Reset**: Clear and start over

### Styling

Uses **Tailwind CSS** utility classes:
- Gradient backgrounds
- Responsive grid layout
- Hover effects and transitions
- Clean card-based UI

### API Communication

- **Endpoint**: `POST /api/recommend`
- **Format**: multipart/form-data
- **Response**: JSON with recommendations array
- **Error Handling**: User-friendly error messages

## Troubleshooting

**Port 5173 already in use?**
```bash
npm run dev -- --port 3000
```

**API calls not reaching backend?**
- Ensure Spring Boot runs on `localhost:8080`
- Check `vite.config.js` proxy configuration
- Verify backend CORS is enabled

**Tailwind styles not showing?**
- Restart dev server: `npm run dev`
- Clear browser cache
- Check that files are in `src/` directory

## Frontend Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| react | 18.2.0 | UI framework |
| react-dom | 18.2.0 | DOM rendering |
| vite | 4.3.9 | Build tool |
| tailwindcss | 3.3.0 | Utility CSS framework |

## Next Steps

1. Update image paths in RecommendationGrid if backend serves different format
2. Add loading spinner animations
3. Add image caching/optimization
4. Add authentication if needed
5. Deploy to production with Docker
