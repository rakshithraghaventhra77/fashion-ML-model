import { useState } from 'react'
import UploadBox from './components/UploadBox'
import ImagePreview from './components/ImagePreview'
import RecommendationGrid from './components/RecommendationGrid'

/**
 * Main App component
 * 
 * Flow:
 * 1. User uploads an image using UploadBox
 * 2. Image is previewed with ImagePreview
 * 3. API call is made to backend /api/recommend endpoint
 * 4. Recommendations are displayed in RecommendationGrid
 * 5. Loading and error states are managed here
 */
export default function App() {
  // State to store the selected file from upload
  const [selectedFile, setSelectedFile] = useState(null)
  
  // State to store the preview URL of selected image
  const [previewUrl, setPreviewUrl] = useState(null)
  
  // State to store API recommendations
  const [recommendations, setRecommendations] = useState(null)
  
  // State for API loading indicator
  const [loading, setLoading] = useState(false)
  
  // State for error messages
  const [error, setError] = useState(null)

  /**
   * Handles file selection from upload box
   * Creates a preview URL for the selected image
   * Resets previous recommendations
   */
  const handleFileSelect = (file) => {
    setSelectedFile(file)
    // Create a temporary URL to display the image
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    // Clear previous results when new file is selected
    setRecommendations(null)
    setError(null)
  }

  /**
   * Sends the selected image to backend API
   * Makes POST request to /api/recommend with multipart/form-data
   * Updates recommendations state with response
   */
  const handleRecommend = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      // Create FormData object to send file as multipart/form-data
      const formData = new FormData()
      formData.append('file', selectedFile)

      // Call backend API
      const response = await fetch('/api/recommend', {
        method: 'POST',
        body: formData,
      })

      // Check if response is successful
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      // Parse JSON response from backend
      const data = await response.json()

      // Store recommendations in state
      setRecommendations(data.recommendations)

    } catch (err) {
      // Display error message if API call fails
      setError(err.message || 'Failed to get recommendations')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  /**
   * Resets all state to initial values
   * Clears uploads, previews, and recommendations
   */
  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setRecommendations(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header Section */}
      <header className="bg-white shadow">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
            🎨 Fashion Recommendation
          </h1>
          <p className="text-gray-600 mt-2">
            Upload an image to discover similar fashion items
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Upload and Preview */}
          <div className="space-y-6">
            {/* Upload Box Component */}
            <UploadBox onFileSelect={handleFileSelect} />

            {/* Image Preview Component */}
            {previewUrl && (
              <ImagePreview 
                previewUrl={previewUrl}
                fileName={selectedFile?.name}
              />
            )}

            {/* Action Buttons */}
            {selectedFile && (
              <div className="flex gap-4">
                <button
                  onClick={handleRecommend}
                  disabled={loading}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Finding Similar Items...' : 'Get Recommendations'}
                </button>
                <button
                  onClick={handleReset}
                  className="flex-1 bg-gray-200 text-gray-800 py-3 rounded-lg font-semibold hover:bg-gray-300 transition-all"
                >
                  Reset
                </button>
              </div>
            )}

            {/* Error Message Display */}
            {error && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                <p className="font-semibold">Error</p>
                <p>{error}</p>
              </div>
            )}
          </div>

          {/* Right Column: Recommendations Grid */}
          <div>
            {recommendations && (
              <RecommendationGrid recommendations={recommendations} />
            )}
            {!recommendations && !error && selectedFile && (
              <div className="bg-white rounded-lg shadow-md p-8 text-center text-gray-500">
                <p>Click "Get Recommendations" to see similar items</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
