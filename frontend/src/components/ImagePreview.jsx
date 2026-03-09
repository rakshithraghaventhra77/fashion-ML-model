/**
 * ImagePreview Component
 * 
 * Displays the selected image before sending to API
 * Shows file name and basic image information
 * 
 * Props:
 * - previewUrl: URL for the preview image (blob URL)
 * - fileName: Original file name from upload
 */
export default function ImagePreview({ previewUrl, fileName }) {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Header with file name */}
      <div className="bg-gradient-to-r from-blue-500 to-indigo-500 text-white p-4">
        <p className="font-semibold">Selected Image</p>
        {fileName && <p className="text-sm opacity-90">{fileName}</p>}
      </div>

      {/* Image display */}
      <div className="p-4">
        <img
          src={previewUrl}
          alt="Selected preview"
          className="w-full h-auto rounded-lg object-cover max-h-96"
        />
      </div>
    </div>
  )
}
