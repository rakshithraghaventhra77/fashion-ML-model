import { useRef } from 'react'

/**
 * UploadBox Component
 * 
 * Provides a drag-and-drop area for image uploads
 * Supports both click-to-select and drag-and-drop
 * 
 * Props:
 * - onFileSelect: Function called when file is selected (receives File object)
 */
export default function UploadBox({ onFileSelect }) {
  // Reference to hidden file input element
  const fileInputRef = useRef(null)

  /**
   * Handles click event - opens file picker dialog
   */
  const handleClick = () => {
    fileInputRef.current?.click()
  }

  /**
   * Handles file selection from file input
   * Validates that selected file is an image
   */
  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      onFileSelect(file)
    } else if (file) {
      alert('Please select an image file')
    }
  }

  /**
   * Prevents default drag behavior and allows drop
   */
  const handleDragOver = (event) => {
    event.preventDefault()
    event.stopPropagation()
    event.currentTarget.classList.add('border-blue-500', 'bg-blue-50')
  }

  /**
   * Removes visual feedback when dragging leaves the area
   */
  const handleDragLeave = (event) => {
    event.preventDefault()
    event.currentTarget.classList.remove('border-blue-500', 'bg-blue-50')
  }

  /**
   * Handles dropped files
   * Validates that dropped file is an image
   */
  const handleDrop = (event) => {
    event.preventDefault()
    event.stopPropagation()
    event.currentTarget.classList.remove('border-blue-500', 'bg-blue-50')

    // Get file from drop event
    const file = event.dataTransfer?.files?.[0]
    if (file && file.type.startsWith('image/')) {
      onFileSelect(file)
    } else if (file) {
      alert('Please drop an image file')
    }
  }

  return (
    <div
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center bg-white hover:bg-gray-50 cursor-pointer transition-all duration-200"
    >
      {/* Hidden file input element */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {/* Upload Icon */}
      <div className="text-5xl mb-4">📤</div>

      {/* Upload Text */}
      <p className="text-lg font-semibold text-gray-700 mb-2">
        Drag and drop an image here
      </p>
      <p className="text-gray-500">
        or click to select from your computer
      </p>
    </div>
  )
}
