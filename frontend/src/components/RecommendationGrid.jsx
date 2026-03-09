/**
 * RecommendationGrid Component
 * 
 * Displays similar fashion items returned from the API
 * Shows images in a responsive grid with similarity scores
 * 
 * Props:
 * - recommendations: Array of recommendation objects from API
 *   Each object contains: image_path, score (similarity score)
 */
export default function RecommendationGrid({ recommendations }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Header */}
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        ✨ Recommended Items
      </h2>

      {/* Recommendations Grid - 2 columns on mobile, 3 on larger screens */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {recommendations && recommendations.length > 0 ? (
          recommendations.map((item, index) => (
            <div key={index} className="bg-gray-50 rounded-lg overflow-hidden hover:shadow-lg transition-shadow">
              {/* Item Image */}
              <div className="relative overflow-hidden bg-gray-200 h-48">
                <img
                  src={item.image_path}
                  alt={`Recommendation ${index + 1}`}
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-200"
                  onError={(e) => {
                    // Fallback if image path doesn't work
                    e.target.src = 'https://via.placeholder.com/200?text=No+Image'
                  }}
                />
              </div>

              {/* Item Details */}
              <div className="p-3">
                {/* Similarity Score */}
                {item.score !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Similarity</span>
                    <span className="text-lg font-bold text-green-600">
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>
                )}

                {/* Rank/Position */}
                <div className="text-xs text-gray-500 mt-2">
                  #{ index + 1}
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="col-span-full text-center text-gray-500 py-8">
            No recommendations available
          </div>
        )}
      </div>

      {/* Summary */}
      {recommendations && recommendations.length > 0 && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-gray-700">
            Found <span className="font-bold text-blue-600">{recommendations.length}</span> similar items
            based on visual similarity
          </p>
        </div>
      )}
    </div>
  )
}
