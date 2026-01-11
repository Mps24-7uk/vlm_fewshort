import React, { useState } from 'react';
import { FiCheck, FiX, FiLoader } from 'react-icons/fi';

const ReviewCard = ({ item, onReview }) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleDecision = async (decision) => {
    setIsLoading(true);
    try {
      await onReview(item.id, decision);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-500 transition-colors">
      {/* Image Display */}
      {item.image_path && (
        <div className="mb-4">
          <p className="text-sm text-gray-400 mb-2">Image:</p>
          <div className="bg-gray-900 rounded p-3 max-h-64 overflow-auto">
            <p className="text-gray-300 font-mono text-sm break-words">
              {item.image_path}
            </p>
          </div>
        </div>
      )}

      {/* Prediction Details */}
      {item.predicted_class && (
        <div className="mb-3">
          <p className="text-sm text-gray-400">Predicted Class:</p>
          <p className="text-white font-semibold text-lg">{item.predicted_class}</p>
        </div>
      )}

      {/* Reason/Description */}
      {item.reason && (
        <div className="mb-4">
          <p className="text-sm text-gray-400 mb-1">Reason:</p>
          <p className="text-gray-300 text-sm bg-gray-900/50 p-2 rounded">
            {item.reason}
          </p>
        </div>
      )}

      {/* Confidence Score */}
      {item.confidence !== undefined && (
        <div className="mb-4">
          <p className="text-sm text-gray-400 mb-1">Confidence:</p>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full"
              style={{ width: `${item.confidence * 100}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-1">{(item.confidence * 100).toFixed(1)}%</p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3 pt-4 border-t border-gray-700">
        <button
          onClick={() => handleDecision('approve')}
          disabled={isLoading}
          className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
        >
          {isLoading ? (
            <FiLoader className="animate-spin" size={18} />
          ) : (
            <FiCheck size={18} />
          )}
          Approve
        </button>
        <button
          onClick={() => handleDecision('reject')}
          disabled={isLoading}
          className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
        >
          {isLoading ? (
            <FiLoader className="animate-spin" size={18} />
          ) : (
            <FiX size={18} />
          )}
          Reject
        </button>
      </div>
    </div>
  );
};

export default ReviewCard;
