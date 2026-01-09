import React, { useState } from 'react';
import { FiCheck, FiX, FiLoader } from 'react-icons/fi';

const ReviewCard = ({ item, onReview }) => {
  const [deciding, setDeciding] = useState(false);
  const [localError, setLocalError] = useState(null);

  const handleDecision = async (decision) => {
    try {
      setDeciding(true);
      setLocalError(null);
      await onReview(item.id, decision);
    } catch (err) {
      setLocalError('Failed to submit decision');
      setDeciding(false);
    }
  };

  const confidenceColor = 
    item.confidence >= 0.8 ? 'text-green-400' :
    item.confidence >= 0.6 ? 'text-yellow-400' :
    'text-red-400';

  const confidenceBarColor = 
    item.confidence >= 0.8 ? 'bg-green-500' :
    item.confidence >= 0.6 ? 'bg-yellow-500' :
    'bg-red-500';

  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden shadow-2xl border border-gray-700 hover:border-blue-500 transition duration-300 h-full flex flex-col">
      {/* Image Container */}
      <div className="relative bg-gray-900 overflow-hidden aspect-square">
        <img
          src={item.image_path}
          alt={`Review item ${item.id}`}
          className="w-full h-full object-cover hover:scale-105 transition duration-300"
        />
        <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-md px-3 py-1 rounded-full">
          <p className="text-white text-sm font-semibold">#{item.id}</p>
        </div>
      </div>

      {/* Content */}
      <div className="p-6 flex-1 flex flex-col">
        {/* Prediction Info */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <label className="text-gray-400 text-sm font-semibold">
              🎯 PREDICTION
            </label>
            <span className={`text-xl font-bold ${confidenceColor}`}>
              {(item.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <h2 className="text-2xl font-bold text-white mb-4">
            {item.predicted_class}
          </h2>

          {/* Confidence Bar */}
          <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
            <div
              className={`h-full ${confidenceBarColor} transition-all duration-300`}
              style={{ width: `${item.confidence * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Reason/Details */}
        <div className="mb-6 pb-6 border-b border-gray-700">
          <label className="text-gray-400 text-sm font-semibold block mb-2">
            📋 DETAILS
          </label>
          <p className="text-gray-300 text-sm leading-relaxed">
            {item.reason || 'Requires human verification'}
          </p>
        </div>

        {/* Additional Info */}
        {item.neighbor_labels && (
          <div className="mb-6 pb-6 border-b border-gray-700">
            <label className="text-gray-400 text-sm font-semibold block mb-3">
              🔍 SIMILAR SAMPLES
            </label>
            <div className="flex gap-2 flex-wrap">
              {item.neighbor_labels.slice(0, 5).map((label, idx) => (
                <span
                  key={idx}
                  className="bg-blue-900/40 text-blue-300 px-3 py-1 rounded-lg text-xs font-medium border border-blue-700/50"
                >
                  {label}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Error Message */}
        {localError && (
          <div className="mb-4 bg-red-900/20 border border-red-500/50 text-red-300 text-sm px-3 py-2 rounded">
            {localError}
          </div>
        )}

        {/* Decision Buttons */}
        <div className="flex gap-4 mt-auto">
          <button
            onClick={() => handleDecision('approve')}
            disabled={deciding}
            className="flex-1 group relative overflow-hidden bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white font-bold py-4 px-6 rounded-xl transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-green-500/50"
          >
            <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition duration-300"></div>
            <div className="relative flex items-center justify-center gap-3">
              {deciding ? (
                <>
                  <FiLoader className="animate-spin" size={20} />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <FiCheck size={22} className="group-hover:scale-125 transition duration-300" />
                  <span className="text-lg">Approve</span>
                </>
              )}
            </div>
          </button>

          <button
            onClick={() => handleDecision('reject')}
            disabled={deciding}
            className="flex-1 group relative overflow-hidden bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white font-bold py-4 px-6 rounded-xl transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-red-500/50"
          >
            <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition duration-300"></div>
            <div className="relative flex items-center justify-center gap-3">
              {deciding ? (
                <>
                  <FiLoader className="animate-spin" size={20} />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <FiX size={22} className="group-hover:scale-125 transition duration-300" />
                  <span className="text-lg">Reject</span>
                </>
              )}
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ReviewCard;
