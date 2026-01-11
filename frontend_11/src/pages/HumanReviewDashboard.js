import React, { useState, useEffect } from 'react';
import ReviewCard from '../components/ReviewCard';
import ReviewStats from '../components/ReviewStats';
import api from '../services/api';
import { FiRefreshCw, FiAlertCircle } from 'react-icons/fi';

const HumanReviewDashboard = () => {
  const [reviewItems, setReviewItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    pending: 0,
    approved: 0,
    rejected: 0,
  });

  useEffect(() => {
    fetchReviewItems();
  }, []);

  const fetchReviewItems = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await api.get('/api/review/items');
      setReviewItems(response.data.items || []);
      setStats(response.data.stats || { pending: 0, approved: 0, rejected: 0 });
    } catch (err) {
      setError('Failed to fetch review items. Please try again.');
      console.error('Error fetching review items:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReview = async (itemId, decision) => {
    try {
      const response = await api.post(`/api/review/${itemId}/decision`, {
        decision,
      });

      // Update the review items
      setReviewItems(reviewItems.filter(item => item.id !== itemId));

      // Update stats
      setStats(response.data.stats);
    } catch (err) {
      setError('Failed to submit review decision. Please try again.');
      console.error('Error submitting review:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                👤 Human Review Dashboard
              </h1>
              <p className="text-gray-400">
                Review predictions that require human verification
              </p>
            </div>
            <button
              onClick={fetchReviewItems}
              disabled={loading}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition duration-200 disabled:opacity-50"
            >
              <FiRefreshCw className={loading ? 'animate-spin' : ''} />
              Refresh
            </button>
          </div>

          {/* Stats */}
          <ReviewStats stats={stats} />
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-8 bg-red-900/20 border border-red-500 text-red-300 px-6 py-4 rounded-lg flex items-start gap-3">
            <FiAlertCircle className="mt-1 flex-shrink-0" size={20} />
            <div>
              <h3 className="font-semibold mb-1">Error</h3>
              <p>{error}</p>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-16">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-gray-400 text-lg">Loading review items...</p>
          </div>
        )}

        {/* Empty State */}
        {!loading && reviewItems.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center py-16">
            <div className="text-6xl mb-4">✅</div>
            <p className="text-gray-400 text-lg">No pending reviews</p>
            <p className="text-gray-500 text-sm">All predictions have been reviewed!</p>
          </div>
        )}

        {/* Review Items Grid */}
        {!loading && reviewItems.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {reviewItems.map(item => (
              <ReviewCard
                key={item.id}
                item={item}
                onReview={handleReview}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default HumanReviewDashboard;
