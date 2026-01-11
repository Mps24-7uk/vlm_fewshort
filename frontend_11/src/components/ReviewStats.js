import React from 'react';
import { FiCheckCircle, FiXCircle, FiClock } from 'react-icons/fi';

const ReviewStats = ({ stats }) => {
  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center gap-3">
          <FiClock className="text-yellow-500" size={24} />
          <div>
            <p className="text-gray-400 text-sm">Pending</p>
            <p className="text-white text-2xl font-bold">{stats.pending || 0}</p>
          </div>
        </div>
      </div>
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center gap-3">
          <FiCheckCircle className="text-green-500" size={24} />
          <div>
            <p className="text-gray-400 text-sm">Approved</p>
            <p className="text-white text-2xl font-bold">{stats.approved || 0}</p>
          </div>
        </div>
      </div>
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center gap-3">
          <FiXCircle className="text-red-500" size={24} />
          <div>
            <p className="text-gray-400 text-sm">Rejected</p>
            <p className="text-white text-2xl font-bold">{stats.rejected || 0}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReviewStats;
