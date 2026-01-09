import React from 'react';
import { FiClock, FiCheck, FiX } from 'react-icons/fi';

const ReviewStats = ({ stats }) => {
  const total = stats.pending + stats.approved + stats.rejected;
  const approvalRate = total > 0 ? ((stats.approved / total) * 100).toFixed(1) : 0;

  const statCards = [
    {
      label: 'Pending Reviews',
      value: stats.pending,
      icon: FiClock,
      color: 'from-yellow-500 to-orange-500',
      bgColor: 'bg-yellow-900/20',
      borderColor: 'border-yellow-500/30',
    },
    {
      label: 'Approved',
      value: stats.approved,
      icon: FiCheck,
      color: 'from-green-500 to-emerald-500',
      bgColor: 'bg-green-900/20',
      borderColor: 'border-green-500/30',
    },
    {
      label: 'Rejected',
      value: stats.rejected,
      icon: FiX,
      color: 'from-red-500 to-pink-500',
      bgColor: 'bg-red-900/20',
      borderColor: 'border-red-500/30',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      {statCards.map((stat, index) => {
        const Icon = stat.icon;
        return (
          <div
            key={index}
            className={`${stat.bgColor} border ${stat.borderColor} rounded-lg p-6 backdrop-blur-sm hover:border-opacity-100 transition duration-200`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm font-medium mb-1">
                  {stat.label}
                </p>
                <p className={`text-3xl font-bold bg-gradient-to-r ${stat.color} bg-clip-text text-transparent`}>
                  {stat.value}
                </p>
              </div>
              <div className={`bg-gradient-to-r ${stat.color} p-3 rounded-lg`}>
                <Icon size={24} className="text-white" />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default ReviewStats;
