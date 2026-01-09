import React, { useState, useEffect } from 'react';
import HumanReviewDashboard from './pages/HumanReviewDashboard';
import './App.css';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800">
      <HumanReviewDashboard />
    </div>
  );
}

export default App;
