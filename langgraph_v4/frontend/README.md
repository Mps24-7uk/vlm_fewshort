# Human Review Frontend

A modern React.js and Tailwind CSS frontend for the False Positive Detection System's human review component.

## Features

- 🎨 Modern, responsive UI with Tailwind CSS
- 📊 Real-time review statistics dashboard
- 🖼️ Image display with prediction details
- ⚡ Fast decision submission (approve/reject)
- 📱 Mobile-friendly design
- 🔄 Real-time updates with axios

## Setup Instructions

### Prerequisites

- Node.js 14+ and npm
- Backend API running on `http://localhost:5000`

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (or update the existing one):
```env
REACT_APP_API_URL=http://localhost:5000
```

### Running the Application

**Development Mode:**
```bash
npm start
```

The application will open in your browser at `http://localhost:3000`

**Production Build:**
```bash
npm run build
```

The optimized build will be in the `build/` directory.

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── ReviewCard.js       # Individual review card component
│   │   └── ReviewStats.js      # Statistics display component
│   ├── pages/
│   │   └── HumanReviewDashboard.js  # Main dashboard page
│   ├── services/
│   │   └── api.js              # Axios API client
│   ├── App.js                  # Root component
│   ├── App.css                 # App styles
│   ├── index.js                # Entry point
│   └── index.css               # Global styles
├── package.json
├── tailwind.config.js
└── README.md
```

## Component Details

### ReviewCard
Displays individual items requiring human review with:
- Image preview
- Prediction class and confidence score
- Confidence visualization bar
- Similar sample labels
- Approve/Reject buttons

### ReviewStats
Shows dashboard statistics:
- Pending reviews count
- Approved reviews count
- Rejected reviews count
- Visual progress indicators

### HumanReviewDashboard
Main page component that:
- Fetches review items from the backend
- Manages review state
- Handles user decisions
- Updates statistics in real-time

## API Integration

The frontend communicates with the backend using these endpoints:

**GET /api/review/items**
Returns pending review items and statistics

**POST /api/review/{id}/decision**
Submits a review decision
```json
{
  "decision": "approve" | "reject"
}
```

**GET /api/review/stats**
Returns overall review statistics

## Customization

### Colors and Styling
Edit `tailwind.config.js` to customize the color scheme and theme.

### API Base URL
Change `REACT_APP_API_URL` in `.env` to point to your backend server.

### Refresh Interval
Modify the `useEffect` in `HumanReviewDashboard.js` to add auto-refresh functionality.

## Troubleshooting

**CORS Errors:**
Ensure the backend has CORS enabled for `http://localhost:3000`

**Images Not Loading:**
Verify that image paths in the API response are accessible and correct

**API Connection Failed:**
Check that the backend is running on the configured `REACT_APP_API_URL`

## License

This project is part of the False Positive Detection System.
