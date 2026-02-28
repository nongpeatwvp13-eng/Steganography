#!/bin/bash
# Start the Flask backend in the background
python Backend/app.py &
BACKEND_PID=$!

# Start the Vite frontend dev server (root is already set to Frontend in vite.config.js)
npm run dev

# Clean up backend when frontend exits
kill $BACKEND_PID 2>/dev/null
