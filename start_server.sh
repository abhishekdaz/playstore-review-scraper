#!/bin/bash

# Activate virtual environment and start the server
source venv/bin/activate
echo "Starting Play Store Review Scraper API..."
echo "Server will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
uvicorn main:app --reload --port 8000 