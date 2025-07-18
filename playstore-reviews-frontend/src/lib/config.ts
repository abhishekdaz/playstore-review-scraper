export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  search: `${API_BASE_URL}/search`,
  reviews: `${API_BASE_URL}/reviews`,
  analyze: `${API_BASE_URL}/analyze`,
  analyzeDirect: `${API_BASE_URL}/analyze/direct`,
  analyzeSentimentSeparation: `${API_BASE_URL}/analyze/sentiment_separation`,
  reviewsCsv: `${API_BASE_URL}/reviews/csv`,
} as const; 