/**
 * Play Store Review Scraper - Main Page Component
 * 
 * Enhanced with comprehensive NLP analysis capabilities including:
 * - Advanced sentiment analysis with confidence scores
 * - Issue and feature request detection
 * - Topic modeling and theme extraction
 * - Actionable insights and recommendations
 * - Executive summary with health assessment
 * 
 * Features:
 * - Dark theme with glassmorphism design
 * - Multi-select star rating filters
 * - Scalable review count options (100-5000)
 * - Real-time search and filtering
 * - CSV export functionality
 * - Comprehensive analysis dashboard
 */

"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";

// ===== INTERFACES =====
interface Review {
  rating: number;
  review: string;
  date: string;
  sentiment_analysis?: {
    final_sentiment: string;
    confidence: number;
    vader_sentiment: string;
    textblob_sentiment: string;
  };
  issues?: string[];
  features?: string[];
  classification?: string;
}

interface SearchResult {
  title: string;
  appId: string;
  developer: string;
  summary: string;
  icon: string;
  score: number;
  scoreText: string;
  priceText: string;
  free: boolean;
  url: string;
  ratings?: number; // Total number of ratings
  screenshots?: string[]; // App screenshots  
}

interface AnalysisInsights {
  priority_issues: Array<{
    issue_type: string;
    frequency: number;
    percentage: number;
    severity_level: string;
    priority_score: number;
  }>;
  top_feature_requests: Array<{
    feature_type: string;
    requests: number;
    percentage: number;
    priority_level: string;
    combined_score: number;
  }>;
  sentiment_trends: {
    overall_sentiment: string;
    sentiment_distribution: Record<string, number>;
    average_confidence: number;
  };
  recommendations: Array<{
    category: string;
    action: string;
    priority: string;
    impact: string;
    timeline: string;
    affected_users: string;
  }>;
  key_themes: Array<{
    theme_name: string;
    keywords: string[];
    review_count: number;
    percentage: number;
  }>;
  urgency_score: number;
  summary: {
    total_reviews_analyzed: number;
    overall_health: string;
    main_concern: string;
    top_opportunity: string;
    action_required: boolean;
  };
}

interface AnalysisResults {
  meta: {
    total_reviews: number;
    valid_reviews: number;
    analysis_timestamp: string;
    app_id: string;
  };
  sentiment_analysis: {
    reviews_with_sentiment: Review[];
    sentiment_counts: Record<string, number>;
    overall_sentiment: string;
    average_confidence: number;
  };
  insights: AnalysisInsights;
  statistics: {
    avg_rating: number;
    rating_distribution: Record<string, number>;
    avg_review_length: number;
  };
}

// ===== CONSTANTS =====
const REVIEW_COUNT_OPTIONS = [
  { value: 100, label: "100 reviews" },
  { value: 500, label: "500 reviews" },
  { value: 1000, label: "1,000 reviews" },
  { value: 2000, label: "2,000 reviews" },
  { value: 5000, label: "5,000 reviews" }
];

const STAR_RATINGS = [1, 2, 3, 4, 5];

// Removed unused API_BASE constant

export default function Home() {
  // ===== STATE =====
  const [activeTab, setActiveTab] = useState<'search' | 'analysis'>('search');
  const [url, setUrl] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [reviewCount, setReviewCount] = useState(100);
  const [selectedStars, setSelectedStars] = useState<number[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  
  // Results state
  const [reviews, setReviews] = useState<Review[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [totalReviews, setTotalReviews] = useState(0);
  const [appId, setAppId] = useState("");

  // App info state  
  const [selectedApp, setSelectedApp] = useState<SearchResult | null>(null);

  // Removed unused app type configuration state variables

  // ===== HELPER FUNCTIONS =====
  const toggleStarFilter = useCallback((star: number) => {
    setSelectedStars(prev => 
      prev.includes(star)
        ? prev.filter(s => s !== star)
        : [...prev, star]
    );
  }, []);

  const clearStarFilters = useCallback(() => {
    setSelectedStars([]);
  }, []);

  const getSentimentColor = (sentiment: string): string => {
    switch (sentiment) {
      case 'positive': return 'bg-green-600';
      case 'negative': return 'bg-red-600';
      case 'neutral': return 'bg-yellow-600';
      default: return 'bg-gray-600';
    }
  };

  const getSeverityColor = (severity: string): string => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'bg-red-600';
      case 'high': return 'bg-orange-600';
      case 'medium': return 'bg-yellow-600';
      case 'low': return 'bg-green-600';
      default: return 'bg-gray-600';
    }
  };

  const getHealthColor = (health: string): string => {
    switch (health.toLowerCase()) {
      case 'excellent': return 'text-green-400';
      case 'good': return 'text-blue-400';
      case 'fair': return 'text-yellow-400';
      case 'poor': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  // Removed unused app type configuration functions

  // ===== API FUNCTIONS =====
  const searchApps = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    setError("");
    
    try {
      const response = await fetch("http://localhost:8000/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, limit: 10 })
      });
      
      if (!response.ok) throw new Error("Search failed");
      
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setIsSearching(false);
    }
  };

  const fetchReviews = async () => {
    if (!url.trim()) {
      setError("Please enter a valid Play Store URL");
      return;
    }

    setIsLoading(true);
    setError("");
    
    try {
      const payload = {
        url,
        count: reviewCount,
        star_filters: selectedStars.length > 0 ? selectedStars : undefined
      };

      const response = await fetch("http://localhost:8000/reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error("Failed to fetch reviews");

      const data = await response.json();
      setReviews(data.reviews || []);
      setTotalReviews(data.total_reviews || 0);
      setAppId(data.app_id || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch reviews");
    } finally {
      setIsLoading(false);
    }
  };

  const runAnalysis = async () => {
    if (!url.trim()) {
      setError("Please enter a valid Play Store URL");
      return;
    }

    setIsAnalyzing(true);
    setError("");
    
    try {
      const payload = {
        url,
        count: Math.min(reviewCount, 1000), // Limit analysis to 1000 reviews for performance
        star_filters: selectedStars.length > 0 ? selectedStars : undefined,
        include_sentiment: true,
        include_topics: true,
        include_classification: true
      };

      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error("Analysis failed");

      const data = await response.json();
      setAnalysisResults(data);
      setActiveTab('analysis');
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const exportToCSV = async () => {
    if (!url.trim()) {
      setError("Please enter a valid Play Store URL");
      return;
    }

    setIsLoading(true);
    
    try {
      const payload = {
        url,
        count: reviewCount,
        star_filters: selectedStars.length > 0 ? selectedStars : undefined
      };

      const response = await fetch("http://localhost:8000/reviews/csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error("Export failed");

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `reviews_${appId || 'export'}_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setIsLoading(false);
    }
  };

  const selectApp = (app: SearchResult) => {
    setUrl(app.url);
    setSelectedApp(app);
    setSearchResults([]);
    setSearchQuery("");
  };

  // ===== RENDER =====
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
            Play Store Review Analyzer
          </h1>
          <p className="text-gray-300 text-lg max-w-3xl mx-auto">
            Extract, analyze, and gain actionable insights from Google Play Store reviews with advanced AI-powered sentiment analysis and theme detection.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="flex bg-slate-800/50 backdrop-blur-sm rounded-lg p-2">
            <button
              onClick={() => setActiveTab('search')}
              className={`px-6 py-2 rounded-md font-medium transition-all duration-200 ${
                activeTab === 'search'
                  ? 'bg-purple-600 text-white shadow-lg'
                  : 'text-gray-300 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              Search & Reviews
            </button>
            <button
              onClick={() => setActiveTab('analysis')}
              className={`px-6 py-2 rounded-md font-medium transition-all duration-200 ${
                activeTab === 'analysis'
                  ? 'bg-purple-600 text-white shadow-lg'
                  : 'text-gray-300 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              AI Analysis
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-300">
            {error}
          </div>
        )}

        {/* Search & Reviews Tab */}
        {activeTab === 'search' && (
          <>
            {/* Main Controls */}
            <Card className="mb-8 bg-slate-800/50 backdrop-blur-sm border-slate-700">
              <CardHeader>
                <CardTitle className="text-2xl text-white">App Search & Configuration</CardTitle>
                <CardDescription className="text-gray-300">
                  Search for apps or enter a Play Store URL directly
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* App Search */}
                <div className="space-y-3">
                  <Label htmlFor="search" className="text-gray-200 font-medium">
                    🔍 Search Play Store Apps
                  </Label>
                  <div className="flex gap-3">
                    <Input
                      id="search"
                      type="text"
                      placeholder="Search for apps (e.g., WhatsApp, Instagram, TikTok)"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && searchApps()}
                      className="flex-1 bg-slate-900/50 border-slate-600 text-white placeholder-gray-400 focus:border-purple-500 focus:ring-purple-500/20"
                    />
                    <Button 
                      onClick={searchApps} 
                      disabled={isSearching || !searchQuery.trim()}
                      className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-medium px-6"
                    >
                      {isSearching ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Searching...
                        </>
                      ) : (
                        'Search'
                      )}
                    </Button>
                  </div>
                </div>

                {/* Search Results */}
                {searchResults.length > 0 && (
                  <div className="space-y-3">
                    <Label className="text-gray-200 font-medium">Search Results</Label>
                    <div className="grid gap-3 max-h-96 overflow-y-auto custom-scrollbar">
                      {searchResults.map((app, index) => (
                        <div
                          key={index}
                          onClick={() => selectApp(app)}
                          className="flex items-center gap-4 p-4 bg-slate-900/30 border border-slate-600 rounded-lg hover:bg-slate-700/30 cursor-pointer transition-all duration-200 hover:border-purple-500"
                        >
                          {app.icon && (
                            <img
                              src={app.icon}
                              alt={app.title}
                              className="w-12 h-12 rounded-lg"
                              onError={(e) => {
                                const target = e.target as HTMLImageElement;
                                target.style.display = 'none';
                              }}
                            />
                          )}
                          <div className="flex-1 min-w-0">
                            <h3 className="font-semibold text-white truncate">{app.title}</h3>
                            <p className="text-sm text-gray-400 truncate">{app.developer}</p>
                            <p className="text-xs text-gray-500 line-clamp-2">{app.summary || 'No description available'}</p>
                          </div>
                          <div className="flex flex-col items-end gap-1">
                            <Badge variant="secondary" className="bg-yellow-600/20 text-yellow-400">
                              ⭐ {app.scoreText}
                            </Badge>
                            {app.ratings && (
                              <div className="text-xs text-gray-500">
                                {app.ratings.toLocaleString()} ratings
                              </div>
                            )}
                            <Badge variant="outline" className="text-green-400 border-green-600">
                              {app.priceText}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Selected App Display */}
                {selectedApp && (
                  <div className="p-4 bg-slate-900/30 border border-slate-600 rounded-lg">
                    <Label className="text-gray-200 font-medium mb-3 block">📱 Selected App</Label>
                    <div className="flex items-center gap-4">
                      {selectedApp.icon && (
                        <img
                          src={selectedApp.icon}
                          alt={selectedApp.title}
                          className="w-16 h-16 rounded-xl shadow-lg"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.style.display = 'none';
                          }}
                        />
                      )}
                      <div className="flex-1">
                        <h3 className="text-xl font-bold text-white">{selectedApp.title}</h3>
                        <p className="text-gray-400">{selectedApp.developer}</p>
                        <div className="flex items-center gap-4 mt-2">
                          <Badge variant="secondary" className="bg-yellow-600/20 text-yellow-400">
                            ⭐ {selectedApp.scoreText}
                          </Badge>
                          {selectedApp.ratings && (
                            <span className="text-sm text-gray-400">
                              {selectedApp.ratings.toLocaleString()} total ratings
                            </span>
                          )}
                          <Badge variant="outline" className="text-green-400 border-green-600">
                            {selectedApp.priceText}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Direct URL Input */}
                <div className="space-y-3">
                  <Label htmlFor="url" className="text-gray-200 font-medium">
                    🔗 Play Store URL
                  </Label>
                  <Input
                    id="url"
                    type="url"
                    placeholder="https://play.google.com/store/apps/details?id=com.whatsapp"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    className="bg-slate-900/50 border-slate-600 text-white placeholder-gray-400 focus:border-purple-500 focus:ring-purple-500/20"
                  />
                </div>

                {/* Configuration Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Review Count */}
                  <div className="space-y-3">
                    <Label className="text-gray-200 font-medium">📊 Number of Reviews</Label>
                    <Select value={reviewCount.toString()} onValueChange={(value) => setReviewCount(parseInt(value))}>
                      <SelectTrigger className="bg-slate-900/50 border-slate-600 text-white focus:border-purple-500">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-slate-600">
                        {REVIEW_COUNT_OPTIONS.map((option) => (
                          <SelectItem key={option.value} value={option.value.toString()} className="text-white hover:bg-slate-700">
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Star Rating Filters */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label className="text-gray-200 font-medium">⭐ Filter by Star Rating</Label>
                      {selectedStars.length > 0 && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={clearStarFilters}
                          className="text-gray-400 hover:text-white text-xs"
                        >
                          Clear All
                        </Button>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {STAR_RATINGS.map((star) => (
                        <button
                          key={star}
                          onClick={() => toggleStarFilter(star)}
                          className={`
                            flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200
                            ${selectedStars.includes(star)
                              ? 'bg-purple-600 text-white shadow-lg scale-105'
                              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-600/50 hover:text-white'
                            }
                          `}
                        >
                          <span>⭐</span>
                          <span>{star}</span>
                        </button>
                      ))}
                    </div>
                    {selectedStars.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        <span className="text-sm text-gray-400">Active filters:</span>
                        {selectedStars.map((star) => (
                          <Badge key={star} variant="secondary" className="bg-purple-600/20 text-purple-300">
                            {star}⭐
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-wrap gap-3 pt-4">
                  <Button 
                    onClick={fetchReviews} 
                    disabled={isLoading || !url.trim()}
                    className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-medium px-6"
                  >
                    {isLoading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Fetching Reviews...
                      </>
                    ) : (
                      '📝 Get Reviews'
                    )}
                  </Button>

                  <Button 
                    onClick={runAnalysis} 
                    disabled={isAnalyzing || !url.trim()}
                    className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-medium px-6"
                  >
                    {isAnalyzing ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Analyzing...
                      </>
                    ) : (
                      '🧠 AI Analysis'
                    )}
                  </Button>

                  <Button 
                    onClick={exportToCSV} 
                    disabled={isLoading || !url.trim()}
                    variant="outline"
                    className="border-slate-600 text-gray-300 hover:bg-slate-700/50 hover:text-white font-medium px-6"
                  >
                    📥 Export CSV
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Reviews Display */}
            {reviews.length > 0 && (
              <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle className="text-2xl text-white">Reviews</CardTitle>
                      <CardDescription className="text-gray-300">
                        {totalReviews} reviews found • App ID: {appId}
                      </CardDescription>
                    </div>
                    <div className="text-right">
                      <Badge className="bg-blue-600/20 text-blue-300 text-sm px-3 py-1">
                        {selectedStars.length > 0 ? `Filtered: ${selectedStars.join(', ')}⭐` : 'All ratings'}
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 max-h-96 overflow-y-auto custom-scrollbar">
                    {reviews.map((review, index) => (
                      <div key={index} className="p-4 bg-slate-900/30 border border-slate-600 rounded-lg">
                        <div className="flex justify-between items-start mb-3">
                          <div className="flex items-center gap-2">
                            <Badge 
                              className={`
                                ${review.rating >= 4 ? 'bg-green-600' : 
                                  review.rating >= 3 ? 'bg-yellow-600' : 'bg-red-600'
                                } text-white font-medium
                              `}
                            >
                              {review.rating}⭐
                            </Badge>
                            {review.sentiment_analysis && (
                              <Badge 
                                className={`${getSentimentColor(review.sentiment_analysis.final_sentiment)} text-white text-xs`}
                              >
                                {review.sentiment_analysis.final_sentiment} ({Math.round(review.sentiment_analysis.confidence * 100)}%)
                              </Badge>
                            )}
                          </div>
                          <span className="text-sm text-gray-400">{review.date}</span>
                        </div>
                        <p className="text-gray-200 leading-relaxed mb-2">{review.review}</p>
                        {((review.issues && review.issues.length > 0) || (review.features && review.features.length > 0)) && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {review.issues && review.issues.map((issue, i) => (
                              <Badge key={i} variant="destructive" className="text-xs">
                                🚨 {issue.replace(/_/g, ' ')}
                              </Badge>
                            ))}
                            {review.features && review.features.map((feature, i) => (
                              <Badge key={i} variant="secondary" className="text-xs bg-blue-600/20 text-blue-300">
                                💡 {feature.replace(/_/g, ' ')}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}

        {/* Analysis Tab */}
        {activeTab === 'analysis' && analysisResults && (
          <div className="space-y-6">
            {/* Executive Summary */}
            <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
              <CardHeader>
                <CardTitle className="text-2xl text-white">📊 Executive Summary</CardTitle>
                <CardDescription className="text-gray-300">
                  AI-powered analysis of {analysisResults.meta.total_reviews} reviews
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="text-center">
                    <div className={`text-3xl font-bold ${getHealthColor(analysisResults.insights.summary?.overall_health || 'Unknown')}`}>
                      {analysisResults.insights.summary?.overall_health || 'Unknown'}
                    </div>
                    <div className="text-sm text-gray-400">App Health</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-400">
                      {analysisResults.insights.urgency_score}/100
                    </div>
                    <div className="text-sm text-gray-400">Urgency Score</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-400">
                      {analysisResults.statistics.avg_rating.toFixed(1)}⭐
                    </div>
                    <div className="text-sm text-gray-400">Average Rating</div>
                  </div>
                  <div className="text-center">
                    <div className={`text-3xl font-bold ${getSentimentColor(analysisResults.insights.sentiment_trends.overall_sentiment)}`}>
                      {analysisResults.insights.sentiment_trends.overall_sentiment}
                    </div>
                    <div className="text-sm text-gray-400">Overall Sentiment</div>
                  </div>
                </div>
                
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-slate-900/30 rounded-lg">
                    <div className="text-sm text-gray-400 mb-1">Main Concern</div>
                    <div className="text-white font-medium">{analysisResults.insights.summary?.main_concern || 'No major concerns identified'}</div>
                  </div>
                  <div className="p-4 bg-slate-900/30 rounded-lg">
                    <div className="text-sm text-gray-400 mb-1">Top Opportunity</div>
                    <div className="text-white font-medium">{analysisResults.insights.summary?.top_opportunity || 'No opportunities identified'}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Priority Issues */}
            {analysisResults.insights.priority_issues?.length > 0 && (
              <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
                <CardHeader>
                  <CardTitle className="text-2xl text-white">🚨 Priority Issues</CardTitle>
                  <CardDescription className="text-gray-300">
                    Critical problems that need immediate attention
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {analysisResults.insights.priority_issues.slice(0, 5).map((issue, index) => (
                      <div key={index} className="flex items-center justify-between p-4 bg-slate-900/30 rounded-lg">
                        <div className="flex items-center gap-4">
                          <div className="text-2xl font-bold text-gray-400">#{index + 1}</div>
                          <div>
                            <div className="font-semibold text-white">{issue.issue_type}</div>
                            <div className="text-sm text-gray-400">
                              {issue.frequency} reports ({issue.percentage}% of reviews)
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Badge className={`${getSeverityColor(issue.severity_level)} text-white`}>
                            {issue.severity_level}
                          </Badge>
                          <div className="text-right">
                            <div className="text-lg font-bold text-purple-400">{issue.priority_score}</div>
                            <div className="text-xs text-gray-400">Priority Score</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Feature Requests */}
            {analysisResults.insights.top_feature_requests?.length > 0 && (
              <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
                <CardHeader>
                  <CardTitle className="text-2xl text-white">💡 Top Feature Requests</CardTitle>
                  <CardDescription className="text-gray-300">
                    Most requested features from user feedback
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {analysisResults.insights.top_feature_requests.slice(0, 5).map((feature, index) => (
                      <div key={index} className="flex items-center justify-between p-4 bg-slate-900/30 rounded-lg">
                        <div className="flex items-center gap-4">
                          <div className="text-2xl font-bold text-gray-400">#{index + 1}</div>
                          <div>
                            <div className="font-semibold text-white">{feature.feature_type}</div>
                            <div className="text-sm text-gray-400">
                              {feature.requests} requests ({feature.percentage}% of reviews)
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Badge className={`${
                            feature.priority_level === 'High' ? 'bg-green-600' :
                            feature.priority_level === 'Medium' ? 'bg-yellow-600' : 'bg-gray-600'
                          } text-white`}>
                            {feature.priority_level}
                          </Badge>
                          <div className="text-right">
                            <div className="text-lg font-bold text-blue-400">{feature.combined_score}</div>
                            <div className="text-xs text-gray-400">Combined Score</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            {analysisResults.insights.recommendations?.length > 0 && (
              <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
                <CardHeader>
                  <CardTitle className="text-2xl text-white">🎯 Actionable Recommendations</CardTitle>
                  <CardDescription className="text-gray-300">
                    Data-driven suggestions for improvement
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {analysisResults.insights.recommendations.map((rec, index) => (
                      <div key={index} className="p-4 bg-slate-900/30 rounded-lg">
                        <div className="flex justify-between items-start mb-2">
                          <Badge className={`${
                            rec.priority === 'Critical' ? 'bg-red-600' :
                            rec.priority === 'High' ? 'bg-orange-600' :
                            rec.priority === 'Medium-High' ? 'bg-yellow-600' : 'bg-blue-600'
                          } text-white`}>
                            {rec.priority}
                          </Badge>
                          <span className="text-xs text-gray-400">{rec.category}</span>
                        </div>
                        <div className="font-semibold text-white mb-2">{rec.action}</div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-gray-400">Impact:</span>
                            <span className="text-white ml-2">{rec.impact}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Timeline:</span>
                            <span className="text-white ml-2">{rec.timeline}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Affected:</span>
                            <span className="text-white ml-2">{rec.affected_users}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Sentiment Analysis */}
            <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
              <CardHeader>
                <CardTitle className="text-2xl text-white">🎭 Sentiment Analysis</CardTitle>
                <CardDescription className="text-gray-300">
                  Emotional tone and confidence metrics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Sentiment Distribution */}
                  <div>
                    <h4 className="font-semibold text-white mb-4">Sentiment Distribution</h4>
                    <div className="space-y-3">
                      {analysisResults.insights.sentiment_trends?.sentiment_distribution && Object.entries(analysisResults.insights.sentiment_trends.sentiment_distribution).map(([sentiment, count]) => (
                        <div key={sentiment} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className={`w-3 h-3 rounded-full ${getSentimentColor(sentiment)}`}></div>
                            <span className="text-white capitalize">{sentiment}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-gray-400">{count} reviews</span>
                            <span className="text-sm text-gray-500">
                              ({Math.round((count / analysisResults.meta.total_reviews) * 100)}%)
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Key Metrics */}
                  <div>
                    <h4 className="font-semibold text-white mb-4">Key Metrics</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Average Confidence:</span>
                        <span className="text-white font-medium">
                          {Math.round((analysisResults.insights.sentiment_trends?.average_confidence || 0) * 100)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Overall Sentiment:</span>
                        <Badge className={`${getSentimentColor(analysisResults.insights.sentiment_trends?.overall_sentiment || 'neutral')} text-white`}>
                          {analysisResults.insights.sentiment_trends?.overall_sentiment || 'Unknown'}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Reviews Analyzed:</span>
                        <span className="text-white font-medium">{analysisResults.meta?.total_reviews || 0}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Key Themes */}
            {analysisResults.insights.key_themes?.length > 0 && (
              <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
                <CardHeader>
                  <CardTitle className="text-2xl text-white">🏷️ Key Themes</CardTitle>
                  <CardDescription className="text-gray-300">
                    Main topics and themes extracted from reviews
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {analysisResults.insights.key_themes.map((theme, index) => (
                      <div key={index} className="p-4 bg-slate-900/30 rounded-lg">
                        <div className="font-semibold text-white mb-2">{theme.theme_name}</div>
                        <div className="text-sm text-gray-400 mb-3">
                          {theme.review_count} reviews ({theme.percentage}%)
                        </div>
                        <div className="flex flex-wrap gap-1 mb-3">
                          {theme.keywords.map((keyword, i) => (
                            <Badge key={i} variant="outline" className="text-xs text-gray-300 border-gray-600">
                              {keyword}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* Empty Analysis State */}
        {activeTab === 'analysis' && !analysisResults && (
          <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
            <CardContent className="text-center py-12">
              <div className="text-6xl mb-4">🧠</div>
              <h3 className="text-xl font-semibold text-white mb-2">No Analysis Yet</h3>
              <p className="text-gray-400 mb-6">
                Run an AI analysis to get comprehensive insights, sentiment analysis, and actionable recommendations.
              </p>
              <Button 
                onClick={() => setActiveTab('search')}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-medium px-6"
              >
                Go to Search & Reviews
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
