# Play Store Review Scraper API
# A FastAPI application that provides endpoints for searching Play Store apps and retrieving reviews
# Features: App search, review extraction, CSV export, and comprehensive NLP analysis

import logging
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from google_play_scraper import reviews, Sort, search, app
import re
import csv
import io
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from analysis_engine import ReviewAnalysisEngine, CategoryConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Analysis engine will be created below
ANALYSIS_ENABLED = True

# ===== APPLICATION SETUP =====
app = FastAPI(
    title="Play Store Review Scraper API",
    description="API for searching Play Store apps and extracting reviews with advanced analysis capabilities",
    version="2.0.0"
)

# Enable CORS for all origins (configure as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive analysis tasks
executor = ThreadPoolExecutor(max_workers=2)

# Initialize analysis engine with default categories
analysis_engine = ReviewAnalysisEngine()

# Define configurations for different app types
APP_TYPE_CONFIGS = {
    "social_media": {
        "privacy_concerns": CategoryConfig(
            name="Privacy Concerns",
            keywords=["privacy", "data", "tracking", "personal information", "share data"],
            subcategories={
                "data_tracking": ["tracking", "data collection", "personal data", "privacy policy"],
                "unwanted_sharing": ["share without permission", "public profile", "data sharing"],
                "account_security": ["account hacked", "security breach", "unauthorized access"]
            },
            alert_threshold=0.08,
            priority_multiplier=1.5
        ),
        "content_moderation": CategoryConfig(
            name="Content Issues",
            keywords=["inappropriate", "spam", "harassment", "bullying", "fake news"],
            subcategories={
                "harassment": ["bullying", "harassment", "abuse", "toxic", "hate"],
                "spam_content": ["spam", "fake posts", "bot accounts", "irrelevant content"],
                "inappropriate_content": ["inappropriate", "offensive", "explicit", "disturbing"]
            },
            alert_threshold=0.1,
            priority_multiplier=1.3
        )
    },
    
    "ecommerce": {
        "payment_issues": CategoryConfig(
            name="Payment Problems",
            keywords=["payment", "checkout", "credit card", "billing", "refund", "transaction"],
            subcategories={
                "payment_failures": ["payment failed", "transaction error", "card declined"],
                "billing_issues": ["wrong charge", "duplicate charge", "billing error"],
                "refund_problems": ["refund denied", "no refund", "refund process"]
            },
            alert_threshold=0.05,
            priority_multiplier=2.0
        ),
        "delivery_issues": CategoryConfig(
            name="Delivery & Shipping",
            keywords=["delivery", "shipping", "order", "tracking", "delayed", "lost package"],
            subcategories={
                "delayed_delivery": ["late delivery", "delayed", "slow shipping"],
                "lost_packages": ["lost package", "not delivered", "missing order"],
                "tracking_issues": ["tracking not working", "no tracking", "tracking error"]
            },
            alert_threshold=0.12,
            priority_multiplier=1.4
        )
    },
    
    "productivity": {
        "data_sync": CategoryConfig(
            name="Data Synchronization",
            keywords=["sync", "backup", "cloud", "data loss", "synchronization"],
            subcategories={
                "sync_failures": ["sync failed", "not syncing", "sync error"],
                "data_loss": ["lost data", "data disappeared", "backup failed"],
                "cloud_issues": ["cloud not working", "cloud sync", "cloud storage"]
            },
            alert_threshold=0.08,
            priority_multiplier=1.6
        ),
        "collaboration": CategoryConfig(
            name="Collaboration Features",
            keywords=["share", "collaborate", "team", "permissions", "access"],
            subcategories={
                "sharing_issues": ["can't share", "sharing not working", "share failed"],
                "permission_problems": ["no access", "permission denied", "can't edit"],
                "team_features": ["team features", "collaboration", "group work"]
            },
            alert_threshold=0.1,
            priority_multiplier=1.2
        )
    },
    
    "gaming": {
        "gameplay_issues": CategoryConfig(
            name="Gameplay Problems",
            keywords=["lag", "controls", "gameplay", "levels", "difficulty", "mechanics"],
            subcategories={
                "control_issues": ["controls", "touch response", "input lag", "unresponsive"],
                "level_issues": ["level broken", "can't progress", "stuck on level"],
                "game_mechanics": ["unfair", "balance issues", "game mechanics", "difficulty"]
            },
            alert_threshold=0.15,
            priority_multiplier=1.1
        ),
        "monetization": CategoryConfig(
            name="In-App Purchases",
            keywords=["pay to win", "expensive", "microtransactions", "purchase", "ads"],
            subcategories={
                "pay_to_win": ["pay to win", "unfair advantage", "premium features"],
                "expensive_items": ["too expensive", "overpriced", "cost too much"],
                "forced_purchases": ["forced to buy", "can't progress without paying"]
            },
            alert_threshold=0.12,
            priority_multiplier=1.0
        )
    }
}

# ===== DATA MODELS =====
class ReviewRequest(BaseModel):
    """Request model for fetching app reviews"""
    url: str = Field(..., description="Play Store URL of the app")
    count: int = Field(default=100, ge=1, le=5000, description="Number of reviews to fetch (1-5000)")
    star_filters: Optional[List[int]] = Field(default=None, description="List of star ratings to filter by (1-5)")
    negative_only: bool = Field(default=False, description="Export only negative reviews (1-2 stars)")
    export_format: str = Field(default="csv", description="Export format: 'csv' or 'xlsx'")

class SearchRequest(BaseModel):
    """Request model for searching apps"""
    query: str = Field(..., min_length=1, description="Search query for finding apps")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of search results (1-50)")

class AnalysisRequest(BaseModel):
    """Request model for comprehensive review analysis"""
    url: str = Field(..., description="Play Store URL of the app")
    count: int = Field(default=100, ge=10, le=1000, description="Number of reviews for analysis (10-1000)")
    star_filters: Optional[List[int]] = Field(default=None, description="List of star ratings to filter by (1-5)")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_topics: bool = Field(default=True, description="Include topic modeling")
    include_classification: bool = Field(default=True, description="Include issue/feature classification")
    include_insights: bool = Field(default=True, description="Include actionable insights generation")

class DirectAnalysisRequest(BaseModel):
    """Request model for direct review analysis without URL"""
    reviews: List[Dict[str, Any]] = Field(..., description="List of review objects with 'content'/'review' and 'rating'")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_topics: bool = Field(default=True, description="Include topic modeling")
    include_classification: bool = Field(default=True, description="Include issue/feature classification")
    include_insights: bool = Field(default=True, description="Include actionable insights generation")
    app_type: Optional[str] = Field(default=None, description="App type for dynamic analysis")
    sentiment_filter: Optional[str] = Field(default=None, description="Filter by sentiment: 'positive', 'negative', 'neutral', or None for all")

class ReviewResponse(BaseModel):
    """Response model for individual reviews"""
    rating: int = Field(..., description="Star rating (1-5)")
    review: str = Field(..., description="Review text content")
    date: str = Field(..., description="Review date in ISO format")

class AppSearchResult(BaseModel):
    """Response model for app search results"""
    title: str = Field(..., description="App title")
    appId: str = Field(..., description="App package ID")
    developer: str = Field(..., description="Developer name")
    summary: str = Field(..., description="Brief app description")
    icon: str = Field(..., description="App icon URL")
    score: float = Field(..., description="App rating score")
    scoreText: str = Field(..., description="Formatted rating text")
    priceText: str = Field(..., description="Price information")
    free: bool = Field(..., description="Whether the app is free")
    url: str = Field(..., description="Play Store URL")

# ===== UTILITY FUNCTIONS =====
def extract_app_id_from_url(url: str) -> Optional[str]:
    """
    Extract app ID from Play Store URL
    
    Args:
        url: Play Store URL containing app ID
        
    Returns:
        App ID string or None if not found
        
    Example:
        >>> extract_app_id_from_url("https://play.google.com/store/apps/details?id=com.whatsapp")
        "com.whatsapp"
    """
    match = re.search(r"id=([a-zA-Z0-9_.]+)", url)
    return match.group(1) if match else None

def format_search_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format raw search results from google-play-scraper into standardized format
    
    Args:
        raw_results: Raw search results from google-play-scraper
        
    Returns:
        List of formatted app information dictionaries
    """
    formatted_results = []
    
    for app in raw_results:
        try:
            # Safely extract and format app data
            description = app.get("description", "")
            summary = (description[:150] + "...") if len(description) > 150 else description
            
            formatted_app = {
                "title": app.get("title", "Unknown App"),
                "appId": app.get("appId", ""),
                "developer": app.get("developer", "Unknown Developer"),
                "summary": summary,
                "icon": app.get("icon", ""),
                "score": float(app.get("score", 0)) if app.get("score") is not None else 0.0,
                "scoreText": f"{app.get('score', 0):.1f}" if app.get("score") else "N/A",
                "priceText": "Free" if app.get("free", True) else f"${app.get('price', 0)}",
                "free": app.get("free", True),
                "url": f"https://play.google.com/store/apps/details?id={app.get('appId', '')}"
            }
            formatted_results.append(formatted_app)
        except Exception as e:
            # Log error and skip malformed result
            print(f"Error formatting search result: {e}")
            continue
    
    return formatted_results

def filter_reviews_by_rating(reviews_list: List[Dict[str, Any]], star_filters: List[int]) -> List[Dict[str, Any]]:
    """
    Filter reviews by star ratings
    
    Args:
        reviews_list: List of review dictionaries
        star_filters: List of star ratings to include (1-5)
        
    Returns:
        Filtered list of reviews
    """
    if not star_filters:
        return reviews_list
    
    return [review for review in reviews_list if review["rating"] in star_filters]

def generate_csv_content(reviews_list: List[Dict[str, Any]], app_id: str) -> str:
    """
    Generate CSV content from reviews data
    
    Args:
        reviews_list: List of review dictionaries
        app_id: App identifier for the CSV filename
        
    Returns:
        CSV content as string
    """
    output = io.StringIO()
    
    # Define CSV headers
    fieldnames = ['app_id', 'rating', 'review', 'date', 'review_length', 'export_timestamp']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    
    # Write headers
    writer.writeheader()
    
    # Write review data
    export_timestamp = datetime.now().isoformat()
    for review in reviews_list:
        writer.writerow({
            'app_id': app_id,
            'rating': review['rating'],
            'review': review['review'].replace('\n', ' ').replace('\r', ' '),  # Clean newlines for CSV
            'date': review['date'],
            'review_length': len(review['review']),
            'export_timestamp': export_timestamp
        })
    
    csv_content = output.getvalue()
    output.close()
    return csv_content

async def run_analysis_async(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run comprehensive analysis in a separate thread to avoid blocking with proper error handling
    """
    try:
        # Basic validation
        if not reviews or not isinstance(reviews, list):
            raise ValueError("No valid reviews provided for analysis")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, analysis_engine.comprehensive_analysis, reviews)
        
        if result is None:
            raise ValueError("Analysis returned null result")
            
        return result
    except Exception as e:
        logger.error(f"Async analysis execution failed: {e}")
        raise ValueError(f"Analysis failed: {str(e)}") from e

def get_app_metadata(app_id: str) -> Dict[str, Any]:
    """
    Fetch app metadata including total reviews and rating distribution
    
    Args:
        app_id: Google Play app ID
        
    Returns:
        Dictionary containing app metadata or empty dict if failed
    """
    try:
        if not SCRAPER_AVAILABLE:
            return {}
            
        app_info = app(app_id, lang="en", country="us")
        
        # Extract relevant metadata
        metadata = {
            "title": app_info.get("title", ""),
            "developer": app_info.get("developer", ""),
            "total_ratings": app_info.get("ratings", 0),  # Total number of ratings
            "total_reviews": app_info.get("reviews", 0),  # Total number of written reviews
            "average_score": app_info.get("score", 0),    # Average rating score
            "histogram": app_info.get("histogram", {}),   # Star distribution
            "installs": app_info.get("installs", ""),
            "updated": app_info.get("updated", ""),
            "version": app_info.get("version", "")
        }
        
        return metadata
        
    except Exception as e:
        print(f"Error fetching app metadata: {e}")
        return {}

# ===== API ENDPOINTS =====
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Play Store Review Scraper API",
        "status": "healthy",
        "version": "2.0.0",
        "analysis_enabled": ANALYSIS_ENABLED,
        "endpoints": {
            "search": "/search - Search for apps",
            "reviews": "/reviews - Get app reviews",
            "reviews_csv": "/reviews/csv - Export reviews as CSV",
            "analyze": "/analyze - Comprehensive review analysis",
            "docs": "/docs - API documentation"
        }
    }

@app.post("/search", response_model=Dict[str, Any], tags=["Search"])
async def search_apps(payload: SearchRequest):
    """
    Search for apps in the Play Store
    
    Args:
        payload: Search request containing query and limit
        
    Returns:
        Dictionary containing search results and metadata
        
    Raises:
        HTTPException: If search fails or invalid parameters
    """
    try:
        # Validate search query
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        # Perform search using google-play-scraper
        raw_results = search(payload.query, lang="en", country="us", n_hits=payload.limit)
        
        # Format results
        formatted_results = format_search_results(raw_results)
        
        return {
            "results": formatted_results,
            "query": payload.query,
            "total_found": len(formatted_results),
            "search_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Search failed: {str(e)}"
        )

@app.post("/reviews", response_model=Dict[str, Any], tags=["Reviews"])
async def get_reviews(payload: ReviewRequest):
    """
    Fetch reviews for a specific app
    
    Args:
        payload: Review request containing URL, count, and optional star filters
        
    Returns:
        Dictionary containing app reviews and metadata
        
    Raises:
        HTTPException: If URL is invalid or review fetching fails
    """
    try:
        # Extract app ID from URL
        app_id = extract_app_id_from_url(payload.url)
        if not app_id:
            raise HTTPException(status_code=400, detail="Invalid Play Store URL")
        
        # Fetch reviews using google-play-scraper
        result, _ = reviews(
            app_id, 
            lang="en", 
            country="us", 
            sort=Sort.NEWEST, 
            count=payload.count
        )
        
        # Format reviews
        formatted_reviews = [
            {
                "rating": review["score"], 
                "review": review["content"] or "No review text", 
                "date": review["at"].isoformat()
            }
            for review in result
        ]
        
        # Apply star rating filters if provided
        if payload.star_filters:
            formatted_reviews = filter_reviews_by_rating(formatted_reviews, payload.star_filters)
        
        # Calculate rating distribution before filtering
        rating_distribution = {str(i): 0 for i in range(1, 6)}
        for review in result:  # Use original result for accurate distribution
            rating_str = str(review["score"])
            if rating_str in rating_distribution:
                rating_distribution[rating_str] += 1
        
        # Fetch app metadata for total stats
        app_metadata = get_app_metadata(app_id)
        
        return {
            "app_id": app_id,
            "reviews": formatted_reviews,
            "total_reviews": len(formatted_reviews),
            "total_fetched": len(result),  # Total before filtering
            "rating_distribution": rating_distribution,
            "app_metadata": app_metadata,  # Total app stats from Play Store
            "requested_count": payload.count,
            "applied_filters": payload.star_filters or [],
            "fetch_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch reviews: {str(e)}"
        )

@app.post("/reviews/csv", tags=["Export"])
async def export_reviews_csv(payload: ReviewRequest):
    """
    Export app reviews as CSV file
    
    Args:
        payload: Review request containing URL, count, and optional star filters
        
    Returns:
        CSV file as downloadable response
        
    Raises:
        HTTPException: If URL is invalid or review fetching fails
    """
    try:
        # Extract app ID from URL
        app_id = extract_app_id_from_url(payload.url)
        if not app_id:
            raise HTTPException(status_code=400, detail="Invalid Play Store URL")
        
        # Fetch reviews (reuse logic from get_reviews)
        result, _ = reviews(
            app_id, 
            lang="en", 
            country="us", 
            sort=Sort.NEWEST, 
            count=payload.count
        )
        
        # Format reviews
        formatted_reviews = [
            {
                "rating": review["score"], 
                "review": review["content"] or "No review text", 
                "date": review["at"].isoformat()
            }
            for review in result
        ]
        
        # Apply star rating filters if provided
        if payload.star_filters:
            formatted_reviews = filter_reviews_by_rating(formatted_reviews, payload.star_filters)
        
        # Generate CSV content
        csv_content = generate_csv_content(formatted_reviews, app_id)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{app_id}_reviews_{timestamp}.csv"
        
        # Return CSV as downloadable file
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to export reviews: {str(e)}"
        )

@app.post("/analyze", response_model=Dict[str, Any], tags=["Analysis"])
async def analyze_reviews(payload: AnalysisRequest):
    """
    Perform comprehensive analysis on app reviews including sentiment, topics, and insights
    
    Args:
        payload: Analysis request with URL, count, and analysis options
        
    Returns:
        Comprehensive analysis results including sentiment, topics, issues, and recommendations
        
    Raises:
        HTTPException: If analysis is not available or fails
    """
    if not ANALYSIS_ENABLED:
        raise HTTPException(
            status_code=503, 
            detail="Analysis engine not available. Please install analysis dependencies."
        )
    
    try:
        # Extract app ID from URL
        app_id = extract_app_id_from_url(payload.url)
        if not app_id:
            raise HTTPException(status_code=400, detail="Invalid Play Store URL")
        
        # Fetch reviews
        result, _ = reviews(
            app_id, 
            lang="en", 
            country="us", 
            sort=Sort.NEWEST, 
            count=payload.count
        )
        
        # Format reviews
        formatted_reviews = [
            {
                "rating": review["score"], 
                "review": review["content"] or "No review text", 
                "date": review["at"].isoformat()
            }
            for review in result
        ]
        
        # Apply star rating filters if provided
        if payload.star_filters:
            formatted_reviews = filter_reviews_by_rating(formatted_reviews, payload.star_filters)
        
        if len(formatted_reviews) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Not enough reviews for meaningful analysis (minimum 10 required)"
            )
        
        # Run comprehensive analysis asynchronously
        analysis_results = await run_analysis_async(formatted_reviews)
        
        # Add metadata
        analysis_results['meta'] = {
            **analysis_results.get('meta', {}),
            'app_id': app_id,
            'requested_count': payload.count,
            'applied_filters': payload.star_filters or [],
            'analysis_options': {
                'sentiment': payload.include_sentiment,
                'topics': payload.include_topics,
                'classification': payload.include_classification
            }
        }
        
        return analysis_results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze/direct", response_model=Dict[str, Any], tags=["Analysis"])
async def analyze_reviews_direct(payload: DirectAnalysisRequest):
    """
    Direct comprehensive analysis of provided reviews with enhanced categorization
    
    Args:
        payload: Direct analysis request with review list and analysis options
        
    Returns:
        Comprehensive analysis results with new categorization system:
        - UX Issues (UI, Navigation, Flow, Feedback)
        - Tech Issues (Crashes, Bugs, Freezes, Performance)
        - Critical Complaints (Fraud, Fake Profiles, Security, Money Issues)
        - Feature Requests (Dark Mode, Search, Notifications, etc.)
        - Actionable insights with critical alerts
        
    Raises:
        HTTPException: If analysis fails or invalid input
    """
    if not ANALYSIS_ENABLED:
        raise HTTPException(status_code=503, detail="Analysis engine not available. Please install required dependencies.")
    
    try:
        if not payload.reviews:
            raise HTTPException(status_code=400, detail="No reviews provided for analysis")
        
        # Validate and format reviews
        formatted_reviews = []
        for i, review in enumerate(payload.reviews):
            if not isinstance(review, dict):
                raise HTTPException(status_code=400, detail=f"Review {i} must be a dictionary")
            
            # Handle both 'content' and 'review' keys for backward compatibility
            content = review.get('content') or review.get('review', '')
            rating = review.get('rating', 3)
            
            if not content.strip():
                continue  # Skip empty reviews
            
            formatted_review = {
                "rating": int(rating),
                "review": content,
                "date": review.get('date', datetime.now().isoformat())
            }
            formatted_reviews.append(formatted_review)
        
        if not formatted_reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found after processing")
        
        if len(formatted_reviews) < 3:
            raise HTTPException(status_code=400, detail="Minimum 3 valid reviews required for analysis")
        
        # Run comprehensive analysis
        analysis_results = await run_analysis_async(formatted_reviews)
        
        # Build response based on requested components
        response = {
            "analysis_type": "direct",
            "analysis_timestamp": datetime.now().isoformat(),
            "total_reviews_analyzed": len(formatted_reviews),
            "categorization_system": {
                "ux_issues": ["UI Issues", "Navigation Issues", "Flow Issues", "Feedback Issues"],
                "tech_issues": ["Crashes", "Bugs", "Freezes", "Performance Errors"],
                "critical_complaints": ["Fraud/Scam", "Fake Profiles", "Security Issues", "Money Issues", "Data Loss"],
                "feature_requests": ["Dark Mode", "Search", "Notifications", "Customization", "etc."]
            }
        }
        
        if payload.include_sentiment:
            response["sentiment_analysis"] = analysis_results.get("sentiment_analysis", {})
        
        if payload.include_topics:
            response["topics"] = analysis_results.get("topics", {})
        
        if payload.include_classification:
            response["classification"] = analysis_results.get("classification", {})
            
        if payload.include_insights:
            response["insights"] = analysis_results.get("insights", {})
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Direct analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Direct analysis failed: {str(e)}"
        )

@app.get("/analyze/sample", tags=["Analysis"])
async def get_sample_analysis():
    """
    Get a sample analysis result for demo purposes
    """
    return {
        "sample": True,
        "message": "This is a sample analysis response. Use POST /analyze with real review data.",
        "structure": {
            "meta": "Analysis metadata and configuration",
            "sentiment_analysis": "Sentiment scores and distribution",
            "topics": "Extracted themes and topics",
            "classification": "Issues and feature requests categorization with new system",
            "insights": "Actionable recommendations and priority analysis",
            "statistics": "Statistical summary of reviews"
        },
        "new_categorization": {
            "ux_issues": "User experience problems (UI, Navigation, Flow, Feedback)",
            "tech_issues": "Technical problems (Crashes, Bugs, Freezes, Performance)",
            "critical_complaints": "Serious user complaints (Fraud, Fake profiles, Security, Money issues)",
            "feature_requests": "User requests for new features"
        },
        "example_insights": {
            "critical_alerts": ["2 fraud/scam reports detected", "High fake profile complaints"],
            "priority_ux_issues": ["Navigation issues", "UI problems"],
            "priority_tech_issues": ["App crashes", "Performance problems"],
            "top_feature_requests": ["Dark mode", "Better search"],
            "urgency_score": 75,
            "recommendations": [
                "Address critical security complaints immediately",
                "Fix navigation and UI issues",
                "Consider implementing dark mode"
            ]
        }
    }

@app.post("/configure_categories")
async def configure_categories(request: dict):
    """
    Configure analysis categories for specific app types or custom configurations.
    
    Body:
    {
        "app_type": "social_media" | "ecommerce" | "productivity" | "gaming" | "custom",
        "custom_categories": { ... } // Optional, for custom configurations
    }
    """
    try:
        app_type = request.get("app_type", "general")
        custom_categories = request.get("custom_categories", {})
        
        if app_type in APP_TYPE_CONFIGS:
            # Use predefined configuration for app type
            new_categories = {**analysis_engine.categories, **APP_TYPE_CONFIGS[app_type]}
            analysis_engine.update_categories(new_categories)
        elif app_type == "custom" and custom_categories:
            # Use custom configuration
            # Convert dict to CategoryConfig objects
            converted_categories = {}
            for cat_id, cat_data in custom_categories.items():
                converted_categories[cat_id] = CategoryConfig(
                    name=cat_data.get("name", cat_id),
                    keywords=cat_data.get("keywords", []),
                    subcategories=cat_data.get("subcategories", {}),
                    alert_threshold=cat_data.get("alert_threshold", 0.05),
                    priority_multiplier=cat_data.get("priority_multiplier", 1.0)
                )
            
            new_categories = {**analysis_engine.categories, **converted_categories}
            analysis_engine.update_categories(new_categories)
        
        return {
            "success": True,
            "message": f"Categories configured for {app_type}",
            "active_categories": list(analysis_engine.categories.keys())
        }
        
    except Exception as e:
        logger.error(f"Error configuring categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_app_types")
async def get_available_app_types():
    """Get list of available predefined app type configurations."""
    return {
        "app_types": list(APP_TYPE_CONFIGS.keys()),
        "descriptions": {
            "social_media": "Privacy, content moderation, social features",
            "ecommerce": "Payments, delivery, customer service",
            "productivity": "Data sync, collaboration, workflow",
            "gaming": "Gameplay, monetization, performance"
        },
        "current_categories": list(analysis_engine.categories.keys())
    }

@app.post("/analyze_with_config")
async def analyze_with_configuration(request: DirectAnalysisRequest):
    """
    Analyze reviews with dynamic category configuration.
    
    This endpoint allows you to specify an app type or custom categories
    and then perform analysis with those specific categorizations.
    """
    try:
        # Configure categories if specified
        if hasattr(request, 'app_type') and request.app_type:
            if request.app_type in APP_TYPE_CONFIGS:
                new_categories = {**analysis_engine.categories, **APP_TYPE_CONFIGS[request.app_type]}
                analysis_engine.update_categories(new_categories)
        
        # Perform analysis
        reviews_data = [{"review": review, "rating": 5} for review in request.reviews]
        results = analysis_engine.comprehensive_analysis(reviews_data)
        
        # Add configuration info to results
        results["configuration"] = {
            "categories_used": list(analysis_engine.categories.keys()),
            "app_type": getattr(request, 'app_type', 'general')
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in configured analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/advanced", response_model=Dict[str, Any], tags=["Analysis"])
async def analyze_reviews_advanced(payload: DirectAnalysisRequest):
    """
    Advanced AI Analysis with Business Intelligence
    
    Provides deeper insights including:
    - Business impact signals (revenue risk, churn indicators)
    - Emotional intensity analysis beyond sentiment
    - Competitive intelligence extraction
    - Actionable pain points with user solutions
    - Smart feature prioritization with urgency
    - Advanced aspect-based sentiment analysis
    """
    if not ANALYSIS_ENABLED:
        raise HTTPException(status_code=503, detail="Analysis engine not available.")
    
    try:
        if not payload.reviews or len(payload.reviews) == 0:
            raise HTTPException(status_code=400, detail="No reviews provided for advanced analysis")
        
        logger.info(f"Starting advanced AI analysis of {len(payload.reviews)} reviews")
        
        # Validate and format reviews
        formatted_reviews = []
        for i, review in enumerate(payload.reviews):
            if not isinstance(review, dict):
                raise HTTPException(status_code=400, detail=f"Review {i} must be a dictionary")
            
            content = review.get('content') or review.get('review', '')
            if not content.strip():
                continue
                
            formatted_review = {
                "review_id": review.get('review_id', f"review_{i}"),
                "text": content,
                "content": content,
                "rating": int(review.get('rating', 3)),
                "date": review.get('date', datetime.now().isoformat())
            }
            formatted_reviews.append(formatted_review)
        
        if not formatted_reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found after processing")
        
        # Run advanced analysis
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, analysis_engine.advanced_analysis, formatted_reviews)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        logger.info("Advanced AI analysis completed successfully")
        return {
            "success": True,
            "message": "Advanced AI analysis completed successfully",
            "data": result,
            "analysis_type": "advanced_ai",
            "total_reviews_analyzed": len(formatted_reviews),
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as he:
        logger.error(f"Advanced analysis HTTP error: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Advanced analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")

@app.post("/analyze/sentiment_separation", response_model=Dict[str, Any], tags=["Analysis"])
async def analyze_sentiment_separation(payload: DirectAnalysisRequest):
    """
    Separate Sentiment Analysis
    
    Separates reviews into positive and negative sentiment themes with detailed analysis:
    
    **Positive Themes:**
    - Design & Interface Praise
    - Functionality Appreciation  
    - Feature Appreciation
    - Ease of Use
    - Performance Excellence
    - Overall Satisfaction
    
    **Negative Themes:**
    - Usability Problems
    - Performance Problems
    - Feature Complaints
    - Functionality Problems
    - Design & Interface Issues
    - Overall Dissatisfaction
    
    Each theme includes:
    - Actual review texts with highlighted phrases
    - Satisfaction/Severity levels
    - Percentage breakdowns
    - Review count statistics
    """
    if not ANALYSIS_ENABLED:
        raise HTTPException(status_code=503, detail="Analysis engine not available.")
    
    try:
        if not payload.reviews or len(payload.reviews) == 0:
            raise HTTPException(status_code=400, detail="No reviews provided for sentiment analysis")
        
        logger.info(f"Starting sentiment separation analysis of {len(payload.reviews)} reviews")
        
        # Validate and format reviews
        formatted_reviews = []
        for i, review in enumerate(payload.reviews):
            if not isinstance(review, dict):
                raise HTTPException(status_code=400, detail=f"Review {i} must be a dictionary")
            
            content = review.get('content') or review.get('review', '')
            if not content.strip():
                continue
                
            formatted_review = {
                "review": content,
                "rating": int(review.get('rating', 3)),
                "date": review.get('date', datetime.now().isoformat())
            }
            formatted_reviews.append(formatted_review)
        
        if not formatted_reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found after processing")
        
        # Run sentiment separation analysis
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, analysis_engine.separate_sentiment_analysis, formatted_reviews)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        logger.info("Sentiment separation analysis completed successfully")
        return {
            "success": True,
            "message": "Sentiment separation analysis completed successfully",
            "data": result,
            "analysis_type": "sentiment_separation",
            "total_reviews_analyzed": len(formatted_reviews),
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as he:
        logger.error(f"Sentiment separation HTTP error: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Sentiment separation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sentiment separation failed: {str(e)}")

@app.post("/segment_reviews", response_model=Dict[str, Any], tags=["Analysis"])
async def segment_reviews_by_themes(payload: DirectAnalysisRequest):
    """
    Segment reviews by themes and show actual reviews under each category with triggering terms/phrases.
    
    Provides detailed segmentation of reviews into:
    - UX Issues (navigation, interface, usability, accessibility, flow)
    - Tech Issues (crashes, bugs, performance, compatibility, connectivity)  
    - Critical Negative (fraud/scam, security, trust issues, data loss, billing)
    - Positive Features (design, functionality, features, ease of use)
    - Feature Requests (missing features, improvements, customization, integrations)
    
    Each theme shows:
    - Actual review texts
    - Triggering terms/phrases that caused the categorization
    - Term frequency analysis
    - Sentiment breakdown
    
    Args:
        payload: Review list with optional sentiment filter ('positive', 'negative', 'neutral')
        
    Returns:
        Segmented reviews with themes, subcategories, triggering terms, and actual review texts
    """
    if not ANALYSIS_ENABLED:
        raise HTTPException(status_code=503, detail="Analysis engine not available.")
    
    try:
        if not payload.reviews:
            raise HTTPException(status_code=400, detail="No reviews provided for segmentation")
        
        # Validate and format reviews
        formatted_reviews = []
        for i, review in enumerate(payload.reviews):
            if not isinstance(review, dict):
                raise HTTPException(status_code=400, detail=f"Review {i} must be a dictionary")
            
            content = review.get('content') or review.get('review', '')
            rating = review.get('rating', 3)
            
            if not content.strip():
                continue
            
            formatted_review = {
                "rating": int(rating),
                "review": content,
                "date": review.get('date', datetime.now().isoformat())
            }
            formatted_reviews.append(formatted_review)
        
        if not formatted_reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found after processing")
        
        # Extract sentiment filter from request if provided
        sentiment_filter = getattr(payload, 'sentiment_filter', None)
        
        # Perform theme segmentation
        segmentation_results = analysis_engine.segment_reviews_by_themes(
            formatted_reviews, 
            sentiment_filter=sentiment_filter
        )
        
        # Add metadata
        segmentation_results.update({
            "endpoint": "segment_reviews",
            "total_input_reviews": len(formatted_reviews),
            "processing_timestamp": datetime.now().isoformat()
        })
        
        return segmentation_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Review segmentation error: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

# ===== APPLICATION STARTUP =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 