# Play Store Review Analysis Engine
# A modular, comprehensive review analysis system with configurable categories
# Supports sentiment analysis, topic modeling, issue classification, and insights generation

# Standard library imports
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import traceback

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Scientific computing imports with fallbacks
try:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn libraries available - Full analysis features enabled")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn/NumPy/Pandas not available - Limited analysis features")

# NLP library imports with fallbacks
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    logger.info("TextBlob available for sentiment analysis")
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available - Limited sentiment analysis")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    logger.info("VADER sentiment analyzer available")
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER not available - Limited sentiment analysis")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    logger.info("NLTK libraries available")
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available - Using fallback text processing")

# Optional transformer imports with fallback handling
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available - Advanced sentiment models enabled")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.info("Transformers library not available - Using traditional sentiment analysis")

@dataclass
class CategoryConfig:
    """Configuration for analysis categories"""
    name: str
    keywords: List[str]
    subcategories: Dict[str, List[str]] = field(default_factory=dict)
    severity_weights: Dict[str, float] = field(default_factory=lambda: {"high": 1.0, "medium": 0.7, "low": 0.4})
    alert_threshold: float = 0.05  # 5% by default
    priority_multiplier: float = 1.0

class ModelInitializer:
    """Handles initialization of various NLP models and dependencies"""
    
    @staticmethod
    def setup_nltk_dependencies():
        """Download required NLTK data with fallback"""
        if not NLTK_AVAILABLE:
            return False
            
        required_downloads = ['vader_lexicon', 'punkt', 'stopwords', 'wordnet']
        for item in required_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}') if item == 'punkt' else nltk.data.find(f'corpora/{item}')
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK {item}...")
                    nltk.download(item, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK {item}: {e}")
        return True
    
    @staticmethod
    def initialize_fallback_stopwords():
        """Initialize fallback stopwords if NLTK is not available"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
            'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }

class SentimentAnalyzer:
    """Handles all sentiment analysis functionality"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Sentiment keywords for enhanced analysis
        self.positive_keywords = [
            'excellent', 'amazing', 'great', 'love', 'perfect', 'awesome', 'fantastic', 'brilliant',
            'wonderful', 'outstanding', 'superb', 'incredible', 'good', 'nice', 'helpful', 'useful',
            'easy', 'simple', 'fast', 'quick', 'smooth', 'reliable', 'stable', 'clean', 'beautiful'
        ]
        
        self.negative_keywords = [
            'terrible', 'awful', 'horrible', 'hate', 'worst', 'bad', 'poor', 'useless', 'broken',
            'annoying', 'frustrating', 'disappointing', 'slow', 'laggy', 'buggy', 'crash', 'freeze',
            'confusing', 'difficult', 'hard', 'complicated', 'ugly', 'outdated', 'spam', 'ads'
        ]
        
        # Initialize transformer models if available
        self.roberta_tokenizer = None
        self.roberta_model = None
        self.roberta_pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_transformer_models()
    
    def _initialize_transformer_models(self):
        """Initialize transformer-based sentiment models."""
        try:
            logger.info("Loading Twitter-RoBERTa sentiment model...")
            
            # Initialize Twitter-RoBERTa model for sentiment analysis
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            self.roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline for easier inference
            self.roberta_pipeline = pipeline(
                "sentiment-analysis",
                model=self.roberta_model,
                tokenizer=self.roberta_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Twitter-RoBERTa model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load transformer models: {e}")
            self.roberta_tokenizer = None
            self.roberta_model = None
            self.roberta_pipeline = None
    
    def _preprocess_for_roberta(self, text: str) -> str:
        """Preprocess text for RoBERTa model following Twitter conventions"""
        # Replace user mentions and URLs as per model training
        text = re.sub(r'@\w+', '@user', text)
        text = re.sub(r'http\S+|www\S+|https\S+', 'http', text, flags=re.MULTILINE)
        return text
    
    def analyze_sentiment_roberta(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Twitter-RoBERTa model"""
        if not self.roberta_pipeline:
            return {"available": False, "error": "RoBERTa model not available"}
        
        try:
            # Preprocess text
            processed_text = self._preprocess_for_roberta(text)
            
            # Truncate if too long (RoBERTa has 512 token limit)
            if len(processed_text) > 400:  # Conservative limit
                processed_text = processed_text[:400]
            
            # Get prediction
            result = self.roberta_pipeline(processed_text)[0]
            
            # Map labels to standard format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            sentiment_label = label_mapping.get(result['label'], result['label'].lower())
            confidence = result['score']
            
            # Convert to polarity score (-1 to 1)
            if sentiment_label == 'positive':
                polarity = confidence
            elif sentiment_label == 'negative':
                polarity = -confidence
            else:  # neutral
                polarity = 0.0
            
            return {
                "available": True,
                "sentiment": sentiment_label,
                "confidence": confidence,
                "polarity": polarity,
                "model": "twitter-roberta-base-sentiment-latest"
            }
            
        except Exception as e:
            logger.error(f"RoBERTa sentiment analysis failed: {e}")
            return {"available": False, "error": str(e)}
    
    def analyze_sentiment_hybrid(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis using multiple approaches
        Combines VADER, TextBlob, and keyword-based analysis
        """
        # VADER analysis - good for social media text
        if self.vader_analyzer:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_sentiment = self._classify_vader_sentiment(vader_scores['compound'])
        else:
            vader_scores = {'compound': 0.0}
            vader_sentiment = "neutral"
        
        # TextBlob analysis - good for general text
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            textblob_sentiment = self._classify_textblob_sentiment(blob.sentiment.polarity)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
        else:
            textblob_sentiment = "neutral"
            textblob_polarity = 0.0
            textblob_subjectivity = 0.5
        
        # Keyword-based sentiment boost
        keyword_boost = self._apply_keyword_sentiment_boost(text)
        
        # Combine results using weighted approach
        final_sentiment, confidence = self._combine_sentiment_results(
            vader_sentiment, vader_scores['compound'],
            textblob_sentiment, textblob_polarity,
            keyword_boost
        )
        
        return {
            "sentiment": final_sentiment,
            "confidence": confidence,
            "vader": {
                "sentiment": vader_sentiment,
                "scores": vader_scores
            },
            "textblob": {
                "sentiment": textblob_sentiment,
                "polarity": textblob_polarity,
                "subjectivity": textblob_subjectivity
            },
            "keyword_boost": keyword_boost
        }
    
    def _classify_vader_sentiment(self, compound_score: float) -> str:
        """Classify VADER compound score into sentiment categories"""
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    def _classify_textblob_sentiment(self, polarity: float) -> str:
        """Classify TextBlob polarity into sentiment categories"""
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _apply_keyword_sentiment_boost(self, text: str) -> float:
        """Apply keyword-based sentiment boosting"""
        text_lower = text.lower()
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        # Calculate boost based on keyword ratio
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_keywords * 0.3  # Max boost of 0.3
    
    def _combine_sentiment_results(self, vader_sent: str, vader_score: float, 
                                 textblob_sent: str, textblob_score: float,
                                 keyword_boost: float) -> Tuple[str, float]:
        """Combine multiple sentiment analysis results"""
        
        # Weight the methods based on their strengths
        vader_weight = 0.5  # Good for social media
        textblob_weight = 0.3  # Good for general text
        keyword_weight = 0.2  # Boost for domain-specific terms
        
        # Calculate weighted score
        weighted_score = (
            vader_score * vader_weight +
            textblob_score * textblob_weight +
            keyword_boost * keyword_weight
        )
        
        # Determine final sentiment
        if weighted_score >= 0.1:
            final_sentiment = "positive"
        elif weighted_score <= -0.1:
            final_sentiment = "negative"
        else:
            final_sentiment = "neutral"
        
        # Calculate confidence based on agreement
        sentiments = [vader_sent, textblob_sent]
        agreement = sum(1 for s in sentiments if s == final_sentiment) / len(sentiments)
        
        # Boost confidence if keyword analysis agrees
        if (final_sentiment == "positive" and keyword_boost > 0) or \
           (final_sentiment == "negative" and keyword_boost < 0):
            agreement = min(1.0, agreement + 0.2)
        
        # Scale confidence by absolute score magnitude
        confidence = agreement * min(1.0, abs(weighted_score) * 2)
        
        return final_sentiment, confidence

class TextProcessor:
    """Handles text preprocessing and tokenization"""
    
    def __init__(self, stop_words: set):
        self.stop_words = stop_words
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text)
        
        # Remove special characters but keep apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization with stopword removal"""
        words = text.lower().split()
        return [word.strip(string.punctuation) for word in words 
                if word.strip(string.punctuation) not in self.stop_words and len(word) > 2]

# Generic category configurations that work for any app
DEFAULT_CATEGORIES = {
    "ux_issues": CategoryConfig(
        name="UX Issues",
        keywords=["ui", "interface", "design", "layout", "confusing", "hard to use", "difficult", "navigate", "menu"],
        subcategories={
            "ui_issues": ["interface", "design", "layout", "ugly", "bad design", "looks bad", "visual"],
            "navigation_issues": ["navigate", "menu", "find", "lost", "confusing navigation", "hard to find"],
            "flow_issues": ["confusing", "difficult", "complicated", "hard to use", "user flow", "process"],
            "feedback_issues": ["feedback", "response", "notification", "alert", "confirmation"]
        },
        alert_threshold=0.15,
        priority_multiplier=0.8
    ),
    
    "feature_requests": CategoryConfig(
        name="Feature Requests",
        keywords=["need", "want", "add", "feature", "would like", "should have", "missing", "wish", "hope"],
        subcategories={
            "dark_mode": ["dark mode", "dark theme", "night mode", "black theme"],
            "notifications": ["notification", "alert", "remind", "push", "notify"],
            "customization": ["customize", "personal", "settings", "preferences", "options"],
            "search": ["search", "find", "filter", "lookup", "query"],
            "sharing": ["share", "export", "send", "social", "facebook", "twitter"],
            "offline": ["offline", "internet", "connection", "wifi", "data"],
            "widgets": ["widget", "home screen", "launcher", "shortcut"],
            "integration": ["integrate", "connect", "sync", "link", "import"],
            "backup": ["backup", "restore", "save", "export data", "sync"]
        },
        alert_threshold=0.1,
        priority_multiplier=0.6
    ),
    
    "critical_complaints": CategoryConfig(
        name="Critical Complaints", 
        keywords=["scam", "fraud", "fake", "steal", "money", "charge", "billing", "refund", "security", "hack"],
        subcategories={
            "fraud_scam": ["scam", "fraud", "fraudulent", "cheat", "deceive", "steal money", "fake charges"],
            "fake_profiles": ["fake profile", "bot", "spam account", "artificial", "not real", "fake user"],
            "security_issues": ["hack", "security", "password", "breach", "stolen", "compromised", "unsafe"],
            "money_issues": ["charge", "billing", "refund", "money", "payment", "subscription", "unauthorized"],
            "data_loss": ["lost data", "deleted", "disappeared", "missing", "corrupted", "gone"],
            "malware_virus": ["virus", "malware", "trojan", "spyware", "malicious", "infected"]
        },
        alert_threshold=0.02,  # 2% threshold for critical issues
        priority_multiplier=2.0
    ),
    
    "tech_issues": CategoryConfig(
        name="Technical Issues",
        keywords=["crash", "bug", "error", "freeze", "slow", "lag", "glitch", "broken", "not working"],
        subcategories={
            "crashes": ["crash", "crashed", "crashing", "force close", "shuts down", "stops working"],
            "bugs": ["bug", "glitch", "error", "issue", "problem", "broken"],
            "freezes": ["freeze", "frozen", "hang", "stuck", "unresponsive", "not responding"],
            "performance_errors": ["slow", "lag", "sluggish", "performance", "speed", "loading"],
            "login_errors": ["login", "sign in", "authentication", "password", "account"],
            "sync_errors": ["sync", "synchronize", "update", "refresh", "connection"]
        },
        alert_threshold=0.1,
        priority_multiplier=1.2
    ),
    
    "battery_drain": CategoryConfig(
        name="Battery & Performance",
        keywords=["battery", "drain", "power", "cpu", "memory", "ram", "storage", "space"],
        subcategories={
            "battery_drain": ["battery", "drain", "power", "dies fast", "battery life"],
            "memory_issues": ["memory", "ram", "storage", "space", "full", "low storage"],
            "cpu_usage": ["cpu", "processor", "hot", "heating", "warm", "performance"]
        },
        alert_threshold=0.08,
        priority_multiplier=0.9
    ),
    
    "ads_spam": CategoryConfig(
        name="Ads & Spam",
        keywords=["ads", "advertisement", "spam", "pop up", "banner", "commercial", "promotion"],
        subcategories={
            "excessive_ads": ["too many ads", "ad spam", "advertisement", "pop up", "banner"],
            "intrusive_ads": ["intrusive", "annoying ads", "full screen", "can't close", "forced ads"],
            "spam_content": ["spam", "junk", "unwanted", "irrelevant", "promotional"]
        },
        alert_threshold=0.12,
        priority_multiplier=0.7
    )
}

class ReviewAnalysisEngine:
    """
    Comprehensive review analysis engine with multiple NLP techniques
    """
    
    def __init__(self, categories: Optional[Dict[str, CategoryConfig]] = None):
        """Initialize the analysis engine with configurable categories."""
        logger.info("Initializing analysis models...")
        
        # Set up categories
        self.categories = categories or DEFAULT_CATEGORIES
        
        # Initialize model components
        self.model_initializer = ModelInitializer()
        
        # Set up NLTK dependencies
        self.model_initializer.setup_nltk_dependencies()
        
        # Initialize stopwords with fallback
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = self.model_initializer.initialize_fallback_stopwords()
        else:
            self.stop_words = self.model_initializer.initialize_fallback_stopwords()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize text processor
        self.text_processor = TextProcessor(self.stop_words)
        
        # Initialize ML models
        self.vectorizer = None
        self.topic_model = None
        self._initialize_models()
        
        logger.info("Models initialized successfully")
    
    def _initialize_models(self):
        """Initialize ML models for topic modeling."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available - Topic modeling disabled")
            self.vectorizer = None
            self.topic_model = None
            return
            
        try:
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Initialize topic model
            self.topic_model = LatentDirichletAllocation(
                n_components=5,
                random_state=42,
                max_iter=10
            )
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to simple models
            try:
                self.vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                self.topic_model = None
            except:
                self.vectorizer = None
                self.topic_model = None
    
    def update_categories(self, new_categories: Dict[str, CategoryConfig]):
        """Update the analysis categories"""
        self.categories = new_categories
    
    def add_category(self, category_id: str, config: CategoryConfig):
        """Add a new category to the existing ones"""
        self.categories[category_id] = config
    
    # Delegate sentiment analysis to the sentiment analyzer
    def analyze_sentiment_hybrid(self, text: str) -> Dict[str, Any]:
        """Delegate to sentiment analyzer"""
        return self.sentiment_analyzer.analyze_sentiment_hybrid(text)
    
    def analyze_sentiment_roberta(self, text: str) -> Dict[str, Any]:
        """Delegate to sentiment analyzer"""
        return self.sentiment_analyzer.analyze_sentiment_roberta(text)
    
    # Delegate text processing to the text processor
    def preprocess_text(self, text: str) -> str:
        """Delegate to text processor"""
        return self.text_processor.preprocess_text(text)
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Delegate to text processor"""
        return self.text_processor.simple_tokenize(text)

    def extract_topics_tfidf(self, reviews: List[str], n_topics: int = 5) -> Dict[str, Any]:
        """
        Extract topics using TF-IDF and clustering
        """
        if len(reviews) < 3:
            return {'topics': [], 'error': 'Not enough reviews for topic modeling'}
        
        try:
            # Preprocess reviews
            processed_reviews = [self.preprocess_text(review) for review in reviews]
            processed_reviews = [r for r in processed_reviews if len(r.split()) > 3]
            
            if len(processed_reviews) < 3:
                return {'topics': [], 'error': 'Not enough valid reviews after preprocessing'}
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(processed_reviews)
            feature_names = vectorizer.get_feature_names_out()
            
            # Perform clustering
            n_clusters = min(n_topics, len(processed_reviews) // 2)
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Extract topics
            topics = []
            for i in range(n_clusters):
                # Get cluster center
                center = kmeans.cluster_centers_[i]
                
                # Get top terms for this cluster
                top_indices = center.argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                
                # Get reviews in this cluster
                cluster_reviews = [processed_reviews[j] for j, label in enumerate(cluster_labels) if label == i]
                
                topics.append({
                    'topic_id': i,
                    'keywords': top_terms[:8],
                    'size': len(cluster_reviews),
                    'representative_reviews': cluster_reviews[:3]
                })
            
            # Sort by size
            topics.sort(key=lambda x: x['size'], reverse=True)
            
            return {
                'topics': topics,
                'total_topics': len(topics),
                'method': 'TF-IDF + K-Means Clustering'
            }
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return {'topics': [], 'error': f'Topic modeling failed: {str(e)}'}

    def classify_issues_and_features(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify reviews into the new categorization system with enhanced critical complaint detection
        
        Args:
            reviews: List of review dictionaries with 'review'/'content' and 'rating' keys
            
        Returns:
            Dictionary containing classified issues by category with scores and alerts
        """
        try:
            # Initialize result containers for new categories
            ux_issues_found = defaultdict(list)
            feature_requests_found = defaultdict(list)
            critical_complaints_found = defaultdict(list)
            tech_issues_found = defaultdict(list)
            other_issues_found = defaultdict(list)
            
            total_reviews = len(reviews)
            critical_alert_threshold = max(1, int(total_reviews * 0.05))  # 5% threshold for critical alerts
            classified_reviews = []
            
            for review in reviews:
                # Handle both 'review' and 'content' keys for backward compatibility
                text = review.get('review', review.get('content', '')).lower()
                rating = review.get('rating', 3)
                
                if not text:
                    continue
                
                found_categories = []
                
                # Classify based on configured categories
                for category_id, category_config in self.categories.items():
                    # Handle feature requests with subcategories
                    if category_id == 'feature_requests':
                        # Check for feature request keywords first
                        has_feature_request = any(keyword in text for keyword in category_config.keywords)
                        
                        if has_feature_request:
                            # Now check subcategories for specific feature types
                            matched_subcategory = None
                            matched_keyword = None
                            
                            for subcategory, subcategory_keywords in category_config.subcategories.items():
                                for subkeyword in subcategory_keywords:
                                    if subkeyword in text:
                                        matched_subcategory = subcategory
                                        matched_keyword = subkeyword
                                        break
                                if matched_subcategory:
                                    break
                            
                            # Use subcategory if found, otherwise general feature request
                            if matched_subcategory:
                                priority_score = (6 - rating) * category_config.priority_multiplier
                                feature_requests_found[matched_subcategory].append({
                                    'review': review,
                                    'keyword': matched_keyword,
                                    'priority_score': priority_score,
                                    'text_snippet': text[:100] + '...' if len(text) > 100 else text
                                })
                                found_categories.append(f'feature_{matched_subcategory}')
                            else:
                                # General feature request
                                general_keyword = next((kw for kw in category_config.keywords if kw in text), 'feature request')
                                priority_score = (6 - rating) * category_config.priority_multiplier
                                feature_requests_found['general_requests'].append({
                                    'review': review,
                                    'keyword': general_keyword,
                                    'priority_score': priority_score,
                                    'text_snippet': text[:100] + '...' if len(text) > 100 else text
                                })
                                found_categories.append('feature_general_requests')
                    else:
                        # Handle other categories normally
                        keywords = category_config.keywords
                        
                        for keyword in keywords:
                            if keyword in text:
                                # Calculate a simple score based on rating (lower rating = higher severity)
                                severity_score = (6 - rating) * category_config.priority_multiplier
                                
                                if category_id.startswith('ux_'):
                                    ux_issues_found[category_id].append({
                                        'review': review,
                                        'keyword': keyword,
                                        'severity_score': severity_score,
                                        'text_snippet': text[:100] + '...' if len(text) > 100 else text
                                    })
                                elif category_id.startswith('critical_'):
                                    critical_complaints_found[category_id].append({
                                        'review': review,
                                        'keyword': keyword,
                                        'severity_score': severity_score,
                                        'text_snippet': text[:150] + '...' if len(text) > 150 else text,
                                        'alert_priority': 'CRITICAL'
                                    })
                                elif category_id.startswith('tech_'):
                                    tech_issues_found[category_id].append({
                                        'review': review,
                                        'keyword': keyword,
                                        'severity_score': severity_score,
                                        'text_snippet': text[:100] + '...' if len(text) > 100 else text
                                    })
                                else:
                                    other_issues_found[category_id].append({
                                        'review': review,
                                        'keyword': keyword,
                                        'severity_score': severity_score,
                                        'text_snippet': text[:100] + '...' if len(text) > 100 else text
                                    })
                                
                                found_categories.append(category_id)
                                break
                
                # Additional logic for general feature requests
                feature_request_indicators = [
                    'wish', 'want', 'need', 'should add', 'would be nice', 'please add',
                    'hope', 'request', 'suggest', 'could you', 'feature request'
                ]
                
                if any(indicator in text for indicator in feature_request_indicators):
                    if not any('feature_' in cat for cat in found_categories):
                        feature_requests_found['general_requests'].append({
                            'review': review,
                            'keyword': 'general request',
                            'priority_score': 3.0,
                            'priority': 'medium',
                            'text_snippet': text[:100] + '...' if len(text) > 100 else text
                        })
                        found_categories.append('feature_general_requests')
                
                classified_reviews.append({
                    **review,
                    'categories_found': found_categories,
                    'classification_summary': {
                        'has_ux_issues': any('ux_' in cat for cat in found_categories),
                        'has_feature_requests': any('feature_' in cat for cat in found_categories),
                        'has_critical_complaints': any('critical_' in cat for cat in found_categories),
                        'has_tech_issues': any('tech_' in cat for cat in found_categories),
                        'has_other_issues': any('other_' in cat for cat in found_categories)
                    }
                })
            
            # Calculate summary statistics for each category
            def calculate_category_summary(issues_dict, category_name):
                summary = {}
                for issue_type, instances in issues_dict.items():
                    if not instances:
                        continue
                        
                    if 'priority_score' in instances[0]:
                        scores = [inst['priority_score'] for inst in instances]
                        score_type = 'priority'
                        level_func = self._get_priority_level
                    else:
                        scores = [inst['severity_score'] for inst in instances]
                        score_type = 'severity'
                        level_func = self._get_severity_level
                    
                    summary[issue_type] = {
                        'count': len(instances),
                        'percentage': (len(instances) / total_reviews) * 100 if total_reviews > 0 else 0,
                        f'avg_{score_type}_score': np.mean(scores) if scores else 0,
                        f'max_{score_type}_score': max(scores) if scores else 0,
                        f'{score_type}_level': level_func(np.mean(scores) if scores else 0),
                        'examples': instances[:3],  # Keep top 3 examples
                        'category': category_name
                    }
                return summary
            
            # Generate summaries
            ux_summary = calculate_category_summary(ux_issues_found, 'UX Issues')
            feature_summary = calculate_category_summary(feature_requests_found, 'Feature Requests')
            critical_summary = calculate_category_summary(critical_complaints_found, 'Critical Complaints')
            tech_summary = calculate_category_summary(tech_issues_found, 'Tech Issues')
            other_summary = calculate_category_summary(other_issues_found, 'Other Issues')
            
            # Generate critical alerts for high-frequency critical complaints
            critical_alerts = []
            total_critical_count = sum(len(instances) for instances in critical_complaints_found.values())
            
            for complaint_type, instances in critical_complaints_found.items():
                count = len(instances)
                percentage = (count / total_reviews) * 100 if total_reviews > 0 else 0
                
                if count >= critical_alert_threshold:
                    alert_level = 'HIGH' if count >= critical_alert_threshold * 2 else 'MEDIUM'
                    critical_alerts.append({
                        'type': complaint_type,
                        'count': count,
                        'percentage': percentage,
                        'alert_level': alert_level,
                        'message': f"⚠️ CRITICAL ALERT: {count} reports of {complaint_type.replace('_', ' ')} ({percentage:.1f}% of reviews)",
                        'examples': [inst['text_snippet'] for inst in instances[:2]]
                    })
            
            # Add overall critical alert if too many critical complaints
            if total_critical_count > total_reviews * 0.1:  # More than 10% critical complaints
                critical_alerts.insert(0, {
                    'type': 'overall_critical',
                    'count': total_critical_count,
                    'percentage': (total_critical_count / total_reviews) * 100,
                    'alert_level': 'HIGH',
                    'message': f"🚨 HIGH CRITICAL COMPLAINT RATE: {total_critical_count} critical complaints ({(total_critical_count/total_reviews)*100:.1f}% of reviews)",
                    'recommendation': 'Immediate attention required - multiple severe user complaints detected'
                })
            
            return {
                'ux_issues': ux_summary,
                'feature_requests': feature_summary,
                'critical_complaints': critical_summary,
                'tech_issues': tech_summary,
                'other_issues': other_summary,
                'critical_alerts': critical_alerts,
                'classified_reviews': classified_reviews,
                'total_reviews_analyzed': total_reviews,
                'categories_summary': {
                    'ux_issues_count': len(ux_issues_found),
                    'feature_requests_count': len(feature_requests_found),
                    'critical_complaints_count': len(critical_complaints_found),
                    'tech_issues_count': len(tech_issues_found),
                    'other_issues_count': len(other_issues_found),
                    'total_critical_complaints': total_critical_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error in issue/feature classification: {e}")
            return {
                'ux_issues': {},
                'feature_requests': {},
                'critical_complaints': {},
                'tech_issues': {},
                'other_issues': {},
                'critical_alerts': [],
                'classified_reviews': [],
                'total_reviews_analyzed': len(reviews),
                'categories_summary': {},
                'error': str(e)
            }

    def _calculate_issue_severity_score(self, rating: int, severity: str, text: str, keywords: List[str]) -> float:
        """Calculate severity score for an issue (1-5, higher = more severe)"""
        base_severity = {'low': 1, 'medium': 3, 'high': 4, 'critical': 5}[severity]
        
        # Lower ratings indicate more severe issues
        rating_factor = (6 - rating) / 5  # Converts 1-5 rating to 1-0 severity factor
        
        # Count keyword occurrences
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        keyword_factor = min(1.5, 1 + (keyword_count - 1) * 0.1)  # Boost for multiple keywords
        
        return min(5.0, base_severity * rating_factor * keyword_factor)

    def _calculate_feature_priority_score(self, rating: int, priority: str, text: str, keywords: List[str]) -> float:
        """Calculate priority score for a feature request (1-5, higher = higher priority)"""
        base_priority = {'low': 2, 'medium': 3, 'high': 4}[priority]
        
        # Higher ratings from users requesting features indicate higher priority
        rating_factor = rating / 5
        
        # Count keyword occurrences
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        keyword_factor = min(1.5, 1 + (keyword_count - 1) * 0.1)
        
        return min(5.0, base_priority * rating_factor * keyword_factor)

    def generate_actionable_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable insights from the new categorized analysis system
        """
        insights = {
            'critical_alerts': [],
            'priority_ux_issues': [],
            'priority_tech_issues': [],
            'top_feature_requests': [],
            'critical_complaints_summary': {},
            'sentiment_trends': {},
            'recommendations': [],
            'key_themes': [],
            'critical_themes': [],
            'urgency_score': 0,
            'summary': {}
        }
        
        # Get total reviews and classification data
        classification_data = analysis_results.get('classification', {})
        total_reviews = classification_data.get('total_reviews_analyzed', 0)
        
        # Avoid division by zero
        if total_reviews == 0:
            return insights
        
        # Extract critical alerts directly from classification
        insights['critical_alerts'] = classification_data.get('critical_alerts', [])
        
        # Process Critical Complaints Summary
        critical_complaints = classification_data.get('critical_complaints', {})
        total_critical = sum(data.get('count', 0) for data in critical_complaints.values())
        insights['critical_complaints_summary'] = {
            'total_count': total_critical,
            'percentage_of_reviews': round((total_critical / total_reviews) * 100, 1) if total_reviews > 0 else 0,
            'breakdown': critical_complaints,
            'requires_immediate_attention': total_critical > total_reviews * 0.05  # More than 5%
        }
        
        # Priority UX Issues
        ux_issues = classification_data.get('ux_issues', {})
        ux_priorities = []
        for issue_type, data in ux_issues.items():
            if data.get('count', 0) > 0:
                severity_score = data.get('avg_severity_score', 3.0)
                frequency_score = data.get('percentage', 0)
                combined_score = (frequency_score * 0.4) + (severity_score * 0.6)
                
                ux_priorities.append({
                    'issue': issue_type.replace('_', ' ').title(),
                    'count': data.get('count', 0),
                    'percentage': round(frequency_score, 1),
                    'severity_score': round(severity_score, 1),
                    'combined_score': round(combined_score, 1),
                    'severity_level': data.get('severity_level', 'Medium'),
                    'category': 'UX Issues'
                })
        
        ux_priorities.sort(key=lambda x: x['combined_score'], reverse=True)
        insights['priority_ux_issues'] = ux_priorities[:5]
        
        # Priority Tech Issues
        tech_issues = classification_data.get('tech_issues', {})
        tech_priorities = []
        for issue_type, data in tech_issues.items():
            if data.get('count', 0) > 0:
                severity_score = data.get('avg_severity_score', 3.0)
                frequency_score = data.get('percentage', 0)
                combined_score = (frequency_score * 0.4) + (severity_score * 0.6)
                
                tech_priorities.append({
                    'issue': issue_type.replace('_', ' ').title(),
                    'count': data.get('count', 0),
                    'percentage': round(frequency_score, 1),
                    'severity_score': round(severity_score, 1),
                    'combined_score': round(combined_score, 1),
                    'severity_level': data.get('severity_level', 'Medium'),
                    'category': 'Tech Issues'
                })
        
        tech_priorities.sort(key=lambda x: x['combined_score'], reverse=True)
        insights['priority_tech_issues'] = tech_priorities[:5]
        
        # Top Feature Requests with actual reviews and highlighted phrases
        feature_requests = classification_data.get('feature_requests', {})
        feature_priorities = []
        for feature_type, data in feature_requests.items():
            if data.get('count', 0) > 0:
                priority_score = data.get('avg_priority_score', 3.0)
                frequency_score = data.get('percentage', 0)
                combined_score = (frequency_score * 0.6) + (priority_score * 0.4)
                
                # Extract actual reviews and highlighted phrases from examples
                reviews_with_phrases = []
                examples = data.get('examples', [])
                for example in examples[:5]:  # Get up to 5 examples
                    review = example.get('review', {})
                    keyword = example.get('keyword', '')
                    review_text = review.get('content', '') or review.get('text', '') or review.get('review', '')
                    
                    # Extract highlighted phrases around the keyword
                    highlighted_phrases = self._extract_highlighted_phrases(review_text, keyword, feature_type)
                    
                    reviews_with_phrases.append({
                        'review_id': review.get('review_id', ''),
                        'author': review.get('author', 'Anonymous'),
                        'rating': review.get('rating', 0),
                        'date': review.get('date', ''),
                        'full_text': review_text,
                        'highlighted_phrases': highlighted_phrases,
                        'matched_keyword': keyword
                    })
                
                feature_priorities.append({
                    'feature_type': feature_type.replace('_', ' ').title(),
                    'requests': data.get('count', 0),
                    'percentage': round(frequency_score, 1),
                    'priority_score': round(priority_score, 1),
                    'combined_score': round(combined_score, 1),
                    'priority_level': data.get('priority_level', 'Medium'),
                    'category': 'Feature Requests',
                    'actual_reviews': reviews_with_phrases
                })
        
        feature_priorities.sort(key=lambda x: x['combined_score'], reverse=True)
        insights['top_feature_requests'] = feature_priorities[:5]
        
        # Sentiment Analysis
        if 'sentiment_analysis' in analysis_results:
            sentiment_data = analysis_results['sentiment_analysis']
            insights['sentiment_trends'] = {
                'overall_sentiment': sentiment_data.get('overall_sentiment', 'neutral'),
                'sentiment_distribution': sentiment_data.get('sentiment_counts', {}),
                'average_confidence': sentiment_data.get('average_confidence', 0)
            }
        
        # Key Themes from Topics
        if 'topics' in analysis_results and analysis_results['topics'].get('topics'):
            for i, topic in enumerate(analysis_results['topics']['topics'][:3]):
                theme_keywords = topic['keywords'][:5]
                insights['key_themes'].append({
                    'theme_name': f"Theme {i+1}: {theme_keywords[0].title()}",
                    'keywords': theme_keywords,
                    'review_count': topic['size'],
                    'percentage': round((topic['size'] / total_reviews) * 100, 1),
                    'sample_review': topic['representative_reviews'][0] if topic['representative_reviews'] else ""
                })
        
        # Generate Recommendations
        insights['recommendations'] = self._generate_detailed_recommendations(analysis_results)
        
        # Calculate Urgency Score
        insights['urgency_score'] = self._calculate_urgency_score(analysis_results)
        
        # Generate Summary
        insights['summary'] = self._generate_summary(analysis_results, insights)
        
        # Generate Critical Negative Themes (most reliable data)
        insights['critical_themes'] = self._extract_critical_negative_themes(analysis_results)
        
        return insights

    def _get_severity_level(self, score: float) -> str:
        """Convert severity score to level"""
        if score >= 4.5:
            return 'Critical'
        elif score >= 3.5:
            return 'High'
        elif score >= 2.5:
            return 'Medium'
        else:
            return 'Low'

    def _get_priority_level(self, score: float) -> str:
        """Convert priority score to level"""
        if score >= 4.0:
            return 'High'
        elif score >= 3.0:
            return 'Medium'
        else:
            return 'Low'

    def _generate_detailed_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed actionable recommendations based on new categorization"""
        recommendations = []
        
        if 'classification' not in analysis_results:
            return recommendations
            
        classification = analysis_results['classification']
        
        # Critical Complaints - Highest Priority
        critical_complaints = classification.get('critical_complaints', {})
        for complaint_type, data in critical_complaints.items():
            count = data.get('count', 0)
            if count > 0:
                if complaint_type in ['fraud_scam', 'fake_profiles', 'security_issues']:
                    recommendations.append({
                        'category': 'CRITICAL Security Issue',
                        'action': f'Immediately investigate and address {complaint_type.replace("_", " ")} complaints',
                        'priority': 'Critical',
                        'impact': 'Very High',
                        'timeline': 'Immediate (1-3 days)',
                        'affected_users': f"{count} reports ({data.get('percentage', 0):.1f}%)",
                        'severity': 'critical'
                    })
                elif complaint_type == 'money_issues':
                    recommendations.append({
                        'category': 'CRITICAL Billing Issue',
                        'action': 'Review billing system and process refund requests',
                        'priority': 'Critical',
                        'impact': 'High',
                        'timeline': 'Immediate (1-2 days)',
                        'affected_users': f"{count} reports ({data.get('percentage', 0):.1f}%)",
                        'severity': 'critical'
                    })
        
        # Tech Issues - High Priority
        tech_issues = classification.get('tech_issues', {})
        for issue_type, data in tech_issues.items():
            count = data.get('count', 0)
            if count > 0:
                if issue_type == 'crashes' and count >= 2:
                    recommendations.append({
                        'category': 'Critical Bug Fix',
                        'action': 'Implement crash reporting and fix stability issues',
                        'priority': 'High',
                        'impact': 'High',
                        'timeline': 'Short term (1-2 weeks)',
                        'affected_users': f"{count} reports ({data.get('percentage', 0):.1f}%)",
                        'severity': 'high'
                    })
                elif issue_type == 'performance_errors' and count >= 3:
                    recommendations.append({
                        'category': 'Performance Optimization',
                        'action': 'Optimize app loading times and reduce lag',
                        'priority': 'Medium-High',
                        'impact': 'Medium-High',
                        'timeline': 'Short term (2-4 weeks)',
                        'affected_users': f"{count} reports ({data.get('percentage', 0):.1f}%)",
                        'severity': 'medium'
                    })
        
        # UX Issues - Medium Priority
        ux_issues = classification.get('ux_issues', {})
        for issue_type, data in ux_issues.items():
            count = data.get('count', 0)
            if count >= 2:
                if issue_type == 'navigation_issues':
                    recommendations.append({
                        'category': 'UX Navigation',
                        'action': 'Redesign navigation flow and improve menu structure',
                        'priority': 'Medium',
                        'impact': 'Medium',
                        'timeline': 'Medium term (1-2 months)',
                        'affected_users': f"{count} reports ({data.get('percentage', 0):.1f}%)",
                        'severity': 'medium'
                    })
                elif issue_type == 'ui_issues':
                    recommendations.append({
                        'category': 'UI Design',
                        'action': 'Improve visual design and interface elements',
                        'priority': 'Medium',
                        'impact': 'Medium',
                        'timeline': 'Medium term (1-2 months)',
                        'affected_users': f"{count} reports ({data.get('percentage', 0):.1f}%)",
                        'severity': 'medium'
                    })
        
        # Feature Requests - Based on Priority
        feature_requests = classification.get('feature_requests', {})
        for feature_type, data in feature_requests.items():
            count = data.get('count', 0)
            if count >= 2:
                if feature_type == 'dark_mode':
                    recommendations.append({
                        'category': 'Feature Development',
                        'action': 'Implement dark mode theme option',
                        'priority': 'Medium',
                        'impact': 'Medium-High',
                        'timeline': 'Medium term (1-3 months)',
                        'affected_users': f"{count} requests ({data.get('percentage', 0):.1f}%)",
                        'severity': 'enhancement'
                    })
                elif feature_type == 'search':
                    recommendations.append({
                        'category': 'Feature Enhancement',
                        'action': 'Improve search functionality and add filters',
                        'priority': 'Medium',
                        'impact': 'Medium',
                        'timeline': 'Short term (2-6 weeks)',
                        'affected_users': f"{count} requests ({data.get('percentage', 0):.1f}%)",
                        'severity': 'enhancement'
                    })
        
        # Sort by priority and severity
        priority_order = {'Critical': 0, 'High': 1, 'Medium-High': 2, 'Medium': 3, 'Low': 4}
        
        def get_affected_count(rec):
            try:
                affected_str = rec.get('affected_users', '0')
                # Extract number from string like "2 reports (25.0%)"
                import re
                match = re.search(r'(\d+)', affected_str)
                return int(match.group(1)) if match else 0
            except:
                return 0
        
        recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 5), -get_affected_count(x)))
        
        return recommendations[:10]  # Return top 10 recommendations

    def _calculate_urgency_score(self, analysis_results: Dict[str, Any]) -> int:
        """Calculate overall urgency score (0-100) based on new categorization"""
        score = 0
        total_reviews = analysis_results.get('meta', {}).get('total_reviews', 0)
        
        if total_reviews == 0:
            return 0
        
        # Sentiment impact (0-30 points)
        if 'sentiment_analysis' in analysis_results:
            sentiment_data = analysis_results['sentiment_analysis']
            sentiment_counts = sentiment_data.get('sentiment_counts', {})
            negative_percentage = (sentiment_counts.get('negative', 0) / total_reviews) * 100
            score += min(30, negative_percentage)
        
        # Critical Complaints impact (0-40 points) - Highest weight
        if 'classification' in analysis_results:
            classification = analysis_results['classification']
            critical_complaints = classification.get('critical_complaints', {})
            
            critical_weight = 0
            for complaint_type, data in critical_complaints.items():
                count = data.get('count', 0)
                percentage = data.get('percentage', 0)
                
                # Higher weight for security-related complaints
                if complaint_type in ['fraud_scam', 'fake_profiles', 'security_issues']:
                    critical_weight += percentage * 1.5  # 1.5x multiplier
                elif complaint_type in ['money_issues', 'data_loss']:
                    critical_weight += percentage * 1.2  # 1.2x multiplier
                else:
                    critical_weight += percentage
            
            score += min(40, critical_weight)
        
        # Tech Issues impact (0-20 points)
        if 'classification' in analysis_results:
            classification = analysis_results['classification']
            tech_issues = classification.get('tech_issues', {})
            
            tech_weight = 0
            for issue_type, data in tech_issues.items():
                count = data.get('count', 0)
                percentage = data.get('percentage', 0)
                severity_score = data.get('avg_severity_score', 3.0)
                
                # Weight by severity
                if issue_type == 'crashes':
                    tech_weight += percentage * 1.5  # Crashes are most critical
                elif issue_type in ['bugs', 'freezes']:
                    tech_weight += percentage * 1.2
                else:
                    tech_weight += percentage
            
            score += min(20, tech_weight)
        
        # Review volume impact (0-10 points)
        if total_reviews > 100:
            score += 10
        elif total_reviews > 50:
            score += 8
        elif total_reviews > 20:
            score += 5
        else:
            score += 2
        
        return min(100, int(score))

    def _generate_summary(self, analysis_results: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary based on new categorization"""
        total_reviews = analysis_results.get('meta', {}).get('total_reviews', 0)
        
        summary = {
            'total_reviews_analyzed': total_reviews,
            'overall_health': 'Good',  # Will be determined below
            'main_concern': 'None identified',
            'top_opportunity': 'None identified',
            'action_required': False,
            'critical_alerts_count': len(insights.get('critical_alerts', []))
        }
        
        # Determine overall health based on urgency and critical complaints
        urgency = insights.get('urgency_score', 0)
        critical_summary = insights.get('critical_complaints_summary', {})
        critical_percentage = critical_summary.get('percentage_of_reviews', 0)
        
        if urgency >= 70 or critical_percentage > 10:
            summary['overall_health'] = 'Critical'
            summary['action_required'] = True
        elif urgency >= 50 or critical_percentage > 5:
            summary['overall_health'] = 'Poor'
            summary['action_required'] = True
        elif urgency >= 30 or critical_percentage > 2:
            summary['overall_health'] = 'Fair'
            summary['action_required'] = True
        elif urgency >= 15:
            summary['overall_health'] = 'Good'
        else:
            summary['overall_health'] = 'Excellent'
        
        # Identify main concern - prioritize critical complaints
        critical_alerts = insights.get('critical_alerts', [])
        if critical_alerts:
            summary['main_concern'] = critical_alerts[0].get('message', 'Critical security/trust issues detected')
        elif insights.get('priority_tech_issues'):
            top_tech_issue = insights['priority_tech_issues'][0]
            summary['main_concern'] = f"{top_tech_issue['issue']} ({top_tech_issue['count']} reports)"
        elif insights.get('priority_ux_issues'):
            top_ux_issue = insights['priority_ux_issues'][0]
            summary['main_concern'] = f"{top_ux_issue['issue']} ({top_ux_issue['count']} reports)"
        
        # Identify top opportunity
        if insights.get('top_feature_requests'):
            top_feature = insights['top_feature_requests'][0]
            summary['top_opportunity'] = f"{top_feature['feature_type']} ({top_feature['requests']} requests)"
        
        return summary

    def _extract_critical_negative_themes(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract critical themes from negative reviews only (most reliable data)
        Focuses on 1-3 star reviews to avoid paid review contamination
        """
        critical_themes = []
        
        # Get reviews with sentiment analysis
        if 'sentiment_analysis' not in analysis_results:
            return critical_themes
            
        sentiment_data = analysis_results['sentiment_analysis']
        reviews_with_sentiment = sentiment_data.get('reviews_with_sentiment', [])
        
        # Filter to negative reviews only (1-3 star ratings)
        negative_reviews = [
            review for review in reviews_with_sentiment 
            if review.get('rating', 5) <= 3
        ]
        
        if len(negative_reviews) == 0:
            return critical_themes
        
        # Define critical problem themes with specific phrases
        problem_themes = {
            'Crashes & Bugs': {
                'keywords': ['crash', 'bug', 'freeze', 'hang', 'broken', 'error', 'glitch', 'stops working'],
                'problematic_phrases': [
                    'app crashes', 'keeps crashing', 'always crashes', 'crashes constantly',
                    'frozen screen', 'app freezes', 'stops responding', 'not working',
                    'broken feature', 'major bug', 'full of bugs', 'buggy app'
                ]
            },
            'Performance Issues': {
                'keywords': ['slow', 'lag', 'loading', 'timeout', 'delayed', 'stuck', 'wait'],
                'problematic_phrases': [
                    'very slow', 'too slow', 'slow loading', 'takes forever',
                    'lag issues', 'laggy app', 'performance issues', 'slow response',
                    'stuck loading', 'timeout error', 'loading problems'
                ]
            },
            'Login & Account Problems': {
                'keywords': ['login', 'password', 'account', 'authentication', 'sign in', 'verification'],
                'problematic_phrases': [
                    'cannot login', 'login failed', 'password not working', 'login issues',
                    'account locked', 'verification problems', 'sign in error',
                    'authentication failed', 'login not working'
                ]
            },
            'Payment & Billing Issues': {
                'keywords': ['payment', 'billing', 'charge', 'money', 'refund', 'subscription', 'premium'],
                'problematic_phrases': [
                    'billing problems', 'wrong charge', 'payment failed', 'money issues',
                    'refund request', 'unwanted charges', 'billing error', 'payment not working',
                    'subscription issues', 'premium not working'
                ]
            },
            'User Interface Problems': {
                'keywords': ['confusing', 'difficult', 'hard to use', 'navigation', 'interface', 'design'],
                'problematic_phrases': [
                    'confusing interface', 'hard to navigate', 'poor design', 'bad UI',
                    'difficult to use', 'navigation problems', 'interface issues',
                    'unclear layout', 'design problems'
                ]
            },
            'Fake Profiles & Scams': {
                'keywords': ['fake', 'scam', 'fraud', 'catfish', 'bot', 'spam'],
                'problematic_phrases': [
                    'fake profiles', 'too many fakes', 'scam app', 'fraudulent',
                    'fake users', 'catfish profiles', 'spam messages', 'bot accounts',
                    'fake photos', 'scammer alert'
                ]
            },
            'Poor Customer Support': {
                'keywords': ['support', 'help', 'customer service', 'response', 'contact'],
                'problematic_phrases': [
                    'no response', 'poor support', 'customer service issues', 'no help',
                    'support not helpful', 'cannot contact', 'no customer service',
                    'support problems', 'unresponsive support'
                ]
            }
        }
        
        # Analyze each theme
        for theme_name, theme_data in problem_themes.items():
            matching_reviews = []
            found_phrases = []
            
            for review in negative_reviews:
                review_text = review.get('review', '').lower()
                if not review_text:
                    review_text = review.get('content', '').lower()
                
                # Check for keyword matches
                keyword_matches = [kw for kw in theme_data['keywords'] if kw in review_text]
                
                # Check for specific problematic phrases
                phrase_matches = [phrase for phrase in theme_data['problematic_phrases'] if phrase in review_text]
                
                if keyword_matches or phrase_matches:
                    matching_reviews.append(review)
                    found_phrases.extend(phrase_matches)
                    found_phrases.extend(keyword_matches)
            
            # Only include themes with significant negative mentions
            if len(matching_reviews) >= 2:  # At least 2 negative reviews mentioning the issue
                # Determine severity based on frequency and rating severity
                avg_rating = sum(r.get('rating', 1) for r in matching_reviews) / len(matching_reviews)
                frequency_percentage = (len(matching_reviews) / len(negative_reviews)) * 100
                
                # Severity calculation
                if avg_rating <= 1.5 and frequency_percentage >= 20:
                    severity = 'Critical'
                elif avg_rating <= 2.0 and frequency_percentage >= 15:
                    severity = 'High'
                elif frequency_percentage >= 10:
                    severity = 'Medium'
                else:
                    severity = 'Low'
                
                # Get unique phrases and keywords
                unique_phrases = list(set(found_phrases))[:10]  # Top 10 unique phrases
                
                critical_themes.append({
                    'theme_name': theme_name,
                    'keywords': theme_data['keywords'][:8],  # Top 8 keywords
                    'review_count': len(matching_reviews),
                    'percentage': round(frequency_percentage, 1),
                    'severity': severity,
                    'problematic_phrases': unique_phrases
                })
        
        # Sort by severity and frequency
        severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        critical_themes.sort(key=lambda x: (severity_order.get(x['severity'], 4), -x['review_count']))
        
        return critical_themes[:6]  # Return top 6 critical themes

    def _extract_highlighted_phrases(self, text: str, matched_keyword: str, feature_type: str) -> List[str]:
        """Extract highlighted phrases around keywords for feature requests"""
        phrases = []
        text_lower = text.lower()
        
        # Get feature-specific keywords
        feature_keywords = []
        if feature_type in self.categories.get('feature_requests', {}).subcategories:
            feature_keywords = self.categories['feature_requests'].subcategories[feature_type]
        
        # Add general feature request keywords
        general_keywords = ['need', 'want', 'add', 'feature', 'would like', 'should have', 'missing', 'wish', 'hope']
        all_keywords = feature_keywords + general_keywords + [matched_keyword]
        
        # Find phrases containing keywords
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword.lower() in sentence.lower() for keyword in all_keywords):
                # Clean up the sentence
                clean_sentence = sentence.strip(' ,!?.-')
                if len(clean_sentence) > 10:  # Only include meaningful phrases
                    phrases.append(clean_sentence)
        
        # If no sentences found, try to extract phrases around keywords
        if not phrases:
            for keyword in all_keywords:
                if keyword.lower() in text_lower:
                    start = max(0, text_lower.find(keyword.lower()) - 50)
                    end = min(len(text), text_lower.find(keyword.lower()) + len(keyword) + 50)
                    phrase = text[start:end].strip()
                    if phrase:
                        phrases.append(phrase)
        
        return phrases[:3]  # Return top 3 phrases

    def comprehensive_analysis(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on reviews
        """
        if not reviews:
            return {'error': 'No reviews provided for analysis'}
        
        logger.info(f"Starting comprehensive analysis of {len(reviews)} reviews")
        
        # Extract review texts
        review_texts = [review.get('review', '') for review in reviews]
        valid_reviews = [text for text in review_texts if text and len(text.strip()) > 10]
        
        if len(valid_reviews) < 3:
            return {'error': 'Not enough valid reviews for analysis (minimum 3 required)'}
        
        results = {
            'meta': {
                'total_reviews': len(reviews),
                'valid_reviews': len(valid_reviews),
                'analysis_timestamp': datetime.now().isoformat(),
                'engine_version': '2.0.0'
            }
        }
        
        try:
            # 1. Sentiment Analysis
            logger.info("Performing sentiment analysis...")
            sentiment_results = []
            for review in reviews:
                sentiment = self.analyze_sentiment_hybrid(review.get('review', ''))
                sentiment_results.append({
                    **review,
                    'sentiment_analysis': sentiment
                })
            
            # Aggregate sentiment statistics
            sentiment_counts = Counter([r['sentiment_analysis']['sentiment'] for r in sentiment_results])
            avg_confidence = np.mean([r['sentiment_analysis']['confidence'] for r in sentiment_results])
            
            results['sentiment_analysis'] = {
                'reviews_with_sentiment': sentiment_results,
                'sentiment_counts': dict(sentiment_counts),
                'overall_sentiment': max(sentiment_counts, key=sentiment_counts.get),
                'average_confidence': round(avg_confidence, 3)
            }
            
            # 2. Topic Modeling
            logger.info("Performing topic modeling...")
            topic_results = self.extract_topics_tfidf(valid_reviews, n_topics=5)
            results['topics'] = topic_results
            
            # 3. Issue and Feature Classification
            logger.info("Classifying issues and features...")
            classification_results = self.classify_issues_and_features(reviews)
            results['classification'] = classification_results
            
            # 4. Generate Actionable Insights
            logger.info("Generating actionable insights...")
            insights = self.generate_actionable_insights(results)
            results['insights'] = insights
            
            # 5. Statistical Summary
            results['statistics'] = {
                'avg_rating': round(np.mean([r.get('rating', 0) for r in reviews]), 2),
                'rating_distribution': dict(Counter([r.get('rating', 0) for r in reviews])),
                'avg_review_length': round(np.mean([len(r.get('review', '')) for r in reviews]), 1),
                'total_words': sum(len(r.get('review', '').split()) for r in reviews),
                'avg_words_per_review': round(np.mean([len(r.get('review', '').split()) for r in reviews]), 1)
            }
            
            logger.info("Comprehensive analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}

    def segment_reviews_by_themes(self, reviews: List[Dict[str, Any]], sentiment_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Segment reviews by themes and show actual reviews under each category with triggering terms/phrases.
        
        Args:
            reviews: List of review dictionaries
            sentiment_filter: Optional filter ('positive', 'negative', 'neutral', None for all)
            
        Returns:
            Dictionary with reviews segmented by themes including triggering terms
        """
        try:
            # Enhanced theme definitions with more specific terms
            theme_definitions = {
                'ux_issues': {
                    'name': 'User Experience Issues',
                    'terms': {
                        'navigation': ['navigation', 'navigate', 'menu', 'find', 'lost', 'confusing layout', 'hard to find'],
                        'interface': ['ui', 'interface', 'design', 'layout', 'buttons', 'screen', 'display'],
                        'usability': ['usability', 'user friendly', 'easy to use', 'difficult', 'complicated', 'intuitive'],
                        'accessibility': ['accessibility', 'font size', 'contrast', 'readable', 'hard to see', 'small text'],
                        'flow': ['workflow', 'process', 'steps', 'flow', 'sequence', 'order']
                    }
                },
                'tech_issues': {
                    'name': 'Technical Issues', 
                    'terms': {
                        'crashes': ['crash', 'crashes', 'crashed', 'force close', 'shuts down', 'stops working'],
                        'bugs': ['bug', 'bugs', 'glitch', 'error', 'broken', 'not working'],
                        'performance': ['slow', 'lag', 'laggy', 'freeze', 'frozen', 'loading', 'performance'],
                        'compatibility': ['compatibility', 'device', 'version', 'android', 'ios', 'phone'],
                        'connectivity': ['connection', 'network', 'internet', 'wifi', 'offline', 'sync']
                    }
                },
                'critical_negative': {
                    'name': 'Critical Negative Complaints',
                    'terms': {
                        'fraud_scam': ['fraud', 'scam', 'fake', 'cheat', 'steal', 'money stolen', 'unauthorized'],
                        'security': ['security', 'privacy', 'data breach', 'hacked', 'unsafe', 'personal info'],
                        'trust_issues': ['trust', 'reliable', 'dishonest', 'misleading', 'false advertising'],
                        'data_loss': ['lost data', 'deleted', 'missing', 'gone', 'disappeared', 'backup failed'],
                        'billing': ['charged', 'billing', 'refund', 'money', 'payment', 'subscription']
                    }
                },
                'positive_features': {
                    'name': 'Praised Features',
                    'terms': {
                        'design': ['beautiful', 'clean design', 'attractive', 'nice interface', 'well designed'],
                        'functionality': ['works well', 'smooth', 'fast', 'efficient', 'reliable', 'stable'],
                        'features': ['love this feature', 'amazing', 'awesome', 'perfect', 'exactly what i needed'],
                        'ease_of_use': ['easy to use', 'simple', 'intuitive', 'user friendly', 'straightforward']
                    }
                },
                'feature_requests': {
                    'name': 'Feature Requests',
                    'terms': {
                        'missing_features': ['wish', 'need', 'would like', 'please add', 'missing', 'lack'],
                        'improvements': ['improve', 'better', 'enhance', 'upgrade', 'update'],
                        'customization': ['customize', 'personalize', 'settings', 'options', 'configure'],
                        'integrations': ['integrate', 'sync with', 'connect to', 'import', 'export']
                    }
                }
            }
            
            # Initialize results structure
            segmented_reviews = {}
            
            for theme_id, theme_info in theme_definitions.items():
                segmented_reviews[theme_id] = {
                    'theme_name': theme_info['name'],
                    'subcategories': {},
                    'total_reviews': 0,
                    'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0}
                }
                
                for subcat_id, terms in theme_info['terms'].items():
                    segmented_reviews[theme_id]['subcategories'][subcat_id] = {
                        'name': subcat_id.replace('_', ' ').title(),
                        'triggering_terms': [],
                        'reviews': [],
                        'term_frequency': {}
                    }
            
            # Process each review
            for review in reviews:
                text = review.get('review', review.get('content', '')).lower()
                rating = review.get('rating', 3)
                
                if not text:
                    continue
                
                # Determine sentiment for filtering
                review_sentiment = 'positive' if rating >= 4 else 'negative' if rating <= 2 else 'neutral'
                
                # Apply sentiment filter if specified
                if sentiment_filter and review_sentiment != sentiment_filter:
                    continue
                
                # Check each theme
                for theme_id, theme_info in theme_definitions.items():
                    theme_matched = False
                    
                    for subcat_id, terms in theme_info['terms'].items():
                        subcat_matched = False
                        matched_terms = []
                        
                        for term in terms:
                            if term in text:
                                matched_terms.append(term)
                                subcat_matched = True
                                theme_matched = True
                                
                                # Update term frequency
                                if term not in segmented_reviews[theme_id]['subcategories'][subcat_id]['term_frequency']:
                                    segmented_reviews[theme_id]['subcategories'][subcat_id]['term_frequency'][term] = 0
                                segmented_reviews[theme_id]['subcategories'][subcat_id]['term_frequency'][term] += 1
                        
                        if subcat_matched:
                            # Add review to subcategory
                            review_entry = {
                                **review,
                                'matched_terms': matched_terms,
                                'sentiment': review_sentiment,
                                'text_preview': (text[:200] + '...') if len(text) > 200 else text
                            }
                            
                            segmented_reviews[theme_id]['subcategories'][subcat_id]['reviews'].append(review_entry)
                            segmented_reviews[theme_id]['subcategories'][subcat_id]['triggering_terms'].extend(matched_terms)
                    
                    if theme_matched:
                        segmented_reviews[theme_id]['total_reviews'] += 1
                        segmented_reviews[theme_id]['sentiment_breakdown'][review_sentiment] += 1
            
            # Clean up and summarize data
            for theme_id in segmented_reviews:
                for subcat_id in segmented_reviews[theme_id]['subcategories']:
                    subcat = segmented_reviews[theme_id]['subcategories'][subcat_id]
                    
                    # Remove duplicate triggering terms and sort by frequency
                    unique_terms = list(set(subcat['triggering_terms']))
                    subcat['triggering_terms'] = unique_terms
                    subcat['review_count'] = len(subcat['reviews'])
                    
                    # Sort reviews by rating (highest first for positive themes, lowest for negative)
                    if theme_id in ['positive_features', 'feature_requests']:
                        subcat['reviews'].sort(key=lambda x: x.get('rating', 0), reverse=True)
                    else:
                        subcat['reviews'].sort(key=lambda x: x.get('rating', 5))
                    
                    # Keep only top 10 reviews per subcategory for performance
                    subcat['reviews'] = subcat['reviews'][:10]
                    
                    # Get top triggering terms
                    top_terms = sorted(subcat['term_frequency'].items(), key=lambda x: x[1], reverse=True)[:5]
                    subcat['top_triggering_terms'] = [{'term': term, 'frequency': freq} for term, freq in top_terms]
            
            # Calculate summary statistics
            total_segmented = sum(theme['total_reviews'] for theme in segmented_reviews.values())
            total_input = len(reviews)
            
            summary = {
                'total_reviews_analyzed': total_input,
                'total_reviews_segmented': total_segmented,
                'segmentation_coverage': round((total_segmented / total_input) * 100, 1) if total_input > 0 else 0,
                'sentiment_filter_applied': sentiment_filter,
                'themes_found': len([t for t in segmented_reviews.values() if t['total_reviews'] > 0]),
                'top_themes': sorted(
                    [(theme_id, theme['total_reviews']) for theme_id, theme in segmented_reviews.items()],
                    key=lambda x: x[1], reverse=True
                )[:3]
            }
            
            return {
                'segmented_reviews': segmented_reviews,
                'summary': summary,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Review segmentation failed: {e}")
            return {'error': f'Review segmentation failed: {str(e)}'}

    def advanced_analysis(self, reviews: List[Dict]) -> Dict[str, Any]:
        """
        Advanced AI analysis with deeper business intelligence
        
        This provides:
        - Business impact signals (revenue risk, churn indicators)
        - Emotional intensity analysis beyond sentiment
        - Competitive intelligence extraction
        - Actionable pain points with user solutions
        - Smart feature prioritization with urgency
        - Advanced aspect-based sentiment analysis
        """
        try:
            logger.info("Starting advanced AI analysis...")
            
            if not reviews:
                return {'error': 'No reviews provided for advanced analysis'}
            
            # Initialize results structure
            advanced_results = {
                'meta': {
                    'analysis_type': 'advanced_ai',
                    'total_reviews': len(reviews),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '2.0-advanced'
                },
                'business_intelligence': {},
                'emotional_analysis': {},
                'competitive_insights': {},
                'actionable_feedback': {},
                'advanced_sentiment': {},
                'recommendations': []
            }
            
            # Extract text from all reviews
            review_texts = []
            for review in reviews:
                text = review.get('content', '') or review.get('text', '') or review.get('review', '')
                if text and isinstance(text, str):
                    review_texts.append({
                        'text': text,
                        'rating': int(review.get('rating', 0)) if isinstance(review.get('rating'), (int, str)) else 0,
                        'date': review.get('date', ''),
                        'review_id': review.get('review_id', f"review_{len(review_texts)}")
                    })
            
            # 1. Business Impact Analysis
            advanced_results['business_intelligence'] = self._analyze_business_impact(review_texts)
            
            # 2. Emotional Intensity Analysis
            advanced_results['emotional_analysis'] = self._analyze_emotional_intensity(review_texts)
            
            # 3. Competitive Intelligence
            advanced_results['competitive_insights'] = self._extract_competitive_intelligence(review_texts)
            
            # 4. Actionable Feedback Analysis
            advanced_results['actionable_feedback'] = self._analyze_actionable_feedback(review_texts)
            
            # 5. Advanced Aspect-Based Sentiment
            advanced_results['advanced_sentiment'] = self._advanced_aspect_sentiment(review_texts)
            
            # 6. Generate Advanced Recommendations
            advanced_results['recommendations'] = self._generate_advanced_recommendations(advanced_results)
            
            logger.info("Advanced AI analysis completed successfully")
            return advanced_results
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
            return {'error': f'Advanced analysis failed: {str(e)}'}
    
    def _analyze_business_impact(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze business-critical signals"""
        impact_signals = {
            'revenue_risk': {'high': 0, 'medium': 0, 'low': 0, 'signals': []},
            'churn_indicators': {'count': 0, 'percentage': 0, 'phrases': []},
            'advocacy_signals': {'count': 0, 'percentage': 0, 'phrases': []},
            'urgency_distribution': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        }
        
        # Revenue risk patterns
        revenue_risk_patterns = {
            'high': [r'(cancel|uninstall|delete|refund|money back|subscription.*cancel)',
                    r'(switching to|using.*instead|found better|competitor)'],
            'medium': [r'(disappointed|waste.*money|not worth|regret|mistake)',
                      r'(considering.*cancel|thinking.*leaving|might.*switch)'],
            'low': [r'(price.*high|expensive|cost.*much|subscription.*expensive)']
        }
        
        # Churn indicators
        churn_patterns = [r'(leaving|quit|done with|final.*time|last.*chance)',
                         r'(uninstalling|deleted|removed|bye|goodbye)',
                         r'(never.*again|worst.*ever|completely.*useless)']
        
        # Advocacy patterns
        advocacy_patterns = [r'(recommend|tell.*friends|share.*with|suggest.*to)',
                           r'(love.*app|best.*app|amazing|incredible|perfect)',
                           r'(must.*have|essential|can\'t.*live.*without)']
        
        for review in reviews:
            text = review['text'].lower()
            
            # Check revenue risk
            for risk_level, patterns in revenue_risk_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text):
                        impact_signals['revenue_risk'][risk_level] += 1
                        impact_signals['revenue_risk']['signals'].append({
                            'level': risk_level,
                            'review_id': review['review_id'],
                            'phrase': re.search(pattern, text).group(0) if re.search(pattern, text) else '',
                            'rating': review['rating']
                        })
            
            # Check churn indicators
            for pattern in churn_patterns:
                if re.search(pattern, text):
                    impact_signals['churn_indicators']['count'] += 1
                    impact_signals['churn_indicators']['phrases'].append(re.search(pattern, text).group(0))
            
            # Check advocacy signals
            for pattern in advocacy_patterns:
                if re.search(pattern, text):
                    impact_signals['advocacy_signals']['count'] += 1
                    impact_signals['advocacy_signals']['phrases'].append(re.search(pattern, text).group(0))
        
        # Calculate percentages
        total_reviews = len(reviews)
        impact_signals['churn_indicators']['percentage'] = round(
            (impact_signals['churn_indicators']['count'] / total_reviews) * 100, 1
        )
        impact_signals['advocacy_signals']['percentage'] = round(
            (impact_signals['advocacy_signals']['count'] / total_reviews) * 100, 1
        )
        
        return impact_signals
    
    def _analyze_emotional_intensity(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional intensity beyond basic sentiment"""
        emotions = {
            'frustration': {'count': 0, 'phrases': [], 'reviews': []},
            'excitement': {'count': 0, 'phrases': [], 'reviews': []},
            'disappointment': {'count': 0, 'phrases': [], 'reviews': []},
            'satisfaction': {'count': 0, 'phrases': [], 'reviews': []},
            'anger': {'count': 0, 'phrases': [], 'reviews': []},
            'delight': {'count': 0, 'phrases': [], 'reviews': []}
        }
        
        emotion_patterns = {
            'frustration': [r'(frustrated|annoying|irritating|infuriating|fed up)',
                          r'(waste.*time|sick of|can\'t stand|driving.*crazy)'],
            'excitement': [r'(excited|thrilled|amazed|blown away|incredible)',
                         r'(game changer|revolutionary|mind blowing|outstanding)'],
            'disappointment': [r'(disappointed|let down|expected more|underwhelming)',
                             r'(not what.*hoped|thought.*better|hoped for more)'],
            'satisfaction': [r'(satisfied|pleased|happy|content|fulfilled)',
                           r'(exactly what|perfect|just right|exceeded.*expectations)'],
            'anger': [r'(angry|furious|mad|outraged|livid)',
                     r'(hate|despise|terrible|awful|worst ever)'],
            'delight': [r'(delighted|overjoyed|ecstatic|love love|adore)',
                       r'(fantastic|wonderful|brilliant|superb|excellent)']
        }
        
        for review in reviews:
            text = review['text'].lower()
            
            for emotion, patterns in emotion_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        emotions[emotion]['count'] += len(matches)
                        emotions[emotion]['phrases'].extend(matches)
                        emotions[emotion]['reviews'].append({
                            'review_id': review['review_id'],
                            'rating': review['rating'],
                            'snippet': text[:100] + '...' if len(text) > 100 else text
                        })
        
        return emotions
    
    def _extract_competitive_intelligence(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Extract competitive mentions and comparisons"""
        competitors = {
            'social_media': ['instagram', 'facebook', 'twitter', 'tiktok', 'snapchat'],
            'messaging': ['whatsapp', 'telegram', 'signal', 'discord'],
            'productivity': ['notion', 'slack', 'teams', 'zoom', 'asana'],
            'entertainment': ['netflix', 'spotify', 'youtube', 'twitch']
        }
        
        competitive_data = {}
        
        for category, apps in competitors.items():
            competitive_data[category] = {}
            
            for app in apps:
                mentions = []
                for review in reviews:
                    text = review['text'].lower()
                    
                    # Look for competitive mentions with context
                    patterns = [
                        f"(better than {app}|{app} is better|compared to {app})",
                        f"(like {app}|similar to {app}|reminds me of {app})",
                        f"(switch to {app}|moving to {app}|prefer {app})"
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, text):
                            mentions.append({
                                'review_id': review['review_id'],
                                'context': re.search(pattern, text).group(0),
                                'rating': review['rating'],
                                'comparison_type': self._classify_comparison(pattern, text)
                            })
                
                if mentions:
                    competitive_data[category][app] = {
                        'mention_count': len(mentions),
                        'mentions': mentions[:5]  # Top 5 mentions
                    }
        
        return competitive_data
    
    def _classify_comparison(self, pattern: str, text: str) -> str:
        """Classify the type of competitive comparison"""
        if any(word in text for word in ['better', 'superior', 'prefer']):
            return 'competitor_advantage'
        elif any(word in text for word in ['worse', 'not as good', 'inferior']):
            return 'our_advantage'
        elif any(word in text for word in ['like', 'similar', 'reminds']):
            return 'feature_comparison'
        elif any(word in text for word in ['switch', 'moving', 'leaving']):
            return 'churn_signal'
        else:
            return 'general_mention'
    
    def _analyze_actionable_feedback(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Extract actionable feedback with user-suggested solutions"""
        actionable_items = {
            'pain_points_with_solutions': [],
            'feature_gaps': [],
            'urgent_fixes': [],
            'quick_wins': []
        }
        
        # Pattern for pain point + solution statements
        solution_patterns = [
            r'(.+?)(problem|issue|trouble).+?(should|need|want|fix|add|implement)(.+?)(?:[.!?]|$)',
            r'(.+?)(doesn\'t work|broken|fails).+?(if.*could|would be better|suggest)(.+?)(?:[.!?]|$)',
            r'(wish|hope|if only).+?(had|could|would)(.+?)(?:[.!?]|$)'
        ]
        
        for review in reviews:
            text = review['text']
            
            for pattern in solution_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    actionable_items['pain_points_with_solutions'].append({
                        'review_id': review['review_id'],
                        'pain_point': match.group(1).strip() if len(match.groups()) > 0 else '',
                        'suggested_solution': match.group(-1).strip(),
                        'urgency': self._assess_urgency(text),
                        'rating': review['rating'],
                        'full_context': match.group(0)
                    })
        
        return actionable_items
    
    def _assess_urgency(self, text: str) -> str:
        """Assess the urgency level of feedback"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['urgent', 'immediate', 'asap', 'critical', 'emergency']):
            return 'critical'
        elif any(word in text_lower for word in ['important', 'serious', 'soon', 'really need']):
            return 'high'
        elif any(word in text_lower for word in ['would like', 'hope', 'suggestion', 'nice to have']):
            return 'medium'
        else:
            return 'low'
    
    def _advanced_aspect_sentiment(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Advanced aspect-based sentiment analysis"""
        aspects = {
            'user_interface': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': []},
            'performance': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': []},
            'features': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': []},
            'customer_support': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': []},
            'pricing': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': []},
            'reliability': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': []}
        }
        
        aspect_keywords = {
            'user_interface': ['ui', 'interface', 'design', 'layout', 'look', 'appearance'],
            'performance': ['speed', 'fast', 'slow', 'lag', 'performance', 'responsive'],
            'features': ['feature', 'function', 'capability', 'tool', 'option'],
            'customer_support': ['support', 'help', 'customer service', 'assistance'],
            'pricing': ['price', 'cost', 'expensive', 'cheap', 'subscription', 'premium'],
            'reliability': ['reliable', 'stable', 'crash', 'bug', 'error', 'work']
        }
        
        for review in reviews:
            text = review['text'].lower()
            
            for aspect, keywords in aspect_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        # Simple sentiment based on rating
                        if review['rating'] >= 4:
                            sentiment = 'positive'
                        elif review['rating'] <= 2:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                        
                        aspects[aspect][sentiment] += 1
                        aspects[aspect]['mentions'].append({
                            'review_id': review['review_id'],
                            'keyword': keyword,
                            'sentiment': sentiment,
                            'rating': review['rating']
                        })
        
        return aspects
    
    def _generate_advanced_recommendations(self, analysis_data: Dict) -> List[Dict]:
        """Generate advanced recommendations based on analysis"""
        recommendations = []
        
        # Business impact recommendations
        business_intel = analysis_data.get('business_intelligence', {})
        churn_rate = business_intel.get('churn_indicators', {}).get('percentage', 0)
        
        if churn_rate > 10:
            recommendations.append({
                'category': 'Critical Business Risk',
                'priority': 'HIGH',
                'action': f'Address churn indicators immediately - {churn_rate}% of users showing leaving signals',
                'impact': 'Revenue Protection',
                'timeline': 'Within 1 week'
            })
        
        # Emotional analysis recommendations
        emotions = analysis_data.get('emotional_analysis', {})
        frustration_count = emotions.get('frustration', {}).get('count', 0)
        
        if frustration_count > len(analysis_data.get('meta', {}).get('total_reviews', 100)) * 0.15:
            recommendations.append({
                'category': 'User Experience',
                'priority': 'HIGH',
                'action': 'Investigate and address sources of user frustration',
                'impact': 'User Satisfaction',
                'timeline': 'Within 2 weeks'
            })
        
        # Competitive recommendations
        competitive_data = analysis_data.get('competitive_insights', {})
        for category, competitors in competitive_data.items():
            for competitor, data in competitors.items():
                if data.get('mention_count', 0) > 3:
                    recommendations.append({
                        'category': 'Competitive Analysis',
                        'priority': 'MEDIUM',
                        'action': f'Analyze {competitor} features - users making comparisons',
                        'impact': 'Market Position',
                        'timeline': 'Within 1 month'
                    })
        
        return recommendations

    def separate_sentiment_analysis(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Separate reviews into positive and negative sentiments with detailed review extraction
        
        Args:
            reviews: List of review dictionaries with 'review'/'content' and 'rating' keys
            
        Returns:
            Dictionary containing separated positive and negative themes with actual reviews
        """
        try:
            positive_themes = defaultdict(list)
            negative_themes = defaultdict(list)
            
            # Define positive and negative theme categories
            positive_categories = {
                'design_praise': {
                    'name': 'Design & Interface Praise',
                    'keywords': ['beautiful', 'gorgeous', 'stunning', 'elegant', 'clean design', 'nice interface', 
                               'great ui', 'looks good', 'visually appealing', 'attractive', 'polished', 'sleek'],
                    'priority_multiplier': 1.0
                },
                'functionality_praise': {
                    'name': 'Functionality Appreciation',
                    'keywords': ['works perfectly', 'functions well', 'reliable', 'stable', 'smooth', 'efficient',
                               'fast', 'quick', 'responsive', 'seamless', 'flawless', 'consistent'],
                    'priority_multiplier': 1.2
                },
                'feature_love': {
                    'name': 'Feature Appreciation',
                    'keywords': ['love this feature', 'amazing feature', 'great feature', 'useful feature', 
                               'helpful feature', 'convenient', 'powerful', 'innovative', 'brilliant'],
                    'priority_multiplier': 1.1
                },
                'ease_of_use': {
                    'name': 'Ease of Use',
                    'keywords': ['easy to use', 'simple', 'intuitive', 'user friendly', 'straightforward', 
                               'clear', 'logical', 'makes sense', 'no confusion', 'obvious'],
                    'priority_multiplier': 1.0
                },
                'performance_praise': {
                    'name': 'Performance Excellence',
                    'keywords': ['fast loading', 'quick response', 'no lag', 'smooth performance', 'optimized',
                               'lightning fast', 'instant', 'speedy', 'no delays', 'snappy'],
                    'priority_multiplier': 1.3
                },
                'overall_satisfaction': {
                    'name': 'Overall Satisfaction',
                    'keywords': ['love it', 'perfect', 'excellent', 'outstanding', 'amazing', 'fantastic',
                               'wonderful', 'great app', 'best app', 'highly recommend', 'satisfied', 'happy'],
                    'priority_multiplier': 1.0
                }
            }
            
            negative_categories = {
                'usability_issues': {
                    'name': 'Usability Problems',
                    'keywords': ['confusing', 'complicated', 'hard to use', 'difficult', 'not intuitive', 
                               'unclear', 'messy interface', 'poor design', 'bad ui', 'hard to navigate'],
                    'severity_multiplier': 1.2
                },
                'performance_issues': {
                    'name': 'Performance Problems',
                    'keywords': ['slow', 'laggy', 'crashes', 'freezes', 'hangs', 'unresponsive', 'buggy',
                               'glitchy', 'not working', 'broken', 'unstable', 'poor performance'],
                    'severity_multiplier': 1.5
                },
                'feature_complaints': {
                    'name': 'Feature Complaints',
                    'keywords': ['missing feature', 'lacks', 'doesnt have', 'no option', 'limited', 'restricted',
                               'basic', 'primitive', 'outdated', 'behind competition', 'feature poor'],
                    'severity_multiplier': 1.1
                },
                'functionality_issues': {
                    'name': 'Functionality Problems',
                    'keywords': ['not working', 'broken', 'fails', 'error', 'bug', 'issue', 'problem',
                               'malfunction', 'doesnt work', 'stopped working', 'unreliable'],
                    'severity_multiplier': 1.4
                },
                'design_complaints': {
                    'name': 'Design & Interface Issues',
                    'keywords': ['ugly', 'bad design', 'poor layout', 'messy', 'cluttered', 'outdated design',
                               'horrible interface', 'looks bad', 'unattractive', 'unprofessional'],
                    'severity_multiplier': 1.0
                },
                'overall_dissatisfaction': {
                    'name': 'Overall Dissatisfaction',
                    'keywords': ['hate it', 'terrible', 'awful', 'worst', 'horrible', 'disgusting',
                               'disappointed', 'regret', 'waste', 'uninstalling', 'never again'],
                    'severity_multiplier': 1.3
                }
            }
            
            total_reviews = len(reviews)
            classified_reviews = []
            
            for review in reviews:
                text = review.get('review', review.get('content', '')).lower()
                rating = review.get('rating', 3)
                
                if not text:
                    continue
                
                found_categories = []
                
                # Check for positive themes (typically ratings 4-5)
                if rating >= 4:
                    for category_id, category_config in positive_categories.items():
                        for keyword in category_config['keywords']:
                            if keyword in text:
                                # Extract highlighted phrases
                                highlighted_phrases = self._extract_highlighted_phrases(text, keyword, category_id)
                                
                                positive_score = rating * category_config['priority_multiplier']
                                positive_themes[category_id].append({
                                    'review': review,
                                    'keyword': keyword,
                                    'positive_score': positive_score,
                                    'text_snippet': text[:150] + '...' if len(text) > 150 else text,
                                    'highlighted_phrases': highlighted_phrases,
                                    'category_name': category_config['name']
                                })
                                found_categories.append(f'positive_{category_id}')
                                break
                
                # Check for negative themes (typically ratings 1-3)
                if rating <= 3:
                    for category_id, category_config in negative_categories.items():
                        for keyword in category_config['keywords']:
                            if keyword in text:
                                # Extract highlighted phrases
                                highlighted_phrases = self._extract_highlighted_phrases(text, keyword, category_id)
                                
                                severity_score = (4 - rating) * category_config['severity_multiplier']
                                negative_themes[category_id].append({
                                    'review': review,
                                    'keyword': keyword,
                                    'severity_score': severity_score,
                                    'text_snippet': text[:150] + '...' if len(text) > 150 else text,
                                    'highlighted_phrases': highlighted_phrases,
                                    'category_name': category_config['name']
                                })
                                found_categories.append(f'negative_{category_id}')
                                break
                
                classified_reviews.append({
                    'review': review,
                    'categories': found_categories,
                    'sentiment': 'positive' if rating >= 4 else 'negative' if rating <= 2 else 'neutral'
                })
            
            # Calculate summaries for positive themes
            def calculate_positive_summary(themes_dict, category_name):
                if not themes_dict:
                    return []
                
                results = []
                for theme_id, theme_reviews in themes_dict.items():
                    if not theme_reviews:
                        continue
                    
                    # Sort by positive score
                    sorted_reviews = sorted(theme_reviews, key=lambda x: x['positive_score'], reverse=True)
                    
                    # Get actual reviews with enhanced metadata
                    actual_reviews = []
                    for item in sorted_reviews[:15]:  # Top 15 examples
                        review_data = item['review']
                        actual_reviews.append({
                            'review_id': f"review_{len(actual_reviews)}",
                            'author': 'Anonymous',  # Anonymized
                            'rating': review_data.get('rating', 0),
                            'date': review_data.get('date', ''),
                            'full_text': review_data.get('review', review_data.get('content', '')),
                            'highlighted_phrases': item['highlighted_phrases'],
                            'matched_keyword': item['keyword']
                        })
                    
                    avg_score = sum(item['positive_score'] for item in theme_reviews) / len(theme_reviews)
                    avg_rating = sum(item['review'].get('rating', 0) for item in theme_reviews) / len(theme_reviews)
                    
                    category_config = positive_categories.get(theme_id, {})
                    
                    results.append({
                        'theme_type': category_config.get('name', theme_id.replace('_', ' ').title()),
                        'praise_count': len(theme_reviews),
                        'percentage': round((len(theme_reviews) / total_reviews) * 100, 1),
                        'satisfaction_level': self._get_satisfaction_level(avg_score),
                        'combined_score': round(avg_score, 2),
                        'average_rating': round(avg_rating, 1),
                        'actual_reviews': actual_reviews
                    })
                
                return sorted(results, key=lambda x: x['combined_score'], reverse=True)
            
            # Calculate summaries for negative themes
            def calculate_negative_summary(themes_dict, category_name):
                if not themes_dict:
                    return []
                
                results = []
                for theme_id, theme_reviews in themes_dict.items():
                    if not theme_reviews:
                        continue
                    
                    # Sort by severity score
                    sorted_reviews = sorted(theme_reviews, key=lambda x: x['severity_score'], reverse=True)
                    
                    # Get actual reviews with enhanced metadata
                    actual_reviews = []
                    for item in sorted_reviews[:15]:  # Top 15 examples
                        review_data = item['review']
                        actual_reviews.append({
                            'review_id': f"review_{len(actual_reviews)}",
                            'author': 'Anonymous',  # Anonymized
                            'rating': review_data.get('rating', 0),
                            'date': review_data.get('date', ''),
                            'full_text': review_data.get('review', review_data.get('content', '')),
                            'highlighted_phrases': item['highlighted_phrases'],
                            'matched_keyword': item['keyword']
                        })
                    
                    avg_score = sum(item['severity_score'] for item in theme_reviews) / len(theme_reviews)
                    avg_rating = sum(item['review'].get('rating', 0) for item in theme_reviews) / len(theme_reviews)
                    
                    category_config = negative_categories.get(theme_id, {})
                    
                    results.append({
                        'issue_type': category_config.get('name', theme_id.replace('_', ' ').title()),
                        'complaint_count': len(theme_reviews),
                        'percentage': round((len(theme_reviews) / total_reviews) * 100, 1),
                        'severity_level': self._get_severity_level(avg_score),
                        'combined_score': round(avg_score, 2),
                        'average_rating': round(avg_rating, 1),
                        'actual_reviews': actual_reviews
                    })
                
                return sorted(results, key=lambda x: x['combined_score'], reverse=True)
            
            # Generate results
            positive_summary = calculate_positive_summary(positive_themes, 'Positive Themes')
            negative_summary = calculate_negative_summary(negative_themes, 'Negative Themes')
            
            # Calculate overall sentiment distribution
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for review in classified_reviews:
                sentiment_counts[review['sentiment']] += 1
            
            return {
                'positive_themes': positive_summary,
                'negative_themes': negative_summary,
                'sentiment_distribution': sentiment_counts,
                'classification_meta': {
                    'total_reviews_analyzed': total_reviews,
                    'positive_reviews': sentiment_counts['positive'],
                    'negative_reviews': sentiment_counts['negative'],
                    'neutral_reviews': sentiment_counts['neutral'],
                    'positive_percentage': round((sentiment_counts['positive'] / total_reviews) * 100, 1),
                    'negative_percentage': round((sentiment_counts['negative'] / total_reviews) * 100, 1),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Sentiment separation analysis failed: {e}")
            return {
                'positive_themes': [],
                'negative_themes': [],
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'error': f'Sentiment analysis failed: {str(e)}'
            }
    
    def _get_satisfaction_level(self, score: float) -> str:
        """Get satisfaction level from positive score"""
        if score >= 4.5:
            return "Very High"
        elif score >= 3.5:
            return "High"
        elif score >= 2.5:
            return "Medium"
        else:
            return "Low"

# No global instance - will be created in main.py 