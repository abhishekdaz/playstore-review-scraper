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
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

# Enhanced pipeline imports
try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    KEYBERT_AVAILABLE = True
    logger.info("KeyBERT and Sentence Transformers available - Enhanced phrase extraction enabled")
except ImportError:
    KEYBERT_AVAILABLE = False
    logger.warning("KeyBERT not available - Using fallback phrase extraction")

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
    logger.info("SpaCy available - Enhanced text processing enabled")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available - Using fallback text processing")

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    CLUSTERING_AVAILABLE = True
    logger.info("Advanced clustering algorithms available")
except ImportError:
    CLUSTERING_AVAILABLE = False
    logger.warning("Advanced clustering not available")

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

class EnhancedSentimentPipeline:
    """
    Advanced sentiment analysis and issue extraction pipeline
    Uses RoBERTa for sentiment, KeyBERT for phrase extraction, SpaCy for parsing,
    and Agglomerative Clustering for grouping similar complaints
    """
    
    def __init__(self):
        # Initialize RoBERTa sentiment model
        self.roberta_pipeline = None
        self.keybert_model = None
        self.spacy_nlp = None
        self.sentence_model = None
        
        # Severity patterns
        self.severity_patterns = {
            'critical': [
                r'\b(crash|crashes|crashing|frozen|freeze|freezes|not working|broken|unusable|lost.*data|data.*lost)\b',
                r'\b(can\'t.*open|won\'t.*start|completely.*broken|totally.*broken|doesn\'t.*work)\b',
                r'\b(refund|money.*back|fraud|scam|charged.*twice|unauthorized.*charge)\b'
            ],
            'severe': [
                r'\b(slow|lag|laggy|sluggish|battery.*drain|overheating|hot|very.*slow)\b',
                r'\b(bug|bugs|error|errors|glitch|glitches|problem|problems|issue|issues)\b',
                r'\b(annoying|frustrating|terrible|awful|hate.*it|worst.*app)\b'
            ],
            'minor': [
                r'\b(could.*be.*better|minor.*issue|small.*problem|slightly.*slow)\b',
                r'\b(would.*like|suggestion|feature.*request|improvement)\b'
            ]
        }
        
        # Domain-specific patterns
        self.fraud_patterns = [
            r'\b(fraud|scam|scammer|fake|steal|stolen|unauthorized|illegal)\b',
            r'\b(suspicious.*activity|account.*hacked|identity.*theft)\b',
            r'\b(phishing|malware|virus|security.*breach)\b'
        ]
        
        self.money_patterns = [
            r'\b(money|payment|pay|paid|charge|charged|bill|billing|refund|cost|price|expensive)\b',
            r'\b(credit.*card|debit.*card|transaction|purchase|buy|bought|subscription)\b',
            r'\b(\$\d+|\d+.*dollar|dollar.*\d+|free.*trial|premium|upgrade)\b'
        ]
        
        self.update_patterns = [
            r'\b(update|updated|version|new.*version|latest.*version|upgrade|upgraded)\b',
            r'\b(after.*update|since.*update|new.*update|recent.*update)\b',
            r'\b(broke.*after|broken.*since|worse.*after|stopped.*working.*after)\b'
        ]
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models for the pipeline"""
        try:
            # Initialize RoBERTa sentiment model
            if TRANSFORMERS_AVAILABLE:
                logger.info("Loading RoBERTa sentiment model...")
                self.roberta_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("RoBERTa model loaded successfully")
            
            # Initialize KeyBERT model
            if KEYBERT_AVAILABLE:
                logger.info("Loading KeyBERT model...")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.keybert_model = KeyBERT(model=self.sentence_model)
                logger.info("KeyBERT model loaded successfully")
            
            # Initialize SpaCy model
            if SPACY_AVAILABLE:
                try:
                    logger.info("Loading SpaCy English model...")
                    self.spacy_nlp = spacy.load("en_core_web_sm")
                    logger.info("SpaCy model loaded successfully")
                except OSError:
                    logger.warning("SpaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                    self.spacy_nlp = None
        
        except Exception as e:
            logger.error(f"Error initializing enhanced pipeline models: {e}")
    
    def analyze_single_review(self, review_text: str, rating: int) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single review
        """
        result = {
            'original_text': review_text,
            'rating': rating,
            'sentiment': {},
            'key_phrases': [],
            'severity': 'minor',
            'severity_score': 0.0,
            'fraud_related': False,
            'money_related': False,
            'update_related': False,
            'complaint_themes': []
        }
        
        # 1. RoBERTa Sentiment Analysis
        result['sentiment'] = self._analyze_sentiment_roberta(review_text)
        
        # 2. Extract key complaint phrases using KeyBERT
        result['key_phrases'] = self._extract_key_phrases(review_text)
        
        # 3. Detect specific issue types
        result['fraud_related'] = self._detect_fraud_issues(review_text)
        result['money_related'] = self._detect_money_issues(review_text)
        result['update_related'] = self._detect_update_issues(review_text)
        
        # 4. Classify severity
        result['severity'], result['severity_score'] = self._classify_severity(review_text, rating)
        
        # 5. Extract complaint themes
        result['complaint_themes'] = self._extract_complaint_themes(review_text)
        
        return result
    
    def _analyze_sentiment_roberta(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using RoBERTa model"""
        if not self.roberta_pipeline:
            return {'sentiment': 'unknown', 'confidence': 0.0, 'available': False}
        
        try:
            # Preprocess for RoBERTa
            processed_text = re.sub(r'@\w+', '@user', text)
            processed_text = re.sub(r'http\S+|www\S+|https\S+', 'http', processed_text)
            
            # Truncate if too long
            if len(processed_text) > 400:
                processed_text = processed_text[:400]
            
            result = self.roberta_pipeline(processed_text)[0]
            
            # Map labels
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive'
            }
            
            sentiment = label_mapping.get(result['label'], result['label'].lower())
            confidence = result['score']
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'available': True,
                'model': 'roberta'
            }
        
        except Exception as e:
            logger.error(f"RoBERTa sentiment analysis failed: {e}")
            return {'sentiment': 'unknown', 'confidence': 0.0, 'available': False, 'error': str(e)}
    
    def _extract_key_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract key complaint phrases using KeyBERT with maxsum and trigrams"""
        if not self.keybert_model:
            return []
        
        try:
            # Extract keyphrases with different n-gram ranges
            keyphrases = []
            
            # Extract unigrams, bigrams, and trigrams
            for ngram_range in [(1, 1), (1, 2), (1, 3)]:
                phrases = self.keybert_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=ngram_range,
                    stop_words='english',
                    use_maxsum=True,  # Use MaxSum for diversity
                    nr_candidates=20,
                    top_k=5
                )
                
                for phrase, score in phrases:
                    # Filter out generic phrases
                    if len(phrase) > 2 and not phrase.lower() in ['app', 'good', 'bad', 'nice']:
                        keyphrases.append({
                            'phrase': phrase,
                            'relevance_score': score,
                            'ngram_type': f"{ngram_range[0]}-{ngram_range[1]}gram"
                        })
            
            # Sort by relevance and remove duplicates
            unique_phrases = {}
            for phrase_data in keyphrases:
                phrase = phrase_data['phrase']
                if phrase not in unique_phrases or phrase_data['relevance_score'] > unique_phrases[phrase]['relevance_score']:
                    unique_phrases[phrase] = phrase_data
            
            result = list(unique_phrases.values())
            result.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return result[:10]  # Return top 10 phrases
        
        except Exception as e:
            logger.error(f"KeyBERT phrase extraction failed: {e}")
            return []
    
    def _detect_fraud_issues(self, text: str) -> bool:
        """Detect fraud-related mentions using keyword patterns"""
        text_lower = text.lower()
        for pattern in self.fraud_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def _detect_money_issues(self, text: str) -> bool:
        """Detect money-related mentions using keyword patterns"""
        text_lower = text.lower()
        for pattern in self.money_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def _detect_update_issues(self, text: str) -> bool:
        """Detect update-related issues using keyword patterns"""
        text_lower = text.lower()
        for pattern in self.update_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def _classify_severity(self, text: str, rating: int) -> Tuple[str, float]:
        """Classify severity based on text patterns and rating"""
        text_lower = text.lower()
        severity_scores = {'critical': 0, 'severe': 0, 'minor': 0}
        
        # Pattern matching
        for severity, patterns in self.severity_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                severity_scores[severity] += matches
        
        # Rating influence
        if rating <= 2:
            severity_scores['critical'] += 2
            severity_scores['severe'] += 1
        elif rating == 3:
            severity_scores['severe'] += 1
        
        # Text length influence (longer complaints tend to be more severe)
        if len(text) > 200:
            severity_scores['severe'] += 1
        
        # Determine final severity
        max_score = max(severity_scores.values())
        if max_score == 0:
            return 'minor', 0.1
        
        for severity, score in severity_scores.items():
            if score == max_score:
                normalized_score = min(score / 5.0, 1.0)  # Normalize to 0-1
                return severity, normalized_score
        
        return 'minor', 0.1
    
    def _extract_complaint_themes(self, text: str) -> List[str]:
        """Extract complaint themes using SpaCy dependency parsing"""
        if not self.spacy_nlp:
            # Fallback to simple keyword extraction
            return self._fallback_theme_extraction(text)
        
        try:
            doc = self.spacy_nlp(text)
            themes = []
            
            # Look for negative adjectives with objects
            for token in doc:
                if token.pos_ == 'ADJ' and token.sentiment < 0:  # Negative adjective
                    # Find what the adjective is describing
                    for child in token.children:
                        if child.pos_ in ['NOUN', 'PROPN']:
                            themes.append(f"{token.lemma_} {child.lemma_}")
                
                # Look for complaint patterns: "problem with X", "issue with X"
                if token.lemma_ in ['problem', 'issue', 'bug', 'error']:
                    for child in token.children:
                        if child.dep_ == 'prep' and child.lemma_ == 'with':
                            for grandchild in child.children:
                                if grandchild.pos_ in ['NOUN', 'PROPN']:
                                    themes.append(f"{token.lemma_} with {grandchild.lemma_}")
            
            return list(set(themes))[:5]  # Return unique themes, max 5
        
        except Exception as e:
            logger.error(f"SpaCy theme extraction failed: {e}")
            return self._fallback_theme_extraction(text)
    
    def _fallback_theme_extraction(self, text: str) -> List[str]:
        """Fallback theme extraction using simple patterns"""
        themes = []
        text_lower = text.lower()
        
        # Simple pattern matching for common complaint themes
        theme_patterns = {
            'performance': r'\b(slow|lag|performance|speed|fast|quick)\b',
            'crashes': r'\b(crash|freeze|stuck|hang|quit)\b',
            'interface': r'\b(ui|interface|design|layout|button|menu)\b',
            'features': r'\b(feature|function|option|setting|tool)\b',
            'bugs': r'\b(bug|error|glitch|problem|issue)\b',
            'ads': r'\b(ad|ads|advertisement|banner|popup)\b',
            'payment': r'\b(pay|payment|charge|bill|money|cost|price)\b',
            'login': r'\b(login|log.*in|sign.*in|account|password)\b'
        }
        
        for theme, pattern in theme_patterns.items():
            if re.search(pattern, text_lower):
                themes.append(theme)
        
        return themes
    
    def cluster_complaints(self, analyzed_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Group similar complaints using Agglomerative Clustering
        """
        if not CLUSTERING_AVAILABLE or len(analyzed_reviews) < 2:
            return self._fallback_clustering(analyzed_reviews)
        
        try:
            # Extract features for clustering
            complaint_texts = []
            complaint_data = []
            
            for review in analyzed_reviews:
                if review['sentiment'].get('sentiment') in ['negative', 'neutral'] and review['key_phrases']:
                    # Use key phrases as feature representation
                    phrase_text = ' '.join([phrase['phrase'] for phrase in review['key_phrases']])
                    complaint_texts.append(phrase_text)
                    complaint_data.append(review)
            
            if len(complaint_texts) < 2:
                return self._fallback_clustering(analyzed_reviews)
            
            # Create embeddings using sentence transformer
            if self.sentence_model:
                embeddings = self.sentence_model.encode(complaint_texts)
            else:
                # Fallback to TF-IDF
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                embeddings = vectorizer.fit_transform(complaint_texts).toarray()
            
            # Determine optimal number of clusters
            n_clusters = min(max(2, len(complaint_texts) // 5), 10)
            
            # Perform clustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clustering.fit_predict(embeddings)
            
            # Group reviews by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(complaint_data[i])
            
            # Analyze each cluster
            cluster_analysis = {}
            for cluster_id, cluster_reviews in clusters.items():
                cluster_analysis[f"cluster_{cluster_id}"] = self._analyze_cluster(cluster_reviews)
            
            return {
                'clusters': cluster_analysis,
                'total_clusters': len(clusters),
                'total_complaints': len(complaint_data),
                'clustering_method': 'agglomerative'
            }
        
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return self._fallback_clustering(analyzed_reviews)
    
    def _analyze_cluster(self, cluster_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a cluster of similar reviews"""
        if not cluster_reviews:
            return {}
        
        # Extract top complaint phrases
        phrase_counts = Counter()
        severity_scores = []
        money_involved = 0
        update_related = 0
        
        for review in cluster_reviews:
            # Count phrase frequencies
            for phrase_data in review['key_phrases']:
                phrase_counts[phrase_data['phrase']] += 1
            
            # Collect severity scores
            severity_scores.append(review['severity_score'])
            
            # Count special flags
            if review['money_related']:
                money_involved += 1
            if review['update_related']:
                update_related += 1
        
        # Calculate cluster statistics
        avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0
        top_phrases = phrase_counts.most_common(5)
        
        # Determine dominant severity level
        severity_counts = Counter([review['severity'] for review in cluster_reviews])
        dominant_severity = severity_counts.most_common(1)[0][0]
        
        return {
            'count': len(cluster_reviews),
            'top_complaint_phrases': [{'phrase': phrase, 'count': count} for phrase, count in top_phrases],
            'average_severity_score': round(avg_severity, 3),
            'dominant_severity': dominant_severity,
            'money_involved_count': money_involved,
            'money_involved_percentage': round((money_involved / len(cluster_reviews)) * 100, 1),
            'update_related_count': update_related,
            'update_related_percentage': round((update_related / len(cluster_reviews)) * 100, 1),
            'sample_reviews': [
                {
                    'text': review['original_text'][:200] + '...' if len(review['original_text']) > 200 else review['original_text'],
                    'rating': review['rating'],
                    'severity': review['severity']
                }
                for review in cluster_reviews[:3]  # Include 3 sample reviews
            ]
        }
    
    def _fallback_clustering(self, analyzed_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback clustering based on severity and themes"""
        clusters = {'high_severity': [], 'medium_severity': [], 'low_severity': []}
        
        for review in analyzed_reviews:
            if review['severity'] == 'critical':
                clusters['high_severity'].append(review)
            elif review['severity'] == 'severe':
                clusters['medium_severity'].append(review)
            else:
                clusters['low_severity'].append(review)
        
        # Analyze each cluster
        cluster_analysis = {}
        for cluster_name, cluster_reviews in clusters.items():
            if cluster_reviews:
                cluster_analysis[cluster_name] = self._analyze_cluster(cluster_reviews)
        
        return {
            'clusters': cluster_analysis,
            'total_clusters': len(cluster_analysis),
            'total_complaints': len(analyzed_reviews),
            'clustering_method': 'severity_based_fallback'
        }
    
    def process_reviews_batch(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of reviews through the complete enhanced pipeline
        """
        logger.info(f"Processing {len(reviews)} reviews through enhanced pipeline...")
        
        # Step 1: Analyze each review individually
        analyzed_reviews = []
        for review in reviews:
            review_text = review.get('content', review.get('review', ''))
            rating = review.get('rating', 3)
            
            if review_text.strip():
                analysis = self.analyze_single_review(review_text, rating)
                analyzed_reviews.append(analysis)
        
        # Step 2: Cluster similar complaints
        clustering_result = self.cluster_complaints(analyzed_reviews)
        
        # Step 3: Generate summary statistics
        summary = self._generate_pipeline_summary(analyzed_reviews, clustering_result)
        
        return {
            'individual_analyses': analyzed_reviews,
            'clustering': clustering_result,
            'summary': summary,
            'total_processed': len(analyzed_reviews)
        }
    
    def _generate_pipeline_summary(self, analyzed_reviews: List[Dict[str, Any]], clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the enhanced pipeline"""
        if not analyzed_reviews:
            return {}
        
        # Sentiment distribution
        sentiment_counts = Counter([review['sentiment'].get('sentiment', 'unknown') for review in analyzed_reviews])
        
        # Severity distribution
        severity_counts = Counter([review['severity'] for review in analyzed_reviews])
        
        # Issue type distribution
        fraud_count = sum(1 for review in analyzed_reviews if review['fraud_related'])
        money_count = sum(1 for review in analyzed_reviews if review['money_related'])
        update_count = sum(1 for review in analyzed_reviews if review['update_related'])
        
        # Top complaint phrases across all reviews
        all_phrases = Counter()
        for review in analyzed_reviews:
            for phrase_data in review['key_phrases']:
                all_phrases[phrase_data['phrase']] += 1
        
        total = len(analyzed_reviews)
        
        return {
            'sentiment_distribution': {
                sentiment: {'count': count, 'percentage': round((count / total) * 100, 1)}
                for sentiment, count in sentiment_counts.items()
            },
            'severity_distribution': {
                severity: {'count': count, 'percentage': round((count / total) * 100, 1)}
                for severity, count in severity_counts.items()
            },
            'issue_types': {
                'fraud_related': {'count': fraud_count, 'percentage': round((fraud_count / total) * 100, 1)},
                'money_related': {'count': money_count, 'percentage': round((money_count / total) * 100, 1)},
                'update_related': {'count': update_count, 'percentage': round((update_count / total) * 100, 1)}
            },
            'top_complaint_phrases': [
                {'phrase': phrase, 'count': count, 'percentage': round((count / total) * 100, 1)}
                for phrase, count in all_phrases.most_common(10)
            ],
            'clusters_found': clustering_result.get('total_clusters', 0),
            'average_confidence': round(
                sum(review['sentiment'].get('confidence', 0) for review in analyzed_reviews) / total, 3
            ) if total > 0 else 0
        }

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
        
        # Initialize enhanced sentiment pipeline
        self.enhanced_pipeline = EnhancedSentimentPipeline()
        
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
        
        # Calculate combined tech issues metrics
        tech_related_themes = ['Crashes & Bugs', 'Performance Issues', 'Login & Account Problems']
        tech_issues_count = 0
        tech_issues_percentage = 0.0
        
        for theme in insights['critical_themes']:
            if theme['theme_name'] in tech_related_themes:
                tech_issues_count += theme['review_count']
                tech_issues_percentage += theme['percentage']
        
        insights['tech_issues_summary'] = {
            'total_count': tech_issues_count,
            'total_percentage': round(tech_issues_percentage, 1),
            'theme_count': len([t for t in insights['critical_themes'] if t['theme_name'] in tech_related_themes])
        }
        
        # Generate comprehensive complaint cluster analysis
        if 'reviews' in analysis_results:
            insights['complaint_clusters'] = self.analyze_complaint_clusters(analysis_results['reviews'])
        
        # Add comprehensive negative review cluster analysis (new 6-step process)
        try:
            logger.info("Performing comprehensive 6-step negative review cluster analysis...")
            negative_cluster_analysis = self.analyze_negative_review_clusters(analysis_results.get('reviews', []))
            insights['negative_cluster_analysis'] = negative_cluster_analysis
        except Exception as e:
            logger.warning(f"Negative cluster analysis failed: {e}")
            insights['negative_cluster_analysis'] = {
                "total_negative_reviews": 0,
                "cluster_summary": []
            }
        
        # Add critical user complaints analysis (improved criticality scoring)
        try:
            logger.info("Performing critical user complaints analysis with improved scoring...")
            critical_complaints_analysis = self.analyze_critical_user_complaints(analysis_results.get('reviews', []))
            insights['critical_user_complaints'] = critical_complaints_analysis
        except Exception as e:
            logger.warning(f"Critical user complaints analysis failed: {e}")
            insights['critical_user_complaints'] = {
                "total_negative_reviews": 0,
                "critical_issues": []
            }
        
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
        Enhanced with cluster complaint phrases and frequency data
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
            phrase_frequencies = {}
            
            for review in negative_reviews:
                review_text = review.get('review', '').lower()
                if not review_text:
                    review_text = review.get('content', '').lower()
                
                # Check for keyword matches
                keyword_matches = [kw for kw in theme_data['keywords'] if kw in review_text]
                
                # Check for specific problematic phrases with frequency counting
                phrase_matches = []
                for phrase in theme_data['problematic_phrases']:
                    if phrase in review_text:
                        phrase_matches.append(phrase)
                        phrase_frequencies[phrase] = phrase_frequencies.get(phrase, 0) + 1
                
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
                
                # Get unique phrases and keywords with their frequencies
                unique_phrases = list(set(found_phrases))[:10]  # Top 10 unique phrases
                
                # Create phrase frequency list with severity scores
                frequent_complaints = []
                for phrase, count in sorted(phrase_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]:
                    # Calculate average severity for this phrase
                    phrase_severities = []
                    for review in matching_reviews:
                        review_text = review.get('review', '').lower()
                        if phrase in review_text:
                            # Use rating as inverse severity (1 star = high severity)
                            severity_score = (6 - review.get('rating', 3)) / 5.0  # Convert to 0-1 scale
                            phrase_severities.append(severity_score)
                    
                    avg_severity = sum(phrase_severities) / len(phrase_severities) if phrase_severities else 0.5
                    
                    frequent_complaints.append({
                        "phrase": phrase,
                        "frequency": count,
                        "avg_severity": round(avg_severity, 2)
                    })
                
                critical_themes.append({
                    'theme_name': theme_name,
                    'keywords': theme_data['keywords'][:8],  # Top 8 keywords
                    'review_count': len(matching_reviews),
                    'percentage': round(frequency_percentage, 1),
                    'severity': severity,
                    'problematic_phrases': unique_phrases,
                    'frequent_complaints': frequent_complaints,  # New field with frequency data
                    'scoring_formula': 'frequency × avg_severity (descending)'  # Show formula
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
    
    def extract_cluster_issue_phrases(self, negative_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract high-quality issue phrases from clustered negative reviews.
        
        Args:
            negative_reviews: List of review dicts with 'content', 'sentiment', and 'cluster' keys
            
        Returns:
            List of cluster analysis dicts with top issue phrases
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available - using fallback phrase extraction")
            return self._fallback_cluster_phrase_extraction(negative_reviews)
        
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            import re
            
            # Group reviews by cluster
            clusters = defaultdict(list)
            for review in negative_reviews:
                cluster_id = review.get('cluster', 0)
                clusters[cluster_id].append(review)
            
            cluster_results = []
            
            for cluster_id, cluster_reviews in clusters.items():
                if not cluster_reviews:
                    continue
                
                logger.info(f"Processing cluster {cluster_id} with {len(cluster_reviews)} reviews")
                
                # Step 1: Preprocess reviews - extract content and clean
                texts = []
                review_sentiments = []
                
                for review in cluster_reviews:
                    content = review.get('content', '')
                    if content:
                        # Basic preprocessing - lowercase and basic cleaning
                        cleaned_content = content.lower().strip()
                        texts.append(cleaned_content)
                        
                        # Extract sentiment score
                        sentiment = review.get('sentiment', {})
                        if isinstance(sentiment, dict):
                            sentiment_score = abs(sentiment.get('score', 0.5))  # Use absolute value for severity
                        else:
                            sentiment_score = 0.5  # Default moderate severity
                        review_sentiments.append(sentiment_score)
                
                if not texts:
                    continue
                
                # Step 2: Use CountVectorizer to extract top phrases
                try:
                    vectorizer = CountVectorizer(
                        ngram_range=(2, 4),
                        stop_words='english',
                        min_df=1,
                        max_df=0.9,
                        lowercase=True,
                        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens, min 2 chars
                    )
                    
                    # Fit and transform the texts
                    count_matrix = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get phrase frequencies
                    phrase_frequencies = count_matrix.sum(axis=0).A1
                    
                    # Step 3: Calculate metrics for each phrase
                    phrase_metrics = []
                    
                    for idx, phrase in enumerate(feature_names):
                        frequency = int(phrase_frequencies[idx])
                        if frequency == 0:
                            continue
                        
                        # Find reviews containing this phrase and calculate average sentiment
                        matching_reviews = []
                        phrase_sentiments = []
                        
                        for i, text in enumerate(texts):
                            if phrase in text:
                                matching_reviews.append(i)
                                phrase_sentiments.append(review_sentiments[i])
                        
                        if phrase_sentiments:
                            avg_severity = sum(phrase_sentiments) / len(phrase_sentiments)
                            
                            # Calculate combined score (frequency × severity)
                            combined_score = frequency * avg_severity
                            
                            phrase_metrics.append({
                                "phrase": phrase,
                                "frequency": frequency,
                                "avg_severity": round(avg_severity, 3),
                                "combined_score": round(combined_score, 3)
                            })
                    
                    # Step 4 & 5: Sort by combined score and get top 5
                    phrase_metrics.sort(key=lambda x: x['combined_score'], reverse=True)
                    top_phrases = phrase_metrics[:5]
                    
                    cluster_results.append({
                        "cluster": int(cluster_id),
                        "review_count": len(cluster_reviews),
                        "phrases": top_phrases
                    })
                    
                except Exception as e:
                    logger.warning(f"CountVectorizer failed for cluster {cluster_id}: {e}")
                    # Fallback to simple phrase extraction for this cluster
                    fallback_phrases = self._simple_phrase_extraction(texts, review_sentiments)
                    cluster_results.append({
                        "cluster": int(cluster_id),
                        "review_count": len(cluster_reviews),
                        "phrases": fallback_phrases[:5]
                    })
            
            # Sort clusters by review count (largest first)
            cluster_results.sort(key=lambda x: x['review_count'], reverse=True)
            
            logger.info(f"Successfully extracted phrases from {len(cluster_results)} clusters")
            return cluster_results
            
        except Exception as e:
            logger.error(f"Cluster phrase extraction failed: {e}")
            return self._fallback_cluster_phrase_extraction(negative_reviews)
    
    def _simple_phrase_extraction(self, texts: List[str], sentiments: List[float]) -> List[Dict[str, Any]]:
        """Simple fallback phrase extraction using basic n-gram counting"""
        from collections import Counter
        import re
        
        # Simple n-gram extraction
        phrases = []
        for text in texts:
            # Remove punctuation and split into words
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            
            # Generate 2-4 grams
            for n in range(2, 5):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Calculate metrics
        phrase_metrics = []
        for phrase, frequency in phrase_counts.most_common(20):  # Top 20 to filter from
            # Find matching reviews
            matching_sentiments = []
            for i, text in enumerate(texts):
                if phrase in text.lower():
                    matching_sentiments.append(sentiments[i])
            
            if matching_sentiments and frequency >= 1:
                avg_severity = sum(matching_sentiments) / len(matching_sentiments)
                combined_score = frequency * avg_severity
                
                phrase_metrics.append({
                    "phrase": phrase,
                    "frequency": frequency,
                    "avg_severity": round(avg_severity, 3),
                    "combined_score": round(combined_score, 3)
                })
        
        # Sort by combined score
        phrase_metrics.sort(key=lambda x: x['combined_score'], reverse=True)
        return phrase_metrics
    
    def _fallback_cluster_phrase_extraction(self, negative_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback phrase extraction when scikit-learn is not available"""
        try:
            # Group by cluster
            clusters = defaultdict(list)
            for review in negative_reviews:
                cluster_id = review.get('cluster', 0)
                clusters[cluster_id].append(review)
            
            cluster_results = []
            
            for cluster_id, cluster_reviews in clusters.items():
                texts = []
                sentiments = []
                
                for review in cluster_reviews:
                    content = review.get('content', '')
                    if content:
                        texts.append(content.lower())
                        
                        sentiment = review.get('sentiment', {})
                        if isinstance(sentiment, dict):
                            sentiment_score = abs(sentiment.get('score', 0.5))
                        else:
                            sentiment_score = 0.5
                        sentiments.append(sentiment_score)
                
                if texts:
                    phrases = self._simple_phrase_extraction(texts, sentiments)
                    cluster_results.append({
                        "cluster": int(cluster_id),
                        "review_count": len(cluster_reviews),
                        "phrases": phrases[:5]
                    })
            
            cluster_results.sort(key=lambda x: x['review_count'], reverse=True)
            return cluster_results
            
        except Exception as e:
            logger.error(f"Fallback cluster phrase extraction failed: {e}")
            return []

    def analyze_complaint_clusters(self, reviews_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive complaint clustering and analysis following structured approach
        
        Steps:
        1. Filter negative reviews (score <= 2, NEGATIVE sentiment)
        2. Cluster complaints using sentence transformers
        3. Extract meaningful phrases per cluster
        4. Compute criticality scores
        5. Build structured output
        6. Return UI-ready data
        """
        try:
            # Step 1: Preprocessing
            print("Step 1: Preprocessing negative reviews...")
            negative_reviews = []
            
            for review in reviews_data:
                # Filter by score and sentiment
                score = review.get('score', 5)
                sentiment = review.get('sentiment', {})
                sentiment_label = sentiment.get('label', 'NEUTRAL')
                
                if score <= 2 and sentiment_label.lower() == 'negative':
                    # Default thumbsUpCount if missing
                    if 'thumbsUpCount' not in review or review['thumbsUpCount'] is None:
                        review['thumbsUpCount'] = 1
                    negative_reviews.append(review)
            
            total_negative_reviews = len(negative_reviews)
            print(f"Total negative reviews filtered: {total_negative_reviews}")
            
            if total_negative_reviews < 5:
                return {
                    'total_negative_reviews': total_negative_reviews,
                    'cluster_summary': [],
                    'message': 'Insufficient negative reviews for clustering analysis'
                }
            
            # Step 2: Cluster the complaints
            print("Step 2: Clustering complaints...")
            review_texts = [review['content'] for review in negative_reviews]
            
            # Use sentence transformers for embedding
            if hasattr(self, 'sentence_model') and self.sentence_model:
                embeddings = self.sentence_model.encode(review_texts)
                
                # Use AgglomerativeClustering with distance threshold
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(
                    distance_threshold=1.0,
                    n_clusters=None,
                    linkage='ward'
                )
                cluster_labels = clustering.fit_predict(embeddings)
            else:
                # Fallback to simple clustering if sentence transformers not available
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.cluster import KMeans
                
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(review_texts)
                
                # Use KMeans with automatic cluster number estimation
                n_clusters = min(10, max(2, total_negative_reviews // 20))
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clustering.fit_predict(tfidf_matrix)
            
            # Assign cluster IDs to reviews
            clustered_reviews = []
            for i, review in enumerate(negative_reviews):
                review['cluster_id'] = int(cluster_labels[i])
                clustered_reviews.append(review)
            
            print(f"Created {len(set(cluster_labels))} clusters")
            
            # Step 3: Extract meaningful complaint phrases for each cluster
            print("Step 3: Extracting complaint phrases...")
            cluster_data = {}
            
            for cluster_id in set(cluster_labels):
                cluster_reviews = [r for r in clustered_reviews if r['cluster_id'] == cluster_id]
                cluster_texts = [r['content'] for r in cluster_reviews]
                
                # Extract phrases using CountVectorizer
                from sklearn.feature_extraction.text import CountVectorizer
                vectorizer = CountVectorizer(
                    ngram_range=(2, 4),
                    stop_words='english',
                    max_features=50
                )
                
                try:
                    phrase_matrix = vectorizer.fit_transform(cluster_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    phrase_counts = phrase_matrix.sum(axis=0).A1
                    
                    # Calculate phrase metrics
                    phrase_data = []
                    for i, phrase in enumerate(feature_names):
                        frequency = int(phrase_counts[i])
                        
                        # Calculate average severity for reviews containing this phrase
                        phrase_reviews = [r for r in cluster_reviews if phrase.lower() in r['content'].lower()]
                        if phrase_reviews:
                            avg_severity = sum(r['sentiment']['score'] for r in phrase_reviews) / len(phrase_reviews)
                            phrase_data.append({
                                'phrase': phrase,
                                'frequency': frequency,
                                'avg_severity': round(avg_severity, 3)
                            })
                    
                    # Sort by frequency × avg_severity and keep top 5
                    phrase_data.sort(key=lambda x: x['frequency'] * x['avg_severity'], reverse=True)
                    top_phrases = phrase_data[:5]
                    
                except Exception as e:
                    print(f"Error extracting phrases for cluster {cluster_id}: {e}")
                    top_phrases = []
                
                cluster_data[cluster_id] = {
                    'reviews': cluster_reviews,
                    'phrases': top_phrases
                }
            
            # Step 4: Compute Criticality Score per cluster
            print("Step 4: Computing criticality scores...")
            cluster_summary = []
            
            for cluster_id, data in cluster_data.items():
                reviews = data['reviews']
                review_count = len(reviews)
                percent_of_total = round((review_count / total_negative_reviews) * 100, 1)
                
                # Calculate averages
                avg_severity = sum(r['sentiment']['score'] for r in reviews) / review_count
                avg_votes = sum(r['thumbsUpCount'] for r in reviews) / review_count
                
                # Criticality Score = percent_of_total × avg_severity × avg_votes
                criticality_score = round(percent_of_total * avg_severity * avg_votes, 2)
                
                # Get representative reviews (top 2 by thumbsUpCount)
                representative_reviews = sorted(reviews, key=lambda x: x['thumbsUpCount'], reverse=True)[:2]
                
                cluster_info = {
                    'Cluster ID': cluster_id,
                    'Review Count': review_count,
                    'Percent of Total': percent_of_total,
                    'Avg Severity': round(avg_severity, 2),
                    'Avg Helpful Votes': round(avg_votes, 1),
                    'Criticality Score': criticality_score,
                    'Top Complaint Phrases': data['phrases'],
                    'Representative Reviews': [
                        {
                            'text': r['content'][:150] + '...' if len(r['content']) > 150 else r['content'],
                            'thumbsUpCount': r['thumbsUpCount']
                        }
                        for r in representative_reviews
                    ]
                }
                
                cluster_summary.append(cluster_info)
            
            # Step 5: Sort by Criticality Score descending
            cluster_summary.sort(key=lambda x: x['Criticality Score'], reverse=True)
            
            print(f"Step 5: Generated {len(cluster_summary)} cluster summaries")
            
            # Step 6: Return structured output
            return {
                'total_negative_reviews': total_negative_reviews,
                'cluster_summary': cluster_summary,
                'ui_simplification_notes': [
                    'Merge Critical Themes + Actual User Complaints into one section',
                    'Show Review Count of X / Y inside each cluster card',
                    'Highlight top 2 phrases in user-friendly sentence',
                    'Move representative reviews directly under each problem cluster',
                    'Remove standalone Most Voted Reviews section if shown per cluster',
                    'Add Impact fields (retention drop, revenue loss) if mapped later'
                ]
            }
            
        except Exception as e:
            print(f"Error in complaint cluster analysis: {e}")
            return {
                'total_negative_reviews': 0,
                'cluster_summary': [],
                'error': str(e)
            }

    def extract_cluster_complaint_phrases(self, negative_reviews: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract complaint phrases from clustered negative reviews with simplified output format.
        
        Args:
            negative_reviews: List of review dicts with 'content', 'sentiment', and 'cluster' keys
            
        Returns:
            Dict mapping cluster IDs to lists of phrase objects with format:
            {
                2: [
                    {"phrase": "paid but not credited", "frequency": 12, "avg_severity": 0.91},
                    {"phrase": "refund not received", "frequency": 9, "avg_severity": 0.89}
                ]
            }
        """
        try:
            logger.info(f"Extracting complaint phrases from {len(negative_reviews)} clustered negative reviews")
            
            # Group reviews by cluster
            clusters = defaultdict(list)
            for review in negative_reviews:
                cluster_id = review.get('cluster', 0)
                clusters[cluster_id].append(review)
            
            result = {}
            
            for cluster_id, cluster_reviews in clusters.items():
                if not cluster_reviews:
                    continue
                
                logger.info(f"Processing cluster {cluster_id} with {len(cluster_reviews)} reviews")
                
                # Step 1: Preprocess reviews - extract and clean content
                texts = []
                review_sentiments = []
                
                for review in cluster_reviews:
                    content = review.get('content', '')
                    if content:
                        # Lowercase and basic cleaning
                        cleaned_content = content.lower().strip()
                        texts.append(cleaned_content)
                        
                        # Extract sentiment score (severity)
                        sentiment = review.get('sentiment', {})
                        if isinstance(sentiment, dict):
                            sentiment_score = abs(sentiment.get('score', 0.5))
                        else:
                            sentiment_score = 0.5
                        review_sentiments.append(sentiment_score)
                
                if not texts:
                    result[int(cluster_id)] = []
                    continue
                
                # Step 2: Use CountVectorizer to extract n-gram phrases
                try:
                    vectorizer = CountVectorizer(
                        ngram_range=(2, 4),
                        stop_words='english',
                        lowercase=True,
                        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens
                    )
                    
                    # Fit and transform
                    count_matrix = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get phrase frequencies across all texts
                    phrase_frequencies = count_matrix.sum(axis=0).A1
                    
                    # Step 3: Calculate metrics for each phrase
                    phrase_metrics = []
                    
                    for idx, phrase in enumerate(feature_names):
                        frequency = int(phrase_frequencies[idx])
                        if frequency == 0:
                            continue
                        
                        # Find reviews containing this phrase and calculate average sentiment
                        phrase_sentiments = []
                        
                        for i, text in enumerate(texts):
                            if phrase in text:
                                phrase_sentiments.append(review_sentiments[i])
                        
                        if phrase_sentiments:
                            avg_severity = sum(phrase_sentiments) / len(phrase_sentiments)
                            
                            phrase_metrics.append({
                                "phrase": phrase,
                                "frequency": frequency,
                                "avg_severity": round(avg_severity, 2),
                                "combined_score": frequency * avg_severity
                            })
                    
                    # Step 4: Sort by (frequency × avg_severity) and get top 5
                    phrase_metrics.sort(key=lambda x: x['combined_score'], reverse=True)
                    top_phrases = phrase_metrics[:5]
                    
                    # Remove combined_score from final output
                    final_phrases = []
                    for phrase in top_phrases:
                        final_phrases.append({
                            "phrase": phrase["phrase"],
                            "frequency": phrase["frequency"],
                            "avg_severity": phrase["avg_severity"]
                        })
                    
                    result[int(cluster_id)] = final_phrases
                    
                except Exception as e:
                    logger.warning(f"CountVectorizer failed for cluster {cluster_id}: {e}")
                    # Fallback to simple extraction
                    fallback_phrases = self._simple_phrase_extraction_dict_format(texts, review_sentiments)
                    result[int(cluster_id)] = fallback_phrases[:5]
            
            logger.info(f"Successfully extracted phrases from {len(result)} clusters")
            return result
            
        except Exception as e:
            logger.error(f"Cluster complaint phrase extraction failed: {e}")
            return {}
    
    def _simple_phrase_extraction_dict_format(self, texts: List[str], sentiments: List[float]) -> List[Dict[str, Any]]:
        """Simple fallback phrase extraction returning dict format"""
        from collections import Counter
        import re
        
        # Simple n-gram extraction
        phrases = []
        for text in texts:
            # Remove punctuation and split into words
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            
            # Generate 2-4 grams
            for n in range(2, 5):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Calculate metrics
        phrase_metrics = []
        for phrase, frequency in phrase_counts.most_common(20):
            # Find matching reviews
            matching_sentiments = []
            for i, text in enumerate(texts):
                if phrase in text.lower():
                    matching_sentiments.append(sentiments[i])
            
            if matching_sentiments and frequency >= 1:
                avg_severity = sum(matching_sentiments) / len(matching_sentiments)
                combined_score = frequency * avg_severity
                
                phrase_metrics.append({
                    "phrase": phrase,
                    "frequency": frequency,
                    "avg_severity": round(avg_severity, 2),
                    "combined_score": combined_score
                })
        
        # Sort by combined score and return without combined_score
        phrase_metrics.sort(key=lambda x: x['combined_score'], reverse=True)
        return [{
            "phrase": p["phrase"],
            "frequency": p["frequency"],
            "avg_severity": p["avg_severity"]
        } for p in phrase_metrics]

    def analyze_negative_review_clusters(self, reviews):
        """
        Comprehensive clustering analysis for negative reviews with exact user specifications:
        - Calculate percentages of clustered reviews (not all negative reviews)
        - Extract top 3 complaint phrases using CountVectorizer (2-4 ngrams)
        - Sort phrases by frequency × avg severity
        - Identify most helpful complaint per cluster
        - Return coverage sentence and cluster summaries
        """
        try:
            logger.info("🟣 Step 1: Filter & Prepare negative reviews")
            
            # Filter only reviews where score <= 2 and sentiment.label == "NEGATIVE"
            negative_reviews = []
            for review in reviews:
                # Handle different rating field names
                score = review.get('score', review.get('rating', 5))
                sentiment = review.get('sentiment_analysis', {})
                sentiment_label = sentiment.get('final_sentiment', '').upper()
                
                if score <= 2 and sentiment_label.lower() == 'negative':
                    # Ensure thumbsUpCount is set
                    if 'thumbsUpCount' not in review or review['thumbsUpCount'] is None:
                        review['thumbsUpCount'] = 1
                    negative_reviews.append(review)
            
            total_negative_reviews = len(negative_reviews)
            logger.info(f"Filtered {total_negative_reviews} negative reviews")
            
            if total_negative_reviews < 3:
                return {
                    "total_negative_reviews": total_negative_reviews,
                    "total_clustered_reviews": 0,
                    "coverage_percentage": 0.0,
                    "coverage_sentence": f"Showing AI clusters from 0 of {total_negative_reviews} negative reviews (0% coverage)",
                    "cluster_summary": []
                }
            
            logger.info("🧠 Step 2: Cluster Reviews")
            
            # Prepare content for clustering
            review_contents = [review.get('review', '') for review in negative_reviews]
            
            try:
                # Try sentence transformers clustering first
                from sentence_transformers import SentenceTransformer
                from sklearn.cluster import AgglomerativeClustering
                import numpy as np
                
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode(review_contents)
                
                # Use AgglomerativeClustering with distance_threshold=1.0
                clustering = AgglomerativeClustering(
                    distance_threshold=1.0,
                    n_clusters=None,
                    linkage='ward'
                )
                cluster_labels = clustering.fit_predict(embeddings)
                
            except Exception as e:
                logger.warning(f"Sentence transformers clustering failed: {e}, using TF-IDF fallback")
                # Fallback to TF-IDF clustering
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.cluster import KMeans
                
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(review_contents)
                
                # Use fewer clusters for smaller datasets
                n_clusters = min(5, max(2, total_negative_reviews // 10))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Assign cluster_id to each review
            for i, review in enumerate(negative_reviews):
                review['cluster_id'] = int(cluster_labels[i])
            
            # Group reviews by cluster
            clusters = {}
            for review in negative_reviews:
                cluster_id = review['cluster_id']
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(review)
            
            logger.info(f"Created {len(clusters)} clusters")
            
            logger.info("🗣 Step 3: Extract Top Complaint Phrases per Cluster")
            
            cluster_summary = []
            total_clustered_reviews = len(negative_reviews)
            
            for cluster_id, cluster_reviews in clusters.items():
                if len(cluster_reviews) < 2:  # Skip tiny clusters
                    continue
                
                # Extract complaint phrases using CountVectorizer
                cluster_contents = [review.get('review', '') for review in cluster_reviews]
                
                try:
                    from sklearn.feature_extraction.text import CountVectorizer
                    
                    vectorizer = CountVectorizer(
                        ngram_range=(2, 4),
                        stop_words='english',
                        max_features=50,
                        min_df=1
                    )
                    
                    count_matrix = vectorizer.fit_transform(cluster_contents)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Calculate phrase frequencies and scores
                    phrase_scores = []
                    for i, phrase in enumerate(feature_names):
                        # Get frequency (number of reviews containing this phrase)
                        frequency = (count_matrix[:, i] > 0).sum()
                        
                        # Calculate average severity for reviews containing this phrase
                        phrase_severities = []
                        for j, review in enumerate(cluster_reviews):
                            if count_matrix[j, i] > 0:
                                severity = abs(review.get('sentiment_analysis', {}).get('confidence', 0.5))
                                phrase_severities.append(severity)
                        
                        avg_severity = np.mean(phrase_severities) if phrase_severities else 0.5
                        combined_score = frequency * avg_severity
                        
                        phrase_scores.append({
                            "phrase": phrase,
                            "frequency": int(frequency),
                            "avg_severity": round(float(avg_severity), 3),
                            "combined_score": round(float(combined_score), 3)
                        })
                    
                    # Sort by combined score and take top 3
                    top_phrases = sorted(phrase_scores, key=lambda x: x['combined_score'], reverse=True)[:3]
                    
                except Exception as e:
                    logger.warning(f"Phrase extraction failed for cluster {cluster_id}: {e}")
                    top_phrases = []
                
                logger.info("📊 Step 4: Compute Theme Metrics")
                
                # Calculate cluster metrics
                review_count = len(cluster_reviews)
                percent_of_clustered = (review_count / total_clustered_reviews * 100) if total_clustered_reviews > 0 else 0
                
                # Calculate average severity
                severities = []
                for review in cluster_reviews:
                    severity = abs(review.get('sentiment_analysis', {}).get('confidence', 0.5))
                    severities.append(severity)
                avg_severity = np.mean(severities) if severities else 0.5
                
                # Calculate helpful votes metrics
                helpful_votes = [review.get('thumbsUpCount', 1) for review in cluster_reviews]
                total_helpful_votes = sum(helpful_votes)
                avg_helpful_votes = np.mean(helpful_votes) if helpful_votes else 1.0
                
                # Calculate criticality score: Percent × Avg Severity × Avg Helpful Votes
                criticality_score = percent_of_clustered * avg_severity * avg_helpful_votes
                
                # Determine concern level
                if total_helpful_votes > (review_count * 1.5):
                    concern_level = "High"
                elif total_helpful_votes > (review_count * 1.0):
                    concern_level = "Medium"
                else:
                    concern_level = "Low"
                
                logger.info("⭐️ Step 5: Identify the Most Helpful Complaint")
                
                # Find most helpful complaint
                most_helpful_review = max(cluster_reviews, key=lambda x: x.get('thumbsUpCount', 1))
                most_helpful_complaint = {
                    "text": most_helpful_review.get('review', ''),
                    "thumbsUpCount": most_helpful_review.get('thumbsUpCount', 1),
                    "date": most_helpful_review.get('date', '2025-01-11')
                }
                
                # Generate cluster label based on top complaint phrases
                if top_phrases:
                    first_phrase = top_phrases[0]['phrase']
                    if any(word in first_phrase.lower() for word in ['crash', 'bug', 'error', 'freeze']):
                        cluster_label = "Technical Issues"
                    elif any(word in first_phrase.lower() for word in ['payment', 'billing', 'charge', 'money']):
                        cluster_label = "Payment & Billing"
                    elif any(word in first_phrase.lower() for word in ['login', 'account', 'password']):
                        cluster_label = "Login & Account"
                    elif any(word in first_phrase.lower() for word in ['slow', 'loading', 'performance']):
                        cluster_label = "Performance Issues"
                    else:
                        cluster_label = "General Complaints"
                else:
                    cluster_label = f"Cluster {cluster_id + 1}"
                
                # Representative reviews (top 3 by helpful votes)
                representative_reviews = sorted(cluster_reviews, key=lambda x: x.get('thumbsUpCount', 1), reverse=True)[:3]
                rep_reviews_formatted = []
                for review in representative_reviews:
                    rep_reviews_formatted.append({
                        "text": review.get('review', ''),
                        "thumbsUpCount": review.get('thumbsUpCount', 1)
                    })
                
                cluster_data = {
                    'Cluster ID': cluster_id,
                    'Cluster Label': cluster_label,
                    'Review Count': review_count,
                    'Percent of Clustered Reviews': round(percent_of_clustered, 1),
                    'Total Helpful Votes': total_helpful_votes,
                    'Avg Severity': round(avg_severity, 3),
                    'Avg Helpful Votes': round(avg_helpful_votes, 1),
                    'Criticality Score': round(criticality_score, 2),
                    'Concern Level': concern_level,
                    'Top Complaint Phrases': top_phrases,
                    'Most Helpful Complaint': most_helpful_complaint,
                    'Representative Reviews': rep_reviews_formatted
                }
                
                cluster_summary.append(cluster_data)
            
            logger.info("✅ Step 6: Return Final Output Per Cluster")
            
            # Sort clusters by criticality score (descending)
            cluster_summary.sort(key=lambda x: x['Criticality Score'], reverse=True)
            
            # Calculate final coverage statistics
            final_clustered_reviews = sum(cluster['Review Count'] for cluster in cluster_summary)
            coverage_percentage = (final_clustered_reviews / total_negative_reviews * 100) if total_negative_reviews > 0 else 0
            
            # Generate coverage sentence
            coverage_sentence = f"Showing AI clusters from {final_clustered_reviews} of {total_negative_reviews} negative reviews ({coverage_percentage:.1f}% coverage)"
            
            return {
                "total_negative_reviews": total_negative_reviews,
                "total_clustered_reviews": final_clustered_reviews,
                "coverage_percentage": round(coverage_percentage, 1),
                "coverage_sentence": coverage_sentence,
                "cluster_summary": cluster_summary
            }
            
        except Exception as e:
            logger.error(f"Error in negative review cluster analysis: {str(e)}")
            return {
                "total_negative_reviews": 0,
                "total_clustered_reviews": 0,
                "coverage_percentage": 0.0,
                "coverage_sentence": "Showing AI clusters from 0 of 0 negative reviews (0% coverage)",
                "cluster_summary": [],
                "error": str(e)
            }

    def analyze_critical_user_complaints(self, reviews):
        """
        Improved criticality analysis for negative reviews using semantic similarity clustering.
        
        Formula: criticality = (number of complaints) × (avg. severity from sentiment model) + (number of times reviews were marked helpful × 2)
        
        Uses semantic clustering to group similar complaints and generates human-readable theme names.
        """
        try:
            logger.info("🔴 Starting improved critical user complaints analysis...")
            
            # Step 1: Filter negative reviews (score ≤ 2 OR negative sentiment)
            negative_reviews = []
            for review in reviews:
                # Handle different rating field names
                score = review.get('score', review.get('rating', 5))
                sentiment = review.get('sentiment_analysis', {})
                sentiment_label = sentiment.get('final_sentiment', '').lower()
                
                # More flexible negative review detection
                is_negative = (
                    score <= 2 or  # Low rating
                    sentiment_label == 'negative' or  # Negative sentiment
                    (sentiment.get('vader_compound', 0) < -0.1) or  # VADER negative
                    (sentiment.get('textblob_polarity', 0) < -0.1)  # TextBlob negative
                )
                
                if is_negative:
                    # Ensure required fields are set with defaults
                    if 'thumbsUpCount' not in review or review['thumbsUpCount'] is None:
                        review['thumbsUpCount'] = 0  # Default to 0 for unhelpful
                    # Ensure content field exists
                    if 'content' not in review:
                        review['content'] = review.get('review', '')
                    # Ensure proper date format
                    if 'at' not in review:
                        review['at'] = review.get('date', 'Unknown')
                    negative_reviews.append(review)
            
            if len(negative_reviews) < 5:
                logger.info(f"Not enough negative reviews ({len(negative_reviews)}) for critical analysis")
                return {
                    "total_negative_reviews": len(negative_reviews),
                    "critical_issues": [],
                    "summary_table": [],
                    "message": "Insufficient negative reviews for critical analysis"
                }
            
            logger.info(f"Analyzing {len(negative_reviews)} negative reviews for critical issues")
            
            # Step 2: Use semantic clustering to group similar complaints
            try:
                # Try advanced clustering with sentence transformers
                if KEYBERT_AVAILABLE:
                    from sentence_transformers import SentenceTransformer
                    from sklearn.cluster import KMeans
                    from sklearn.metrics.pairwise import cosine_similarity
                    import numpy as np
                    
                    # Extract review texts
                    texts = [review.get('content', '') for review in negative_reviews]
                    
                    # Create embeddings
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    embeddings = model.encode(texts)
                    
                    # Determine optimal number of clusters (between 3-8)
                    n_clusters = min(max(3, len(negative_reviews) // 10), 8)
                    
                    # Perform clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    
                    # Group reviews by cluster
                    clusters = {}
                    for i, label in enumerate(cluster_labels):
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(negative_reviews[i])
                    
                    logger.info(f"Created {len(clusters)} semantic clusters")
                    
                else:
                    # Fallback to keyword-based clustering
                    clusters = self._fallback_keyword_clustering(negative_reviews)
                    
            except Exception as e:
                logger.warning(f"Semantic clustering failed, using fallback: {e}")
                clusters = self._fallback_keyword_clustering(negative_reviews)
            
            # Step 3: Generate human-readable theme names for each cluster
            theme_clusters = {}
            for cluster_id, cluster_reviews in clusters.items():
                if len(cluster_reviews) < 2:  # Skip clusters with too few reviews
                    continue
                    
                theme_name = self._generate_theme_name(cluster_reviews)
                theme_clusters[theme_name] = cluster_reviews
            
            # Step 4: Calculate criticality scores using the user's exact formula
            critical_issues = []
            summary_table = []
            
            for theme_name, theme_reviews in theme_clusters.items():
                # Number of complaints
                complaint_count = len(theme_reviews)
                
                # Calculate average severity from sentiment model
                severity_scores = []
                for review in theme_reviews:
                    sentiment = review.get('sentiment_analysis', {})
                    
                    # Use VADER compound score as primary severity indicator
                    if 'vader_compound' in sentiment:
                        compound = float(sentiment.get('vader_compound', -0.5))
                        # Convert VADER compound (-1 to 1) to severity (1 to 5 scale)
                        # More negative compound = higher severity
                        severity = 3 + (abs(compound) * 2)  # Range 1-5, negative values get higher severity
                    elif 'textblob_polarity' in sentiment:
                        polarity = float(sentiment.get('textblob_polarity', -0.3))
                        severity = 3 + (abs(polarity) * 2)  # Similar conversion for TextBlob
                    else:
                        # Use review rating as fallback severity indicator
                        rating = review.get('score', review.get('rating', 2))
                        severity = 6 - rating  # Rating 1 = severity 5, Rating 5 = severity 1
                    
                    severity_scores.append(max(1, min(5, severity)))  # Clamp to 1-5 range
                
                avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 3.5
                
                # Number of times reviews were marked helpful
                total_helpful_votes = sum(review.get('thumbsUpCount', 0) for review in theme_reviews)
                
                # Calculate Criticality Score using user's exact formula
                criticality_score = (complaint_count * avg_severity) + (total_helpful_votes * 2)
                
                # Calculate percentage of negative reviews
                percentage_of_negative = (complaint_count / len(negative_reviews)) * 100
                
                # Find most helpful review (highest thumbsUpCount)
                most_helpful_review = max(theme_reviews, key=lambda r: r.get('thumbsUpCount', 0))
                
                # Get top 2-3 representative complaints (mix of helpful and recent)
                representative_complaints = self._get_representative_complaints(theme_reviews)
                
                # Extract top problem phrases
                problem_phrases = self._extract_complaint_phrases(theme_reviews)
                
                # Determine criticality label based on score thresholds
                if criticality_score > 75:
                    criticality_label = "Critical"
                    criticality_tag = "🔴 Critical"
                elif criticality_score > 35:
                    criticality_label = "Major"
                    criticality_tag = "🟠 Major"
                else:
                    criticality_label = "Minor"
                    criticality_tag = "🟡 Minor"
                
                # Format most helpful review date
                formatted_date = self._format_review_date(most_helpful_review.get('at', 'Unknown'))
                
                # Create issue data
                issue_data = {
                    "theme": theme_name,
                    "complaint_count": complaint_count,
                    "criticality_score": round(criticality_score, 1),
                    "percentage_of_negative": round(percentage_of_negative, 1),
                    "criticality_label": criticality_label,
                    "criticality_tag": criticality_tag,
                    "avg_severity": round(avg_severity, 2),
                    "total_helpful_votes": total_helpful_votes,
                    "most_helpful_review": {
                        "text": most_helpful_review.get('content', '')[:300] + "..." if len(most_helpful_review.get('content', '')) > 300 else most_helpful_review.get('content', ''),
                        "thumbsUpCount": most_helpful_review.get('thumbsUpCount', 0),
                        "date": formatted_date,
                        "rating": most_helpful_review.get('score', most_helpful_review.get('rating', 1))
                    },
                    "representative_complaints": representative_complaints,
                    "top_problem_phrases": problem_phrases
                }
                
                critical_issues.append(issue_data)
                
                # Add to summary table
                summary_table.append({
                    "theme": theme_name,
                    "percentage_of_reviews": round(percentage_of_negative, 1),
                    "complaint_count": complaint_count,
                    "criticality_score": round(criticality_score, 1),
                    "label": criticality_label
                })
            
            # Step 5: Sort by criticality score (descending) and number them
            critical_issues.sort(key=lambda x: x['criticality_score'], reverse=True)
            summary_table.sort(key=lambda x: x['criticality_score'], reverse=True)
            
            # Add ranking numbers
            for i, issue in enumerate(critical_issues):
                issue['rank'] = i + 1
            
            logger.info(f"Identified {len(critical_issues)} critical complaint themes")
            
            return {
                "total_negative_reviews": len(negative_reviews),
                "critical_issues": critical_issues,  # Return ALL themes, not limited to 10
                "summary_table": summary_table,
                "analysis_method": "(number of complaints) × (avg. severity from sentiment model) + (number of times reviews were marked helpful × 2)",
                "clustering_method": "Semantic similarity clustering with human-readable theme names"
            }
            
        except Exception as e:
            logger.error(f"Error in critical complaints analysis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "total_negative_reviews": 0,
                "critical_issues": [],
                "summary_table": [],
                "error": str(e)
            }
    
    def _fallback_keyword_clustering(self, negative_reviews):
        """Fallback keyword-based clustering when semantic clustering is not available"""
        issue_themes = {
            "Payment & Billing Problems": [
                "payment", "billing", "charge", "charged", "refund", "money", "cost", 
                "price", "subscription", "cancel", "credit card", "paypal",
                "transaction", "purchase", "buy", "paid", "expensive", "free trial"
            ],
            "Login & Account Issues": [
                "login", "log in", "sign in", "signin", "account", "password", "username",
                "forgot password", "locked out", "can't access", "verification", "authenticate",
                "register", "signup"
            ],
            "App Crashes & Freezing": [
                "crash", "crashes", "freeze", "frozen", "hang", "stuck", "not responding",
                "closes", "shuts down", "stops working", "black screen"
            ],
            "Slow Performance": [
                "slow", "lag", "laggy", "loading", "performance", "speed", "responsive",
                "takes forever", "long time", "waiting"
            ],
            "Technical Bugs & Errors": [
                "bug", "error", "glitch", "broken", "not working", "doesn't work", 
                "malfunction", "issue", "problem", "fail", "failed", "wrong"
            ],
            "User Interface Problems": [
                "interface", "ui", "design", "layout", "confusing", "hard to use",
                "navigation", "menu", "button", "click", "tap", "screen", "display"
            ],
            "Missing Features": [
                "feature", "add", "need", "want", "wish", "should have", "missing",
                "improvement", "better", "enhance", "update", "would like"
            ],
            "Customer Support Issues": [
                "support", "help", "service", "customer", "response", "reply", "contact",
                "assistance", "staff", "team", "representative", "no response"
            ],
            "Scam & Fraud Reports": [
                "scam", "fraud", "fake", "spam", "suspicious", "cheat", "steal",
                "money back", "rip off", "dishonest", "misleading"
            ]
        }
        
        # Group reviews by themes
        theme_reviews = {theme: [] for theme in issue_themes}
        unmatched_reviews = []
        
        for review in negative_reviews:
            review_text = review.get('content', '').lower()
            matched = False
            
            for theme, keywords in issue_themes.items():
                if any(keyword in review_text for keyword in keywords):
                    theme_reviews[theme].append(review)
                    matched = True
                    break
            
            if not matched:
                unmatched_reviews.append(review)
        
        # Add unmatched reviews as "General Complaints"
        if unmatched_reviews:
            theme_reviews["General Complaints"] = unmatched_reviews
        
        # Convert to cluster format
        clusters = {}
        cluster_id = 0
        for theme, reviews in theme_reviews.items():
            if reviews:  # Only include non-empty themes
                clusters[cluster_id] = reviews
                cluster_id += 1
        
        return clusters
    
    def _generate_theme_name(self, cluster_reviews):
        """Generate a human-readable theme name based on cluster content"""
        try:
            # Extract common keywords and phrases
            texts = [review.get('content', '').lower() for review in cluster_reviews]
            combined_text = ' '.join(texts)
            
            # Common complaint patterns
            patterns = {
                "Payment Failures": ["payment", "pay", "charge", "billing", "refund", "money"],
                "Login Problems": ["login", "sign in", "account", "password", "access"],
                "App Crashes": ["crash", "freeze", "hang", "stop", "close", "shut"],
                "Slow Performance": ["slow", "lag", "loading", "speed", "fast"],
                "Scam Reports": ["scam", "fraud", "fake", "spam", "cheat", "steal"],
                "Technical Bugs": ["bug", "error", "glitch", "broken", "not work"],
                "UI/UX Issues": ["confusing", "hard", "difficult", "interface", "design"],
                "Missing Features": ["need", "want", "add", "missing", "should have"],
                "Customer Service": ["support", "help", "service", "response", "contact"],
                "Data Loss": ["lost", "delete", "missing", "gone", "disappear"],
                "Subscription Issues": ["subscription", "cancel", "trial", "auto", "renew"],
                "Update Problems": ["update", "version", "new", "change", "different"]
            }
            
            # Score each pattern
            pattern_scores = {}
            for pattern_name, keywords in patterns.items():
                score = sum(combined_text.count(keyword) for keyword in keywords)
                if score > 0:
                    pattern_scores[pattern_name] = score
            
            # Return the highest scoring pattern
            if pattern_scores:
                return max(pattern_scores.items(), key=lambda x: x[1])[0]
            else:
                return "General Complaints"
                
        except Exception as e:
            logger.warning(f"Could not generate theme name: {e}")
            return "General Complaints"
    
    def _get_representative_complaints(self, theme_reviews):
        """Get top 2-3 representative complaints with highlights"""
        try:
            # Sort by helpfulness and recency
            sorted_reviews = sorted(theme_reviews, key=lambda r: (
                r.get('thumbsUpCount', 0),
                len(r.get('content', ''))
            ), reverse=True)
            
            representative = []
            for review in sorted_reviews[:3]:
                # Extract key phrases from the review
                content = review.get('content', '')
                highlights = self._extract_highlights(content)
                
                representative.append({
                    "text": content[:200] + "..." if len(content) > 200 else content,
                    "rating": review.get('score', review.get('rating', 1)),
                    "thumbsUpCount": review.get('thumbsUpCount', 0),
                    "date": self._format_review_date(review.get('at', 'Unknown')),
                    "highlights": highlights[:3]  # Top 3 highlights
                })
            
            return representative
            
        except Exception as e:
            logger.warning(f"Could not get representative complaints: {e}")
            return []
    
    def _extract_complaint_phrases(self, theme_reviews):
        """Extract top complaint phrases from theme reviews"""
        try:
            if not SKLEARN_AVAILABLE:
                return []
                
            from sklearn.feature_extraction.text import CountVectorizer
            
            texts = [review.get('content', '') for review in theme_reviews]
            
            # Use CountVectorizer to extract 2-3 word phrases
            vectorizer = CountVectorizer(
                ngram_range=(2, 3),
                max_features=15,
                stop_words='english',
                lowercase=True,
                min_df=2  # Must appear in at least 2 reviews
            )
            
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get phrase frequencies
            phrase_frequencies = X.sum(axis=0).A1
            phrase_data = list(zip(feature_names, phrase_frequencies))
            
            # Sort by frequency and get top phrases
            top_phrases = sorted(phrase_data, key=lambda x: x[1], reverse=True)[:5]
            return [{"phrase": phrase, "frequency": int(freq)} for phrase, freq in top_phrases]
            
        except Exception as e:
            logger.warning(f"Could not extract complaint phrases: {e}")
            return []
    
    def _extract_highlights(self, text):
        """Extract key highlights from review text"""
        try:
            # Simple extraction of impactful phrases
            sentences = text.split('.')
            highlights = []
            
            # Look for sentences with strong negative words
            negative_indicators = [
                "terrible", "awful", "horrible", "worst", "hate", "broken", "useless",
                "doesn't work", "not working", "can't", "won't", "never", "always",
                "frustrated", "disappointed", "angry", "annoyed"
            ]
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and any(indicator in sentence.lower() for indicator in negative_indicators):
                    highlights.append(sentence)
                    if len(highlights) >= 3:
                        break
            
            return highlights
            
        except Exception as e:
            return []
    
    def _format_review_date(self, date_str):
        """Format review date consistently and fix year issues"""
        try:
            if date_str == 'Unknown' or not date_str:
                return 'Unknown'
            
            # Handle different date formats
            if isinstance(date_str, str):
                import re
                from datetime import datetime
                
                # Check if it's a timestamp or epoch time
                if date_str.isdigit():
                    timestamp = int(date_str)
                    if timestamp > 1000000000:  # Unix timestamp
                        dt = datetime.fromtimestamp(timestamp)
                        return dt.strftime('%B %d, %Y')
                
                # Try to parse ISO format
                if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    try:
                        dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                        return dt.strftime('%B %d, %Y')
                    except:
                        pass
                
                # Handle relative dates like "2 months ago"
                if 'ago' in date_str.lower():
                    current_year = datetime.now().year
                    if 'month' in date_str:
                        return f"Recent ({current_year})"
                    elif 'day' in date_str:
                        return f"Recent ({current_year})"
                    elif 'year' in date_str:
                        years_ago = re.findall(r'(\d+)', date_str)
                        if years_ago:
                            year = current_year - int(years_ago[0])
                            return f"Approximately {year}"
                
                # If already in readable format, check year
                if re.match(r'\w+ \d{1,2}, \d{4}', date_str):
                    # Extract year and validate
                    year_match = re.search(r'(\d{4})', date_str)
                    if year_match:
                        year = int(year_match.group(1))
                        current_year = datetime.now().year
                        if year > current_year:
                            # Fix future years
                            corrected_date = date_str.replace(str(year), str(current_year))
                            return corrected_date
                    return date_str
                
                # Default case - try to extract meaningful info
                return date_str
            
            return str(date_str)
            
        except Exception as e:
            logger.warning(f"Date formatting error: {e}")
            return 'Unknown'

# No global instance - will be created in main.py 