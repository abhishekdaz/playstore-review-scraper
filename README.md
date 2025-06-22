# Play Store Review Scraper

AI-powered Play Store review analysis tool with generic categorization system for comprehensive sentiment analysis and issue classification.

## Features

- **Generic Categorization System**: Configurable for different app types (social media, e-commerce, gaming, productivity)
- **Custom Category Configuration**: Create and configure custom categories via API endpoints
- **Theme-based Review Segmentation**: Advanced review categorization with sentiment analysis
- **FastAPI Backend**: High-performance API with comprehensive error handling
- **Next.js Frontend**: Modern React frontend with TypeScript and Tailwind CSS
- **Modular Analysis Engine**: Multiple sentiment analyzers with performance optimization
- **Real-time Processing**: 2,107+ reviews/second processing capability
- **Comprehensive Testing**: 100% test pass rate with integration testing

## Architecture

### Backend (FastAPI)
- **Analysis Engine**: Modular NLP processing with VADER, TextBlob, and optional Twitter-RoBERTa
- **Category System**: Dynamic configuration for different app types
- **API Endpoints**: RESTful API with automatic documentation
- **Performance**: Optimized for high-throughput review processing

### Frontend (Next.js)
- **Modern UI**: React with TypeScript and Tailwind CSS
- **Real-time Updates**: Live analysis results and progress tracking
- **Responsive Design**: Mobile-friendly interface
- **Error Handling**: Comprehensive error boundaries and fallbacks

## Quick Start

### Backend Setup
```bash
cd "Playstore Review Scraper"
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd playstore-reviews-frontend
npm install
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Search for apps on Play Store |
| `/reviews` | POST | Fetch reviews for a specific app |
| `/analyze` | POST | Analyze reviews with current configuration |
| `/analyze/direct` | POST | Direct analysis with custom parameters |
| `/configure_categories` | POST | Configure categories for app types |
| `/available_app_types` | GET | Get available app type configurations |
| `/segment_reviews` | POST | Segment reviews by themes |

## Supported App Types

### Built-in Configurations
1. **Social Media**: Privacy concerns, content moderation
2. **E-commerce**: Payment issues, delivery problems
3. **Gaming**: Gameplay mechanics, monetization
4. **Productivity**: Data sync, collaboration features

### Custom Categories
Create domain-specific categories for:
- Medical apps
- Financial services
- Educational tools
- Entertainment platforms

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **NLTK**: Natural language processing
- **TextBlob**: Sentiment analysis
- **Scikit-learn**: Machine learning utilities
- **VADER**: Sentiment intensity analyzer

### Frontend
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/ui**: Modern UI components

## Performance Metrics

- **Processing Speed**: 2,107+ reviews/second
- **Memory Efficiency**: 35KB response for 50 reviews
- **Test Coverage**: 7/7 integration tests passing
- **Analysis Accuracy**: Multi-model sentiment analysis with confidence scores

## Development

### Running Tests
```bash
# Backend integration tests
python test_system_integration.py

# Theme segmentation tests
python test_theme_segmentation.py

# Generic categorization tests
python test_generic_system.py
```

### Project Structure
```
├── analysis_engine.py          # Core NLP analysis engine
├── main.py                     # FastAPI application
├── requirements.txt            # Python dependencies
├── playstore-reviews-frontend/ # Next.js frontend
│   ├── src/app/               # App Router pages
│   ├── src/components/        # React components
│   └── package.json           # Node.js dependencies
└── README.md                  # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Built with modern AI/ML techniques for sentiment analysis
- Utilizes Google Play Store data for real-world testing
- Optimized for production-scale review processing
