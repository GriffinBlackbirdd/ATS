# ATS Resume Analyzer API

[![CI/CD Pipeline](https://github.com/yourusername/ats-analyzer/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/ats-analyzer/actions)
[![Coverage Status](https://codecov.io/gh/yourusername/ats-analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/ats-analyzer)
[![Docker Image](https://ghcr.io/yourusername/ats-analyzer/badge.svg)](https://ghcr.io/yourusername/ats-analyzer)

A FastAPI-based web service that analyzes resume compatibility with job descriptions using ATS (Applicant Tracking System) scoring algorithms.

## 🚀 Features

- **PDF Upload**: Upload resume and job description PDFs
- **ATS Scoring**: Comprehensive scoring based on:
  - Required skills matching (40%)
  - Preferred skills matching (20%) 
  - Keyword coverage (20%)
  - Contact information and formatting (10%)
  - Semantic similarity (10%)
- **Detailed Analysis**: Skill gap analysis, missing skills, recommendations
- **RESTful API**: Easy integration with web frontends
- **Interactive Documentation**: Auto-generated OpenAPI docs

## 🚀 Quick Start

### Using Docker Compose (Recommended)
```bash
# Clone and run
git clone https://github.com/yourusername/ats-analyzer.git
cd ats-analyzer
docker-compose up --build

# Test the API
curl http://localhost:8000/health
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python atsServer.py
# Or: uvicorn atsServer:app --reload
```

## 📚 API Usage

### Full Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "resume=@resume.pdf" \
  -F "job_description=@jd.pdf"
```

### Quick Analysis
```bash
curl -X POST "http://localhost:8000/analyze-quick" \
  -F "resume=@resume.pdf" \
  -F "job_description=@jd.pdf"
```

## 📊 Response Format

```json
{
  "status": "success",
  "results": {
    "ats_score": 85.5,
    "component_scores": {
      "required_skills": "34.0/40",
      "preferred_skills": "18.0/20",
      "keyword_match": "16.0/20",
      "contact_formatting": "8.5/10",
      "semantic_similarity": "9.0/10"
    },
    "detailed_analysis": {
      "strong_matches": ["python", "machine learning"],
      "missing_required": ["aws", "docker"],
      "recommendations": ["Add AWS experience", "Include Docker skills"]
    }
  }
}
```

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=ats_system --cov=atsServer --cov-report=html

# Specific tests
pytest test_ats_system.py -v
```

## 🚀 Deployment

### GitHub Actions CI/CD Pipeline
- ✅ Automated testing across Python 3.9-3.11
- ✅ Code quality checks (Black, flake8, mypy)
- ✅ Security scanning (Bandit, Safety)
- ✅ Docker image building and publishing
- ✅ Automated deployment

### Docker Deployment
```bash
# Build and run
docker build -t ats-analyzer .
docker run -p 8000:8000 ats-analyzer

# Or use pre-built image
docker pull ghcr.io/yourusername/ats-analyzer:latest
```

## 🛠 Development

### Code Quality
```bash
black *.py          # Format code
isort *.py          # Sort imports  
flake8 *.py         # Lint code
mypy *.py           # Type check
bandit -r .         # Security scan
```

Visit `http://localhost:8000/docs` for interactive API documentation!