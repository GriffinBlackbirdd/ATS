# <¯ ATS Resume Analyzer

AI-powered ATS compatibility analysis for resumes and job descriptions. This FastAPI-based service helps job seekers optimize their resumes for Applicant Tracking Systems.

[![CI/CD Pipeline](https://github.com/GriffinBlackbirdd/ATS/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/GriffinBlackbirdd/ATS/actions/workflows/ci-cd.yml)

## =€ Quick Start

### Using Docker (Recommended)

```bash
# Pull the latest image
docker pull ghcr.io/griffinblackbirdd/ats:latest

# Run the container
docker run -p 8000:8000 ghcr.io/griffinblackbirdd/ats:latest
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/GriffinBlackbirdd/ATS.git
cd ATS

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m uvicorn atsServer:app --host 0.0.0.0 --port 8000
```

## =Ë API Endpoints

### Health Check
```http
GET /health
```

Returns the service health status.

### Analyze Resume
```http
POST /analyze
Content-Type: multipart/form-data
```

**Parameters:**
- `resume_file`: PDF file of the resume
- `jd_file`: PDF file of the job description

**Response:**
```json
{
  "ats_score": 85.2,
  "component_scores": {
    "required_skills": "34.0/40",
    "preferred_skills": "16.0/20",
    "keyword_match": "18.5/20",
    "contact_formatting": "8.5/10",
    "semantic_similarity": "8.5/10"
  },
  "detailed_analysis": {
    "strong_matches": ["python", "machine learning", "aws"],
    "missing_required": ["docker", "kubernetes"],
    "missing_preferred": ["react"],
    "recommendations": [
      "Add missing required skills: docker, kubernetes",
      "Consider adding preferred skills: react"
    ]
  }
}
```

## =à Features

- **PDF Text Extraction**: Supports both resume and job description PDFs
- **Skills Matching**: Advanced semantic analysis for skill matching
- **Contact Validation**: Checks for email, phone, and LinkedIn presence
- **Keyword Coverage**: Analyzes keyword alignment with job requirements
- **Detailed Scoring**: Comprehensive breakdown of ATS compatibility
- **Smart Recommendations**: Actionable suggestions for improvement

## =' Technical Stack

- **Backend**: FastAPI (Python)
- **PDF Processing**: PyPDF2
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Registry**: GitHub Container Registry (GHCR)
- **Testing**: Pytest with coverage reporting

## =Ê Scoring Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Required Skills | 40% | Match rate for essential job requirements |
| Preferred Skills | 20% | Match rate for nice-to-have skills |
| Keyword Match | 20% | Overall keyword coverage from job description |
| Contact Formatting | 10% | Presence of contact info and resume structure |
| Semantic Similarity | 10% | AI-powered content relevance analysis |

## =€ Deployment

### Deploy to Render

1. Fork this repository
2. Connect your GitHub account to Render
3. Create a new Web Service
4. Select this repository
5. Use the following settings:
   - **Environment**: Docker
   - **Docker Command**: (leave blank, uses Dockerfile CMD)
   - **Port**: 8000

Or use the `render.yaml` file for one-click deployment.

### Deploy with Docker

```bash
# Build the image
docker build -t ats-analyzer .

# Run locally
docker run -p 8000:8000 ats-analyzer

# Or use the pre-built image
docker run -p 8000:8000 ghcr.io/griffinblackbirdd/ats:latest
```

## >ê Testing

```bash
# Run all tests
pytest test_ats_system.py -v --cov=ats_system --cov-report=html

# Run API tests
pytest test_ats_server.py -v

# Test the deployed API
python test_api.py
```

## =È CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline that:

-  Runs unit tests on Python 3.9, 3.10, and 3.11
-  Performs code quality checks (Black, isort, flake8)
-  Runs security scans (Bandit, Safety)
-  Builds and pushes Docker images
-  Deploys documentation to GitHub Pages

## =Ö Documentation

- **API Docs**: Available at `/docs` when running the service
- **GitHub Pages**: [https://griffinblackbirdd.github.io/ATS/](https://griffinblackbirdd.github.io/ATS/)
- **Interactive API**: Available at `/redoc` when running the service

## > Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## =Ä License

This project is open source and available under the [MIT License](LICENSE).

## = Links

- **Repository**: [https://github.com/GriffinBlackbirdd/ATS](https://github.com/GriffinBlackbirdd/ATS)
- **Docker Image**: `ghcr.io/griffinblackbirdd/ats:latest`
- **Documentation**: [https://griffinblackbirdd.github.io/ATS/](https://griffinblackbirdd.github.io/ATS/)

---

Built with d using FastAPI and Docker