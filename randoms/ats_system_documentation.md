# ATS System Documentation

## Overview

The ATS (Applicant Tracking System) System is a comprehensive resume scoring and analysis tool written by Arreyan Hamid. This system evaluates resumes against job descriptions using sophisticated algorithms to provide detailed ATS compatibility scores.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Class Reference](#class-reference)
- [Scoring Components](#scoring-components)
- [Configuration](#configuration)
- [File Support](#file-support)
- [Output Format](#output-format)

## Features

- **Multi-format Support**: Handles both PDF and TXT files for resumes and job descriptions
- **Comprehensive Skill Database**: Extensive categorized skill sets including programming, AI/ML, cloud, data, and application domains
- **Smart Matching**: Intelligent keyword matching with semantic inference
- **Weighted Scoring**: Five-component scoring system with configurable weights
- **Domain Experience Analysis**: Penalty system for missing critical domain experience
- **Detailed Reporting**: JSON output with comprehensive breakdown and analysis

## Installation

### Dependencies

```bash
pip install pypdf
```

### Required Libraries

- `pypdf`: For PDF text extraction
- `os`: File system operations
- `re`: Regular expression matching
- `collections.defaultdict`: Data structure utilities
- `json`: JSON handling
- `typing`: Type hints

## Usage

### Basic Usage

```python
from ats_system import CleanATSScorer

# Initialize the scorer
ats_scorer = CleanATSScorer()

# Generate report from files
report = ats_scorer.generate_ats_report('resume.pdf', 'job_description.txt')

# Or generate report from text
resume_text = "Your resume content..."
jd_text = "Job description content..."
report = ats_scorer.generate_ats_report_from_text(resume_text, jd_text)

print(f"ATS Score: {report['ats_score']}/100")
```

### Command Line Usage

```bash
python ats_system.py
```

## Class Reference

### CleanATSScorer

The main class that handles all ATS scoring functionality.

#### Constructor

```python
CleanATSScorer()
```

Initializes the scorer with predefined weights and skill databases.

#### Key Methods

##### `extract_text_from_file(file_path: str) -> str`

Extracts text content from PDF or TXT files.

**Parameters:**
- `file_path`: Path to the file (.pdf or .txt)

**Returns:** Extracted text as string

**Raises:** `ValueError` for unsupported file types

##### `generate_ats_report(resume_path: str, jd_path: str) -> Dict`

Generates complete ATS report from file paths.

**Parameters:**
- `resume_path`: Path to resume file
- `jd_path`: Path to job description file

**Returns:** Comprehensive ATS report dictionary

##### `generate_ats_report_from_text(resume_text: str, jd_text: str) -> Dict`

Generates ATS report from text inputs.

**Parameters:**
- `resume_text`: Resume content as string
- `jd_text`: Job description content as string

**Returns:** Comprehensive ATS report dictionary

##### `extract_jd_skills(jd_text: str) -> Dict`

Extracts technical and soft skills from job description.

**Parameters:**
- `jd_text`: Job description text

**Returns:** Dictionary containing:
- `tech_skills`: List of technical skills found
- `soft_skills`: List of soft skills found
- `all_skills`: Combined list of all skills

##### `extract_resume_skills(resume_text: str) -> Dict`

Extracts technical and soft skills from resume.

**Parameters:**
- `resume_text`: Resume text

**Returns:** Dictionary with same structure as `extract_jd_skills`

## Scoring Components

The ATS scoring system uses five weighted components:

### 1. Keyword Match (30%)

- **Purpose**: Measures alignment between resume and job description skills
- **Algorithm**: Smart matching with semantic inference
- **Features**:
  - Direct keyword matching
  - Cloud platform equivalency
  - Semantic inference mapping
  - Multi-word phrase handling

### 2. Skills Relevance (25%)

- **Purpose**: Evaluates how relevant resume skills are to JD requirements
- **Algorithm**: Uses same logic as keyword matching
- **Calculation**: (Matched skills / Total JD skills) × 100

### 3. Experience & Achievements (25%)

- **Purpose**: Analyzes professional experience quality
- **Components**:
  - Action verbs count (60% weight)
  - Quantified achievements (40% weight)
- **Thresholds**:
  - Action verbs: 12 for maximum score
  - Metrics: 8 quantified achievements for maximum score

### 4. Formatting Compliance (10%)

- **Purpose**: Ensures ATS-friendly formatting
- **Checks**:
  - Email address presence (25 points)
  - Phone number presence (25 points)
  - Essential sections: Experience, Education, Skills (50 points)

### 5. Extra Sections (10%)

- **Purpose**: Rewards additional valuable sections
- **Sections Detected**:
  - Certifications
  - Projects
  - Summary/Profile
  - Awards
  - Publications
  - Volunteering

## Configuration

### Scoring Weights

```python
self.weights = {
    'keyword_match': 30,        # 30% - JD extraction + alignment
    'skills_relevance': 25,     # 25% - Skills relevance to role
    'experience_achievements': 25,  # 25% - Action verbs + metrics
    'formatting_compliance': 10,    # 10% - ATS-friendly formatting
    'extra_sections': 10        # 10% - Certifications, projects, awards
}
```

### Skill Categories

The system includes comprehensive skill databases across multiple categories:

- **Programming**: Python, JavaScript, Java, C++, etc.
- **AI/ML**: Machine Learning, Deep Learning, TensorFlow, PyTorch, etc.
- **Data**: SQL, Pandas, NumPy, Data Analysis, etc.
- **Cloud**: AWS, Azure, Google Cloud, Docker, Kubernetes, etc.
- **Web**: FastAPI, Flask, Django, React, Node.js, etc.
- **Tools**: Git, GitHub, JIRA, VS Code, etc.
- **Application Domains**: 150+ specific domain skills across industries

### Domain Experience Penalty

The system applies penalties for missing critical domain experience:

- **No domain experience**: 15% penalty
- **No matching domains**: 15% penalty
- **<50% domain match**: 10% penalty
- **Partial domain match**: 5% penalty
- **Perfect domain match**: No penalty

## File Support

### Supported Formats

- **PDF**: `.pdf` files using pypdf library
- **Text**: `.txt` files with UTF-8 encoding

### Phone Number Formats

The system recognizes various phone number formats:

- Indian format: `098218 70330`, `09821870330`
- International: `+91-9821870330`, `+91 9821870330`
- US format: `123-456-7890`, `(123) 456-7890`
- Simple 10-digit: `9821870330`

## Output Format

### Main Report Structure

```json
{
  "ats_score": 85.7,
  "breakdown": {
    "keyword_match": "25.2/30",
    "skills_relevance": "21.3/25",
    "experience_achievements": "18.5/25",
    "formatting_compliance": "9.0/10",
    "extra_sections": "8.0/10"
  },
  "domain_penalty": "0.95",
  "jd_skills": {
    "tech_skills": ["python", "machine learning", "aws"],
    "soft_skills": ["communication", "teamwork"],
    "total_skills": 25
  },
  "skill_matching": {
    "total_jd_skills": 25,
    "matched_skills": ["python", "sql", "git"],
    "missing_skills": ["kubernetes", "docker"],
    "match_percentage": 84.0
  },
  "formatting_check": {
    "email_present": true,
    "phone_present": true,
    "essential_sections_found": 3,
    "issues": []
  },
  "experience_metrics": {
    "action_verbs_count": 8,
    "quantified_achievements": 5
  },
  "extra_sections": {
    "sections_found": ["projects", "certifications"],
    "count": 2
  }
}
```

### Generated Files

When running the main function, the system generates:

1. **`ats_report.json`**: Complete detailed report
2. **`jd_keywords.json`**: Simplified JD keywords extraction

## Advanced Features

### Semantic Inference

The system includes intelligent inference mappings:

```python
inference_map = {
    'version control': ['git', 'github', 'gitlab', 'svn'],
    'cloud platforms': ['aws', 'azure', 'google cloud', 'gcp'],
    'machine learning': ['tensorflow', 'pytorch', 'scikit-learn'],
    'ai orchestration': ['langchain', 'crewai', 'workflow orchestration']
}
```

### Regex Patterns for Achievements

The system detects quantified achievements using patterns:

- Percentages: `\d+%`
- Dollar amounts: `\$\d+`
- Large numbers: `\d+\s*(?:million|thousand|k|m)`
- Time metrics: `\d+\s*(?:hours?|days?|weeks?)`
- Improvement metrics: `increased.*?by.*?\d+`

### Action Verbs Database

36 professional action verbs are tracked including:
- `achieved`, `analyzed`, `built`, `created`, `designed`
- `developed`, `engineered`, `implemented`, `improved`
- `optimized`, `reduced`, `streamlined`, `transformed`

## Error Handling

The system includes comprehensive error handling:

- File format validation
- PDF reading error handling
- Text encoding issues
- Missing file handling

## Performance Considerations

- **Realistic Scoring**: Maximum score capped at 95%
- **Stringent Requirements**: Higher thresholds for maximum component scores
- **Domain Penalties**: Penalizes missing critical domain experience
- **Flexible Matching**: Handles single words and multi-word phrases

## Future Enhancements

Potential areas for improvement:
- Machine learning-based skill extraction
- Industry-specific scoring weights
- Resume formatting analysis
- Integration with external APIs
- Batch processing capabilities

---

*Documentation generated for ATS System v1.0 by Arreyan Hamid*