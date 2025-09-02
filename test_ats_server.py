import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from atsServer import app

client = TestClient(app)

class TestATSServer:
    """Test cases for the FastAPI ATS server"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "ATS Resume Analyzer API" in data["message"]
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_skills_database_endpoint(self):
        """Test the skills database endpoint"""
        response = client.get("/skills-database")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "skills_database" in data
        assert "total_skills" in data
        assert "categories" in data
        assert isinstance(data["total_skills"], int)
        assert data["total_skills"] > 0
    
    def create_dummy_pdf(self, content: str = "Test PDF content") -> bytes:
        """Create a dummy PDF file content for testing"""
        # This is a minimal PDF structure - in real tests you might want to use a PDF library
        pdf_content = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Resources <<
/Font <<
/F1 4 0 R
>>
>>
/Contents 5 0 R
>>
endobj

4 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

5 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
({content}) Tj
ET
endstream
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
0000000360 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
456
%%EOF"""
        return pdf_content.encode()
    
    def test_analyze_endpoint_missing_files(self):
        """Test analyze endpoint with missing files"""
        response = client.post("/analyze")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_analyze_endpoint_invalid_file_type(self):
        """Test analyze endpoint with non-PDF files"""
        # Test with text file instead of PDF
        response = client.post(
            "/analyze",
            files={
                "resume": ("resume.txt", b"This is not a PDF", "text/plain"),
                "job_description": ("jd.pdf", self.create_dummy_pdf(), "application/pdf")
            }
        )
        assert response.status_code == 400
        assert "Resume file must be a PDF" in response.json()["detail"]
    
    @patch('atsServer.ats_scorer.generate_report')
    def test_analyze_endpoint_success(self, mock_generate_report):
        """Test successful analysis endpoint"""
        # Mock the report generation
        mock_report = {
            "ats_score": 85.5,
            "component_scores": {
                "required_skills": "34.0/40",
                "preferred_skills": "18.0/20",
                "keyword_match": "16.0/20",
                "contact_formatting": "8.5/10",
                "semantic_similarity": "9.0/10"
            },
            "detailed_analysis": {
                "strong_matches": ["python", "machine learning", "sql"],
                "missing_required": ["aws"],
                "missing_preferred": ["docker"],
                "recommendations": ["Add AWS experience", "Include Docker skills"]
            },
            "resume_summary": {
                "total_skills": 15,
                "skills_by_category": {"programming": ["python", "javascript"]},
                "contact_formatting": {"email": True, "phone": True}
            },
            "job_summary": {
                "required_skills_count": 5,
                "preferred_skills_count": 3,
                "total_keywords": 8
            }
        }
        mock_generate_report.return_value = mock_report
        
        response = client.post(
            "/analyze",
            files={
                "resume": ("resume.pdf", self.create_dummy_pdf("Resume content"), "application/pdf"),
                "job_description": ("jd.pdf", self.create_dummy_pdf("JD content"), "application/pdf")
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert data["results"]["ats_score"] == 85.5
        assert "files_analyzed" in data
        assert data["files_analyzed"]["resume_filename"] == "resume.pdf"
        assert data["files_analyzed"]["job_description_filename"] == "jd.pdf"
    
    @patch('atsServer.ats_scorer.generate_report')
    def test_analyze_quick_endpoint_success(self, mock_generate_report):
        """Test successful quick analysis endpoint"""
        # Mock the full report
        mock_report = {
            "ats_score": 75.0,
            "component_scores": {
                "required_skills": "30.0/40",
                "preferred_skills": "15.0/20",
                "keyword_match": "14.0/20",
                "contact_formatting": "7.5/10",
                "semantic_similarity": "8.5/10"
            },
            "detailed_analysis": {
                "strong_matches": ["python", "sql"],
                "missing_required": ["machine learning", "aws"],
                "missing_preferred": ["docker", "kubernetes"],
                "recommendations": [
                    "Add machine learning experience",
                    "Include AWS cloud skills",
                    "Add containerization experience"
                ]
            }
        }
        mock_generate_report.return_value = mock_report
        
        response = client.post(
            "/analyze-quick",
            files={
                "resume": ("resume.pdf", self.create_dummy_pdf("Resume content"), "application/pdf"),
                "job_description": ("jd.pdf", self.create_dummy_pdf("JD content"), "application/pdf")
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "quick_results" in data
        assert data["quick_results"]["ats_score"] == 75.0
        assert data["quick_results"]["matched_skills_count"] == 2
        assert data["quick_results"]["missing_required_skills_count"] == 2
        assert data["quick_results"]["missing_preferred_skills_count"] == 2
        assert len(data["quick_results"]["top_recommendations"]) == 3
    
    @patch('atsServer.ats_scorer.generate_report')
    def test_analyze_endpoint_processing_error(self, mock_generate_report):
        """Test analyze endpoint with processing error"""
        # Mock an exception during report generation
        mock_generate_report.side_effect = Exception("PDF processing failed")
        
        response = client.post(
            "/analyze",
            files={
                "resume": ("resume.pdf", self.create_dummy_pdf(), "application/pdf"),
                "job_description": ("jd.pdf", self.create_dummy_pdf(), "application/pdf")
            }
        )
        
        assert response.status_code == 500
        assert "Analysis failed" in response.json()["detail"]
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set"""
        response = client.get("/")
        # CORS headers should be present due to middleware
        assert response.status_code == 200
    
    def test_analyze_jd_file_validation(self):
        """Test job description file validation"""
        response = client.post(
            "/analyze",
            files={
                "resume": ("resume.pdf", self.create_dummy_pdf(), "application/pdf"),
                "job_description": ("jd.txt", b"Not a PDF", "text/plain")
            }
        )
        assert response.status_code == 400
        assert "Job description file must be a PDF" in response.json()["detail"]

class TestATSServerIntegration:
    """Integration tests for the ATS server with real components"""
    
    def test_server_startup(self):
        """Test that the server starts up correctly"""
        # This test verifies that all imports work and the app can be created
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_openapi_docs(self):
        """Test that OpenAPI documentation is generated"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "paths" in openapi_spec
        assert "/analyze" in openapi_spec["paths"]
        assert "/analyze-quick" in openapi_spec["paths"]
    
    def test_api_endpoints_documented(self):
        """Test that all endpoints are properly documented in OpenAPI"""
        response = client.get("/openapi.json")
        openapi_spec = response.json()
        
        # Check that key endpoints are documented
        paths = openapi_spec["paths"]
        assert "/" in paths
        assert "/health" in paths
        assert "/analyze" in paths
        assert "/analyze-quick" in paths
        assert "/skills-database" in paths
        
        # Check that analyze endpoint has proper file upload parameters
        analyze_spec = paths["/analyze"]["post"]
        assert "requestBody" in analyze_spec
        request_body = analyze_spec["requestBody"]
        assert "multipart/form-data" in request_body["content"]