import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ats_system import ATSScorer


class TestATSScorer:
    """Test cases for the ATS Scorer system"""

    @pytest.fixture
    def ats_scorer(self):
        """Create an ATS scorer instance for testing"""
        return ATSScorer()

    def test_ats_scorer_initialization(self, ats_scorer):
        """Test that ATS scorer initializes with correct weights and skills"""
        assert ats_scorer.weights["required_skills"] == 40
        assert ats_scorer.weights["preferred_skills"] == 20
        assert ats_scorer.weights["keyword_match"] == 20
        assert ats_scorer.weights["contact_formatting"] == 10
        assert ats_scorer.weights["semantic_similarity"] == 10

        # Check that skill categories exist
        assert "programming" in ats_scorer.tech_skills
        assert "ai_ml" in ats_scorer.tech_skills
        assert "data" in ats_scorer.tech_skills
        assert "cloud" in ats_scorer.tech_skills
        assert "web" in ats_scorer.tech_skills
        assert "tools" in ats_scorer.tech_skills

    def test_get_context(self, ats_scorer):
        """Test context extraction around matches"""
        text = "This is a sample text with Python programming language in the middle of the sentence."

        # Create a mock match object
        mock_match = MagicMock()
        mock_match.start.return_value = 30  # Position of "Python"
        mock_match.end.return_value = 36

        context = ats_scorer._get_context(text, mock_match)
        assert "Python" in context
        assert len(context) <= 60  # Should be around 60 chars (30 before + 30 after)

    def test_extract_section(self, ats_scorer):
        """Test section extraction from text"""
        sample_text = """
        Job Title: Software Engineer
        
        Requirements:
        - Python programming
        - Machine learning experience
        - 3+ years experience
        
        Preferred:
        - AWS knowledge
        - Docker experience
        
        Responsibilities:
        - Build applications
        - Work with team
        """

        required_section = ats_scorer._extract_section(
            sample_text, ["requirements", "required"]
        )
        preferred_section = ats_scorer._extract_section(sample_text, ["preferred"])

        assert "Python programming" in required_section
        assert "Machine learning" in required_section
        assert "AWS knowledge" in preferred_section
        assert "Docker experience" in preferred_section

    def test_analyze_contact_and_formatting(self, ats_scorer):
        """Test contact information and formatting analysis"""
        resume_text_with_contact = """
        John Doe
        john.doe@email.com
        +1-555-123-4567
        LinkedIn: linkedin.com/in/johndoe
        
        Experience:
        Software Engineer at Tech Corp
        
        Education:
        BS Computer Science
        
        Skills:
        Python, JavaScript, React
        
        Projects:
        Built web applications
        """

        analysis = ats_scorer._analyze_contact_and_formatting(resume_text_with_contact)

        assert analysis["email"] == True
        assert analysis["phone"] == True
        assert analysis["linkedin"] == True
        assert "experience" in analysis["sections_present"]
        assert "education" in analysis["sections_present"]
        assert "skills" in analysis["sections_present"]
        assert analysis["formatting_score"] > 0

    def test_analyze_contact_missing_info(self, ats_scorer):
        """Test contact analysis with missing information"""
        resume_text_minimal = """
        John Doe
        
        Experience:
        Software Engineer
        """

        analysis = ats_scorer._analyze_contact_and_formatting(resume_text_minimal)

        assert analysis["email"] == False
        assert analysis["phone"] == False
        assert analysis["linkedin"] == False
        assert "experience" in analysis["sections_present"]
        assert "education" not in analysis["sections_present"]
        assert analysis["formatting_score"] < 100

    def test_indian_phone_number_detection(self, ats_scorer):
        """Test Indian phone number detection specifically"""
        resume_with_indian_phone = """
        Arreyan Hamid
        arreyan@email.com
        098218 70330
        
        Experience:
        Software Engineer
        """

        analysis = ats_scorer._analyze_contact_and_formatting(resume_with_indian_phone)
        assert analysis["phone"] == True

    @patch("ats_system.pypdf.PdfReader")
    def test_extract_text_from_pdf_success(self, mock_pdf_reader, ats_scorer):
        """Test successful PDF text extraction"""
        # Mock PDF reader
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF content"

        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader_instance

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            result = ats_scorer.extract_text_from_pdf(temp_path)
            assert result == "Sample PDF content"
        finally:
            os.unlink(temp_path)

    def test_extract_text_from_pdf_file_not_found(self, ats_scorer):
        """Test PDF extraction with non-existent file"""
        with pytest.raises(Exception) as exc_info:
            ats_scorer.extract_text_from_pdf("/nonexistent/path/file.pdf")

        assert "Error reading PDF" in str(exc_info.value)

    def test_calculate_skills_match_no_required(self, ats_scorer):
        """Test skills matching when no required skills"""
        resume_skills = ["python", "javascript"]
        required_skills = []

        result = ats_scorer._calculate_skills_match(resume_skills, required_skills)
        assert result == 100

    def test_calculate_skills_match_direct_match(self, ats_scorer):
        """Test direct skills matching"""
        # Mock the original JD text for context analysis
        ats_scorer._original_jd_text = "We need Python and JavaScript experience"

        resume_skills = ["python", "javascript", "react"]
        required_skills = ["python", "javascript"]

        result = ats_scorer._calculate_skills_match(resume_skills, required_skills)
        assert result == 100.0

    def test_calculate_keyword_coverage(self, ats_scorer):
        """Test keyword coverage calculation"""
        resume_skills = ["python", "javascript", "react", "node.js"]
        jd_skills = ["python", "javascript", "angular"]

        coverage = ats_scorer._calculate_keyword_coverage(resume_skills, jd_skills)
        # 2 matches out of 3 JD skills = 66.67%
        assert abs(coverage - 66.67) < 0.1

    def test_calculate_keyword_coverage_no_jd_skills(self, ats_scorer):
        """Test keyword coverage with no JD skills"""
        resume_skills = ["python", "javascript"]
        jd_skills = []

        coverage = ats_scorer._calculate_keyword_coverage(resume_skills, jd_skills)
        assert coverage == 100

    def test_skills_database_completeness(self, ats_scorer):
        """Test that skills database contains expected skills"""
        # Check programming languages
        assert "python" in ats_scorer.tech_skills["programming"]
        assert "javascript" in ats_scorer.tech_skills["programming"]
        assert "java" in ats_scorer.tech_skills["programming"]

        # Check AI/ML skills
        assert "machine learning" in ats_scorer.tech_skills["ai_ml"]
        assert "tensorflow" in ats_scorer.tech_skills["ai_ml"]
        assert "pytorch" in ats_scorer.tech_skills["ai_ml"]

        # Check cloud skills
        assert "aws" in ats_scorer.tech_skills["cloud"]
        assert "azure" in ats_scorer.tech_skills["cloud"]
        assert "docker" in ats_scorer.tech_skills["cloud"]

    def test_weights_sum_to_100(self, ats_scorer):
        """Test that all weights sum to 100%"""
        total_weight = sum(ats_scorer.weights.values())
        assert total_weight == 100


class TestATSIntegration:
    """Integration tests for the ATS system"""

    @pytest.fixture
    def ats_scorer(self):
        return ATSScorer()

    def test_parse_job_requirements_basic(self, ats_scorer):
        """Test basic job requirements parsing"""
        jd_text = """
        Software Engineer Position
        
        Required Skills:
        - Python programming
        - Machine learning experience
        - SQL databases
        
        Preferred Skills:
        - AWS cloud platforms
        - Docker containerization
        
        Responsibilities:
        - Develop applications
        - Work with data teams
        """

        requirements = ats_scorer.parse_job_requirements(jd_text)

        assert "python" in [skill.lower() for skill in requirements["required_skills"]]
        assert "machine learning" in [
            skill.lower() for skill in requirements["required_skills"]
        ]
        assert len(requirements["all_skills"]) > 0

    def test_analyze_resume_basic(self, ats_scorer):
        """Test basic resume analysis"""
        resume_text = """
        John Doe
        john.doe@email.com
        +1-555-123-4567
        
        Experience:
        Software Engineer with Python, JavaScript, and Machine Learning experience
        
        Skills:
        - Python programming
        - TensorFlow
        - AWS
        - SQL databases
        
        Education:
        BS Computer Science
        """

        analysis = ats_scorer.analyze_resume(resume_text)

        assert len(analysis["all_skills"]) > 0
        assert analysis["contact_formatting"]["email"] == True
        assert analysis["contact_formatting"]["phone"] == True
        assert "experience" in analysis["contact_formatting"]["sections_present"]
