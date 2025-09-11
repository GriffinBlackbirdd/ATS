'''
Code written by Arreyan Hamid
ATS Server
Code documented by Claude for readability
'''


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import Dict, Any
import json
from ats_system import CleanATSScorer

app = FastAPI(
    title="ATS Scorer API",
    description="API for analyzing resumes against job descriptions using ATS scoring",
    version="1.0.0"
)

# Initialize the ATS scorer
ats_scorer = CleanATSScorer()

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(..., description="Resume file (.pdf or .txt)"),
    jd: UploadFile = File(..., description="Job Description file (.pdf or .txt)")
) -> Dict[str, float]:
    """
    Analyze a resume against a job description and return the ATS score.

    Args:
        resume: Resume file (.pdf or .txt)
        jd: Job description file (.pdf or .txt)

    Returns:
        Dictionary containing the ATS score
    """

    def validate_file(filename: str, filetype: str):
        if not (filename.lower().endswith('.pdf') or filename.lower().endswith('.txt')):
            raise HTTPException(
                status_code=400,
                detail=f"{filetype} must be a PDF or TXT file"
            )

    # Validate file types
    validate_file(resume.filename, "Resume")
    validate_file(jd.filename, "Job description")

    try:
        # Preserve correct file extensions for temp files
        resume_ext = os.path.splitext(resume.filename)[1]
        jd_ext = os.path.splitext(jd.filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=resume_ext) as resume_temp:
            resume_content = await resume.read()
            resume_temp.write(resume_content)
            resume_temp_path = resume_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=jd_ext) as jd_temp:
            jd_content = await jd.read()
            jd_temp.write(jd_content)
            jd_temp_path = jd_temp.name

        # Generate ATS report using extract_text_from_file
        report = ats_scorer.generate_ats_report(resume_temp_path, jd_temp_path)

        # Clean up temporary files
        os.unlink(resume_temp_path)
        os.unlink(jd_temp_path)

        # Return only the ATS score
        return {"ats_score": report["ats_score"]}

    except Exception as e:
        # Ensure cleanup in case of failure
        if 'resume_temp_path' in locals() and os.path.exists(resume_temp_path):
            os.unlink(resume_temp_path)
        if 'jd_temp_path' in locals() and os.path.exists(jd_temp_path):
            os.unlink(jd_temp_path)

        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


@app.post("/keywords")
async def extract_keywords(
    jd: UploadFile = File(..., description="Job Description PDF file")
) -> Dict[str, Any]:
    """
    Extract keywords from a job description.

    Args:
        jd: PDF file of the job description

    Returns:
        Dictionary containing extracted keywords (tech_skills, soft_skills, all_skills, total_count)
    """

    # Validate file type
    if not jd.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Job description must be a PDF file")

    try:
        # Create temporary file to save uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as jd_temp:
            jd_content = await jd.read()
            jd_temp.write(jd_content)
            jd_temp_path = jd_temp.name

        # Extract text from JD PDF
        jd_text = ats_scorer.extract_text_from_pdf(jd_temp_path)

        # Extract skills from JD
        jd_skills = ats_scorer.extract_jd_skills(jd_text)

        # Clean up temporary file
        os.unlink(jd_temp_path)

        # Format response similar to jd_keywords.json
        jd_keywords = {
            "tech_skills": sorted(jd_skills['tech_skills']),
            "soft_skills": sorted(jd_skills['soft_skills']),
            "all_skills": sorted(jd_skills['all_skills']),
            "total_count": len(jd_skills['all_skills'])
        }

        return jd_keywords

    except Exception as e:
        # Clean up temporary file in case of error
        if 'jd_temp_path' in locals() and os.path.exists(jd_temp_path):
            os.unlink(jd_temp_path)

        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "ATS Scorer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze resume against job description (requires resume.pdf and jd.pdf)",
            "/keywords": "POST - Extract keywords from job description (requires jd.pdf)",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "ATS Scorer API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)