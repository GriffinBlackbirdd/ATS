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
import threading
import time
import requests
from ats_system import CleanATSScorer
# from ml_ats_scorer import MLATSScorer

app = FastAPI(
    title="ATS Scorer API",
    description="API for analyzing resumes against job descriptions using ATS scoring",
    version="1.0.0"
)

# Initialize the ATS scorers
ats_scorer = CleanATSScorer()
# ml_ats_scorer = MLATSScorer()

# def keep_alive():
#     """
#     Background task to ping the service every 1 minute to keep it alive on Render.
#     """
#     def ping_service():
#         # Get the service URL from environment variable (Render auto-provides this)
#         service_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8008")
#         ping_url = f"{service_url}/ping"

#         while True:
#             try:
#                 time.sleep(60)  # Wait 1 minute (60 seconds)
#                 response = requests.get(ping_url, timeout=10)
#                 print("PING PONG PING PONG")
#             except Exception as e:
#                 print(f"Keep-alive ping failed: {e}")

#     # Start the ping thread as a daemon thread
#     ping_thread = threading.Thread(target=ping_service, daemon=True)
#     ping_thread.start()
#     print("Keep-alive service started - pinging every 1 minute")

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
    jd: UploadFile = File(..., description="Job Description file (.pdf or .txt)")
) -> Dict[str, Any]:
    """
    Extract keywords from a job description.

    Args:
        jd: Job Description file (.pdf or .txt)

    Returns:
        Dictionary containing extracted keywords (tech_skills, soft_skills, all_skills, total_count)
    """

    def validate_file(filename: str, filetype: str):
        if not (filename.lower().endswith('.pdf') or filename.lower().endswith('.txt')):
            raise HTTPException(
                status_code=400,
                detail=f"{filetype} must be a PDF or TXT file"
            )

    # Validate file type
    validate_file(jd.filename, "Job description")

    try:
        # Preserve extension for temp file
        jd_ext = os.path.splitext(jd.filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=jd_ext) as jd_temp:
            jd_content = await jd.read()
            jd_temp.write(jd_content)
            jd_temp_path = jd_temp.name

        # Extract text using extract_text_from_file (supports pdf + txt)
        jd_text = ats_scorer.extract_text_from_file(jd_temp_path)

        # Extract skills from JD text
        jd_skills = ats_scorer.extract_jd_skills(jd_text)

        # Clean up temporary file
        os.unlink(jd_temp_path)

        # Format response
        jd_keywords = {
            "tech_skills": sorted(jd_skills['tech_skills']),
            "soft_skills": sorted(jd_skills['soft_skills']),
            "all_skills": sorted(jd_skills['all_skills']),
            "total_count": len(jd_skills['all_skills'])
        }

        return jd_keywords

    except Exception as e:
        # Clean up temp file on error
        if 'jd_temp_path' in locals() and os.path.exists(jd_temp_path):
            os.unlink(jd_temp_path)

        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# @app.post("/analyze-ml")
# async def analyze_resume_ml(
#     resume: UploadFile = File(..., description="Resume file (.pdf or .txt)"),
#     jd: UploadFile = File(..., description="Job Description file (.pdf or .txt)")
# ) -> Dict[str, Any]:
#     """
#     Analyze a resume against a job description using ML-enhanced ATS scoring.
#     This endpoint uses sentence transformers for semantic similarity analysis.

#     Args:
#         resume: Resume file (.pdf or .txt)
#         jd: Job description file (.pdf or .txt)

#     Returns:
#         Dictionary containing the ML-enhanced ATS score and detailed breakdown
#     """

#     def validate_file(filename: str, filetype: str):
#         if not (filename.lower().endswith('.pdf') or filename.lower().endswith('.txt')):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"{filetype} must be a PDF or TXT file"
#             )

#     # Validate file types
#     validate_file(resume.filename, "Resume")
#     validate_file(jd.filename, "Job description")

#     try:
#         # Preserve correct file extensions for temp files
#         resume_ext = os.path.splitext(resume.filename)[1]
#         jd_ext = os.path.splitext(jd.filename)[1]

#         with tempfile.NamedTemporaryFile(delete=False, suffix=resume_ext) as resume_temp:
#             resume_content = await resume.read()
#             resume_temp.write(resume_content)
#             resume_temp_path = resume_temp.name

#         with tempfile.NamedTemporaryFile(delete=False, suffix=jd_ext) as jd_temp:
#             jd_content = await jd.read()
#             jd_temp.write(jd_content)
#             jd_temp_path = jd_temp.name

#         # Generate ML-enhanced ATS report
#         report = ml_ats_scorer.generate_ml_ats_report(resume_temp_path, jd_temp_path)

#         # Clean up temporary files
#         os.unlink(resume_temp_path)
#         os.unlink(jd_temp_path)

#         # Return the full ML-enhanced report
#         return report

#     except Exception as e:
#         # Ensure cleanup in case of failure
#         if 'resume_temp_path' in locals() and os.path.exists(resume_temp_path):
#             os.unlink(resume_temp_path)
#         if 'jd_temp_path' in locals() and os.path.exists(jd_temp_path):
#             os.unlink(jd_temp_path)

#         raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "ATS Scorer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze resume against job description (rule-based scoring)",
            "/analyze-ml": "POST - Analyze resume against job description (ML-enhanced scoring with semantic similarity)",
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

@app.get("/ping")
async def ping():
    """
    Ping endpoint to keep service alive.
    """
    return {"status": "pong", "message": "Service is alive"}

# @app.on_event("startup")
# async def startup_event():
#     """
#     Initialize keep-alive service when the app starts.
#     """
#     keep_alive()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)