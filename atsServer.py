from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from typing import Dict
import uvicorn
from ats_system import ATSScorer

app = FastAPI(
    title="ATS Resume Analyzer API",
    description="Upload resume and job description PDFs to get ATS compatibility analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ATS scorer
ats_scorer = ATSScorer()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ATS Resume Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Upload resume and JD PDFs for analysis",
            "/health": "GET - Health check endpoint",
            "/docs": "GET - Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "ATS API is running"}


@app.post("/analyze")
async def analyze_resume_and_jd(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: UploadFile = File(..., description="Job Description PDF file"),
):
    """
    Analyze resume against job description for ATS compatibility

    Args:
        resume: PDF file containing the resume
        job_description: PDF file containing the job description

    Returns:
        JSON response with ATS analysis results
    """

    # Validate file types
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume file must be a PDF")

    if not job_description.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Job description file must be a PDF"
        )

    # Create temporary files
    resume_temp_path = None
    jd_temp_path = None

    try:
        # Save uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as resume_temp:
            resume_content = await resume.read()
            resume_temp.write(resume_content)
            resume_temp_path = resume_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as jd_temp:
            jd_content = await job_description.read()
            jd_temp.write(jd_content)
            jd_temp_path = jd_temp.name

        # Generate ATS analysis report
        report = ats_scorer.generate_report(resume_temp_path, jd_temp_path)

        # Add metadata
        response_data = {
            "status": "success",
            "analysis_timestamp": "2024-12-20T12:00:00Z",  # You might want to add actual timestamp
            "files_analyzed": {
                "resume_filename": resume.filename,
                "job_description_filename": job_description.filename,
            },
            "results": report,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Clean up temporary files
        if resume_temp_path and os.path.exists(resume_temp_path):
            os.unlink(resume_temp_path)
        if jd_temp_path and os.path.exists(jd_temp_path):
            os.unlink(jd_temp_path)


@app.post("/analyze-quick")
async def analyze_quick(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: UploadFile = File(..., description="Job Description PDF file"),
):
    """
    Quick analysis returning only essential metrics

    Args:
        resume: PDF file containing the resume
        job_description: PDF file containing the job description

    Returns:
        JSON response with essential ATS metrics only
    """

    # Validate file types
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume file must be a PDF")

    if not job_description.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Job description file must be a PDF"
        )

    resume_temp_path = None
    jd_temp_path = None

    try:
        # Save uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as resume_temp:
            resume_content = await resume.read()
            resume_temp.write(resume_content)
            resume_temp_path = resume_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as jd_temp:
            jd_content = await job_description.read()
            jd_temp.write(jd_content)
            jd_temp_path = jd_temp.name

        # Generate full report
        full_report = ats_scorer.generate_report(resume_temp_path, jd_temp_path)

        # Extract only essential metrics
        quick_response = {
            "status": "success",
            "files_analyzed": {
                "resume_filename": resume.filename,
                "job_description_filename": job_description.filename,
            },
            "quick_results": {
                "ats_score": full_report["ats_score"],
                "component_percentages": full_report["component_scores"],
                "matched_skills_count": len(
                    full_report["detailed_analysis"]["strong_matches"]
                ),
                "missing_required_skills_count": len(
                    full_report["detailed_analysis"]["missing_required"]
                ),
                "missing_preferred_skills_count": len(
                    full_report["detailed_analysis"]["missing_preferred"]
                ),
                "top_recommendations": full_report["detailed_analysis"][
                    "recommendations"
                ][:3],
            },
        }

        return JSONResponse(content=quick_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")

    finally:
        # Clean up temporary files
        if resume_temp_path and os.path.exists(resume_temp_path):
            os.unlink(resume_temp_path)
        if jd_temp_path and os.path.exists(jd_temp_path):
            os.unlink(jd_temp_path)


@app.get("/skills-database")
async def get_skills_database():
    """
    Get the complete skills database used for analysis

    Returns:
        JSON response with all skill categories and skills
    """
    return {
        "status": "success",
        "skills_database": ats_scorer.tech_skills,
        "total_skills": sum(len(skills) for skills in ats_scorer.tech_skills.values()),
        "categories": list(ats_scorer.tech_skills.keys()),
    }


if __name__ == "__main__":
    print("Starting ATS Resume Analyzer API Server...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)
