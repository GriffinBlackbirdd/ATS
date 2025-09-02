#!/usr/bin/env python3
"""
Simple test script to verify the ATS API is working
"""

import requests
import sys
from pathlib import Path

def test_health_check():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:8000")
        return False

def test_file_upload(resume_path: str, jd_path: str):
    """Test file upload endpoint"""
    if not Path(resume_path).exists():
        print(f"❌ Resume file not found: {resume_path}")
        return False
    
    if not Path(jd_path).exists():
        print(f"❌ JD file not found: {jd_path}")
        return False
    
    try:
        with open(resume_path, 'rb') as resume_file, open(jd_path, 'rb') as jd_file:
            files = {
                'resume': ('resume.pdf', resume_file, 'application/pdf'),
                'job_description': ('jd.pdf', jd_file, 'application/pdf')
            }
            
            print("📤 Uploading files for analysis...")
            response = requests.post("http://localhost:8000/analyze-simple", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Analysis successful!")
                print(f"📊 ATS Score: {result['ats_score']}/100")
                print(f"✅ Matched Skills: {len(result['matched_skills'])}")
                print(f"❌ Missing Required: {len(result['missing_required_skills'])}")
                print(f"⭐ Missing Preferred: {len(result['missing_preferred_skills'])}")
                return True
            else:
                print(f"❌ Analysis failed with status {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error during file upload: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing ATS API...")
    
    # Test health check
    if not test_health_check():
        sys.exit(1)
    
    # Check if PDF files are provided
    if len(sys.argv) >= 3:
        resume_path = sys.argv[1]
        jd_path = sys.argv[2]
        test_file_upload(resume_path, jd_path)
    else:
        print("💡 To test file upload, run: python test_api.py <resume.pdf> <jd.pdf>")
    
    print("✅ All tests completed!")