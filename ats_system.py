'''
Code written by Arreyan Hamid
ATS System along with Server
Code documented by Claude for readability
'''


import pypdf, os
import re
from collections import defaultdict
import json
from typing import Dict, List, Tuple
class CleanATSScorer:
    def __init__(self):
        """Initialize ATS Scorer with specified weights."""
        self.weights = {
            'keyword_match': 30,        # 30% - JD extraction + alignment
            'skills_relevance': 25,     # 25% - Skills relevance to role
            'experience_achievements': 25,  # 25% - Action verbs + metrics
            'formatting_compliance': 10,    # 10% - ATS-friendly formatting
            'extra_sections': 10        # 10% - Certifications, projects, awards
        }

        # Comprehensive skill database
        self.tech_skills = {
            'programming': ['python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'scala', 'r', 'sql'],
            'ai_ml': ['machine learning', 'deep learning', 'neural networks', 'tensorflow', 'pytorch',
                     'scikit-learn', 'keras', 'opencv', 'nlp', 'computer vision', 'reinforcement learning',
                     'generative ai', 'langchain', 'crewai', 'llm', 'transformers', 'hugging face',
                     'ai agents', 'agentic systems', 'agentic workflows', 'ai orchestration',
                     'orchestration frameworks', 'agents', 'multi-agent systems', 'autonomous agents',
                     'intelligent agents', 'agent frameworks', 'workflow orchestration', 'llm orchestration'],
            'data': ['sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'pandas', 'numpy', 'matplotlib',
                    'seaborn', 'plotly', 'tableau', 'power bi', 'data analysis', 'data engineering',
                    'data preprocessing', 'etl', 'data warehouse', 'spark', 'hadoop', 'data pipelines',
                    'data manipulation', 'data science', 'analytics'],
            'cloud': ['aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'jenkins',
                     'terraform', 'ansible', 'devops', 'ci/cd', 'cloud computing', 'cloud platforms',
                     'microsoft azure', 'amazon web services'],
            'web': ['fastapi', 'flask', 'django', 'react', 'nodejs', 'express', 'html', 'css',
                   'rest api', 'restful api', 'graphql', 'microservices', 'api development'],
            'tools': ['git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'vs code', 'jupyter',
                     'version control', 'version control systems'],
            'application_domains': [
                # E-commerce & Retail
                'recommendation systems', 'recommender systems', 'recommendation engine', 'collaborative filtering',
                'content-based filtering', 'marketplace', 'marketplaces', 'e-commerce', 'dynamic pricing', 'inventory optimization',
                'customer segmentation', 'personalization', 'product matching', 'supply chain optimization',

                # NLP Applications
                'machine translation', 'translation systems', 'language translation', 'sentiment analysis',
                'text analytics', 'document analysis', 'text summarization', 'named entity recognition',
                'chatbot', 'conversational ai', 'dialogue systems', 'virtual assistant', 'voice recognition',
                'speech-to-text', 'text-to-speech', 'language modeling',

                # Computer Vision
                'image recognition', 'object detection', 'facial recognition', 'computer vision applications',
                'image classification', 'image segmentation', 'optical character recognition', 'ocr',
                'medical imaging', 'surveillance systems', 'quality inspection',
                'augmented reality', 'visual search',

                # Finance & Fintech
                'algorithmic trading', 'fraud detection', 'risk management', 'credit scoring', 'robo-advisors',
                'portfolio optimization', 'financial forecasting', 'regulatory compliance', 'anti-money laundering',
                'aml', 'kyc', 'payment processing', 'blockchain', 'cryptocurrency',

                # Healthcare & Life Sciences
                'healthcare ai', 'clinical decision support', 'drug discovery', 'medical diagnosis',
                'telemedicine', 'electronic health records', 'ehr', 'clinical trials', 'genomics',
                'bioinformatics', 'medical devices', 'radiology', 'pathology',

                # Search & Information Retrieval
                'search engines', 'information retrieval', 'semantic search', 'query understanding',
                'search ranking', 'elasticsearch', 'solr', 'knowledge graphs', 'entity resolution',

                # Prediction & Forecasting
                'demand forecasting', 'predictive maintenance', 'time series analysis', 'forecasting models',
                'predictive analytics', 'business intelligence', 'supply chain forecasting', 'energy forecasting',

                # Security & Cybersecurity
                'cybersecurity', 'threat detection', 'malware analysis', 'network security', 'intrusion detection',
                'vulnerability assessment', 'security analytics', 'incident response', 'threat intelligence',

                # Automotive & Transportation
                'autonomous vehicles', 'self-driving cars', 'route optimization', 'fleet management',
                'traffic optimization', 'logistics', 'delivery optimization', 'ride-sharing',

                # Manufacturing & Industrial
                'predictive maintenance', 'quality control', 'process optimization', 'industrial automation',
                'smart manufacturing', 'digital twin', 'production planning', 'defect detection',

                # Media & Entertainment
                'content recommendation', 'video analysis', 'content moderation', 'streaming optimization',
                'game ai', 'player behavior analysis', 'matchmaking systems', 'content generation',

                # IoT & Smart Systems
                'iot', 'internet of things', 'edge computing', 'smart cities', 'smart homes',
                'sensor data analysis', 'real-time monitoring', 'predictive sensors', 'smart grids',

                # Marketing & Advertising
                'ad targeting', 'programmatic advertising', 'marketing automation', 'customer lifetime value',
                'attribution modeling', 'a/b testing', 'conversion optimization', 'audience segmentation',

                # Energy & Utilities
                'smart grids', 'energy optimization', 'renewable energy', 'power forecasting',
                'energy trading', 'grid management', 'demand response', 'carbon footprint analysis',

                # Real Estate & Construction
                'property valuation', 'real estate analytics', 'construction optimization', 'smart buildings',
                'facility management', 'urban planning', 'property management',

                # Agriculture & Food
                'precision agriculture', 'crop monitoring', 'yield prediction', 'food safety',
                'supply chain traceability', 'agricultural automation', 'livestock monitoring',

                # Education & EdTech
                'personalized learning', 'adaptive learning', 'student performance prediction', 'automated grading',
                'educational analytics', 'learning management systems', 'skill assessment'
            ]
        }

        # Soft skills
        self.soft_skills = [
            'communication', 'teamwork', 'leadership', 'problem solving', 'analytical',
            'critical thinking', 'creativity', 'adaptability', 'time management', 'collaboration',
            'project management', 'attention to detail', 'organizational skills'
        ]

        # Action verbs for experience analysis
        self.action_verbs = [
            'achieved', 'analyzed', 'built', 'created', 'designed', 'developed', 'engineered',
            'established', 'generated', 'implemented', 'improved', 'increased', 'led', 'managed',
            'optimized', 'reduced', 'streamlined', 'transformed', 'delivered', 'executed',
            'automated', 'collaborated', 'coordinated', 'facilitated', 'initiated', 'launched',
            'maintained', 'monitored', 'operated', 'organized', 'supervised', 'trained',
            'leveraged', 'utilized', 'deployed', 'integrated', 'migrated', 'scaled'
        ]

    # def extract_text_from_file(self, pdf_path: str) -> str:
    #     """Extract text content from PDF file."""
    #     try:
    #         with open(pdf_path, 'rb') as file:
    #             reader = pypdf.PdfReader(file)
    #             text = ""
    #             for page in reader.pages:
    #                 text += page.extract_text()
    #             return text.strip()
    #     except Exception as e:
    #         raise Exception(f"Error reading PDF: {str(e)}")

    def extract_text_from_file(self, file_path):
        """
        Extracts text from either PDF (.pdf) or plain text (.txt).
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            return text

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            raise ValueError(f"Unsupported file type: {ext}. Please provide .pdf or .txt")


    def _match_skill_in_text(self, skill: str, text: str) -> bool:
        """Improved skill matching that handles both single words and multi-word phrases."""
        skill_lower = skill.lower()
        text_lower = text.lower()

        # For single words, use word boundaries
        if ' ' not in skill_lower:
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            return bool(re.search(pattern, text_lower))

        # For multi-word phrases, use a more flexible approach
        else:
            # Split the skill into words
            words = skill_lower.split()

            # Create pattern: word1 + flexible space + word2 + flexible space + word3...
            pattern_parts = []
            for i, word in enumerate(words):
                if i == 0:
                    pattern_parts.append(r'\b' + re.escape(word))
                else:
                    pattern_parts.append(r'\s+' + re.escape(word))

            if words:
                pattern_parts.append(r'\b')  # End with word boundary

            pattern = ''.join(pattern_parts)
            return bool(re.search(pattern, text_lower))

    def extract_jd_skills(self, jd_text: str) -> Dict:
        """Extract tech and soft skills from job description with improved phrase matching."""
        jd_lower = jd_text.lower()

        # Extract tech skills
        tech_skills_found = []
        for category, skills_list in self.tech_skills.items():
            for skill in skills_list:
                if self._match_skill_in_text(skill, jd_text):
                    tech_skills_found.append(skill)

        # Extract soft skills
        soft_skills_found = []
        for skill in self.soft_skills:
            if self._match_skill_in_text(skill, jd_text):
                soft_skills_found.append(skill)

        return {
            'tech_skills': list(set(tech_skills_found)),
            'soft_skills': list(set(soft_skills_found)),
            'all_skills': list(set(tech_skills_found + soft_skills_found))
        }

    def extract_resume_skills(self, resume_text: str) -> Dict:
        """Extract tech and soft skills from resume with improved phrase matching."""
        resume_lower = resume_text.lower()

        # Extract tech skills
        tech_skills_found = []
        for category, skills_list in self.tech_skills.items():
            for skill in skills_list:
                if self._match_skill_in_text(skill, resume_text):
                    tech_skills_found.append(skill)

        # Extract soft skills
        soft_skills_found = []
        for skill in self.soft_skills:
            if self._match_skill_in_text(skill, resume_text):
                soft_skills_found.append(skill)

        return {
            'tech_skills': list(set(tech_skills_found)),
            'soft_skills': list(set(soft_skills_found)),
            'all_skills': list(set(tech_skills_found + soft_skills_found))
        }

    def calculate_domain_experience_penalty(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        """Calculate penalty for missing critical application domains."""

        # Find domain-specific skills required by JD
        domain_skills_required = [skill for skill in jd_skills if skill in self.tech_skills['application_domains']]

        if not domain_skills_required:  # No specific domains required
            return 1.0  # No penalty

        # Find domain skills in resume
        resume_domain_skills = [skill for skill in resume_skills if skill in self.tech_skills['application_domains']]

        if not resume_domain_skills:  # No domain experience at all
            return 0.85  # 15% penalty for missing all domain experience

        # Calculate domain match rate
        matched_domains = set(resume_domain_skills) & set(domain_skills_required)
        domain_match_rate = len(matched_domains) / len(domain_skills_required)

        if domain_match_rate == 0:  # No matching domains
            return 0.85  # 15% penalty
        elif domain_match_rate < 0.5:  # Less than 50% domain match
            return 0.90  # 10% penalty
        elif domain_match_rate < 1.0:  # Partial domain match
            return 0.95  # 5% penalty
        else:  # Perfect domain match
            return 1.0  # No penalty

    def calculate_keyword_match(self, resume_skills: List[str], jd_skills: List[str]) -> Tuple[float, Dict]:
        """Calculate keyword match with intelligent inference."""

        # Semantic inference mapping
        inference_map = {
            'version control': ['git', 'github', 'gitlab', 'svn', 'mercurial', 'bitbucket'],
            'version control systems': ['git', 'github', 'gitlab', 'svn', 'mercurial', 'bitbucket'],
            'cloud platforms': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
            'cloud computing': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
            'machine learning': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'ml'],
            'ai orchestration': ['langchain', 'crewai', 'workflow orchestration', 'orchestration frameworks'],
            'orchestration frameworks': ['langchain', 'crewai', 'ai orchestration', 'workflow orchestration'],
            'agentic systems': ['ai agents', 'agents', 'agentic workflows', 'multi-agent systems'],
            'ai agents': ['agentic systems', 'agentic workflows', 'agents', 'multi-agent systems']
        }

        cloud_equivalents = ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services']

        resume_skills_lower = [skill.lower() for skill in resume_skills]
        jd_skills_lower = [skill.lower() for skill in jd_skills]

        if not jd_skills_lower:
            return 100.0, {'matched': [], 'missing': []}

        matched_skills = []
        missing_skills = []

        for jd_skill in jd_skills_lower:
            skill_matched = False

            # Direct match
            if jd_skill in resume_skills_lower:
                matched_skills.append(jd_skill)
                skill_matched = True

            # Cloud platform inference (if generic cloud requirement)
            elif jd_skill in cloud_equivalents or jd_skill in ['cloud platforms', 'cloud computing']:
                if any(cloud_skill in resume_skills_lower for cloud_skill in cloud_equivalents):
                    matched_skills.append(jd_skill)
                    skill_matched = True

            # Semantic inference
            elif jd_skill in inference_map:
                specific_skills = [s.lower() for s in inference_map[jd_skill]]
                if any(specific_skill in resume_skills_lower for specific_skill in specific_skills):
                    matched_skills.append(jd_skill)
                    skill_matched = True

            if not skill_matched:
                missing_skills.append(jd_skill)

        match_percentage = (len(matched_skills) / len(jd_skills_lower)) * 100

        return match_percentage, {
            'matched': matched_skills,
            'missing': missing_skills,
            'total_jd_skills': len(jd_skills_lower),
            'total_matched': len(matched_skills)
        }

    def calculate_skills_relevance(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        """Calculate skills relevance score using the same smart inference as keyword matching."""
        if not jd_skills:
            return 100.0

        # Use the same smart matching logic as calculate_keyword_match
        _, keyword_details = self.calculate_keyword_match(resume_skills, jd_skills)

        # Skills relevance should be the same as keyword match percentage
        # since we're measuring how relevant the resume skills are to JD requirements
        relevance_score = keyword_details['total_matched'] / keyword_details['total_jd_skills'] * 100

        return min(100.0, relevance_score)

    def analyze_experience_achievements(self, resume_text: str) -> Dict:
        """Analyze experience section for action verbs and metrics with more stringent scoring."""
        resume_lower = resume_text.lower()

        # Count action verbs
        action_verb_count = 0
        found_action_verbs = []

        for verb in self.action_verbs:
            if re.search(r'\b' + re.escape(verb.lower()) + r'\b', resume_lower):
                action_verb_count += 1
                found_action_verbs.append(verb)

        # Look for quantified achievements
        metric_patterns = [
            r'\d+%',  # percentages
            r'\$\d+',  # dollar amounts
            r'\d+\s*(?:million|thousand|k|m)',  # large numbers
            r'\d+\s*(?:hours?|days?|weeks?|months?|years?)',  # time metrics
            r'increased.*?by.*?\d+',  # improvement metrics
            r'reduced.*?by.*?\d+',  # reduction metrics
            r'improved.*?by.*?\d+',  # improvement metrics
            r'\d+\+\s*(?:users?|clients?|customers?|students?|projects?)',  # scale metrics
        ]

        quantified_achievements = 0
        for pattern in metric_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            quantified_achievements += len(matches)

        # More stringent scoring for realistic expectations
        action_verb_score = min(100, (action_verb_count / 12) * 100)  # Need 12 action verbs for max score
        metrics_score = min(100, (quantified_achievements / 8) * 100)  # Need 8 metrics for max score
        overall_score = (action_verb_score * 0.6) + (metrics_score * 0.4)

        return {
            'action_verb_count': action_verb_count,
            'quantified_achievements': quantified_achievements,
            'overall_score': overall_score
        }

    def analyze_formatting(self, resume_text: str) -> Dict:
        """Analyze formatting compliance."""
        formatting_score = 0
        issues = []

        # Check for contact information
        email_found = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text))

        # Enhanced phone number patterns to include international formats
        phone_patterns = [
            r'\b0\d{5}\s\d{5}\b',       # Indian format: 098218 70330 (0 + 5digits + space + 5digits)
            r'\b0\d{10}\b',             # Indian format with leading 0: 09821870330
            r'\+\d{1,4}[-.\s]?\d{10}',  # International format: +91-9821870330
            r'\+\d{1,4}\s?\d{10}',      # International format: +91 9821870330
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US format: 123-456-7890
            r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b',  # US format: (123) 456-7890
            r'\b\d{10}\b',              # Simple 10-digit: 9821870330
            r'\b\d{5}\s?\d{5}\b'        # Indian format with space: 98218 70330
        ]

        phone_found = any(re.search(pattern, resume_text) for pattern in phone_patterns)

        if email_found:
            formatting_score += 25
        else:
            issues.append("Missing email address")

        if phone_found:
            formatting_score += 25
        else:
            issues.append("Missing phone number")

        # Check for essential sections
        essential_sections = ['experience', 'education', 'skills']
        sections_found = 0

        for section in essential_sections:
            if re.search(r'\b' + section + r'\b', resume_text.lower()):
                sections_found += 1
            else:
                issues.append(f"Missing {section} section")

        formatting_score += (sections_found / len(essential_sections)) * 50

        return {
            'overall_score': min(100, formatting_score),
            'email_found': email_found,
            'phone_found': phone_found,
            'sections_found': sections_found,
            'issues': issues
        }

    def analyze_extra_sections(self, resume_text: str) -> Dict:
        """Analyze extra sections with realistic scoring expectations and fixed publications detection."""
        resume_lower = resume_text.lower()

        extra_sections = {
            'certifications': [r'\bcertification\b', r'\bcertificate\b', r'\bcertified\b', r'\blicense\b'],
            'projects': [r'\bproject\b', r'\bportfolio\b', r'\bwork sample\b', r'\bgithub\b'],
            'summary': [r'\bsummary\b', r'\bprofile\b', r'\bobjective\b', r'\babout\b'],
            'awards': [r'\baward\b', r'\bachievement\b', r'\brecognition\b', r'\bhonor\b'],
            'publications': [r'\bpublications?\s+section\b', r'\bpublished\s+papers?\b', r'\bresearch\s+publications?\b', r'\bjournal\s+articles?\b'],
            'volunteering': [r'\bvolunteer\b', r'\bcommunity\b', r'\bsocial\b'],
        }

        sections_found = []

        for section_name, patterns in extra_sections.items():
            # Use regex patterns with word boundaries to avoid false positives
            section_found = any(re.search(pattern, resume_lower) for pattern in patterns)
            if section_found:
                sections_found.append(section_name)

        # Realistic scoring: 3+ sections = excellent, 2 = good, 1 = acceptable
        if len(sections_found) >= 3:
            sections_score = 100  # 3+ sections = excellent
        elif len(sections_found) == 2:
            sections_score = 80   # 2 sections = good
        elif len(sections_found) == 1:
            sections_score = 60   # 1 section = acceptable
        else:
            sections_score = 0    # 0 sections = poor

        return {
            'sections_found': sections_found,
            'sections_count': len(sections_found),
            'overall_score': sections_score
        }

    def generate_ats_report_from_text(self, resume_text: str, jd_text: str) -> Dict:
        """Generate ATS report from text inputs with domain experience penalty."""

        # Extract skills
        jd_skills = self.extract_jd_skills(jd_text)
        resume_skills = self.extract_resume_skills(resume_text)

        # Calculate scores
        keyword_score, keyword_details = self.calculate_keyword_match(
            resume_skills['all_skills'], jd_skills['all_skills']
        )

        skills_relevance_score = self.calculate_skills_relevance(
            resume_skills['all_skills'], jd_skills['all_skills']
        )

        experience_analysis = self.analyze_experience_achievements(resume_text)
        formatting_analysis = self.analyze_formatting(resume_text)
        extra_sections_analysis = self.analyze_extra_sections(resume_text)

        # Calculate domain experience penalty
        domain_penalty = self.calculate_domain_experience_penalty(
            resume_skills['all_skills'], jd_skills['all_skills']
        )

        # Calculate weighted final score
        keyword_weighted = (keyword_score / 100) * self.weights['keyword_match']
        skills_weighted = (skills_relevance_score / 100) * self.weights['skills_relevance']
        experience_weighted = (experience_analysis['overall_score'] / 100) * self.weights['experience_achievements']
        formatting_weighted = (formatting_analysis['overall_score'] / 100) * self.weights['formatting_compliance']
        extra_weighted = (extra_sections_analysis['overall_score'] / 100) * self.weights['extra_sections']

        total_score = keyword_weighted + skills_weighted + experience_weighted + formatting_weighted + extra_weighted

        # Apply domain experience penalty
        total_score = total_score * domain_penalty

        # Apply realistic scoring cap - very few resumes should score above 95%
        total_score = min(95.0, total_score)

        return {
            'ats_score': round(total_score, 1),
            'breakdown': {
                'keyword_match': f"{keyword_weighted:.1f}/{self.weights['keyword_match']}",
                'skills_relevance': f"{skills_weighted:.1f}/{self.weights['skills_relevance']}",
                'experience_achievements': f"{experience_weighted:.1f}/{self.weights['experience_achievements']}",
                'formatting_compliance': f"{formatting_weighted:.1f}/{self.weights['formatting_compliance']}",
                'extra_sections': f"{extra_weighted:.1f}/{self.weights['extra_sections']}"
            },
            'domain_penalty': f"{domain_penalty:.2f}" if domain_penalty < 1.0 else "None",
            'jd_skills': {
                'tech_skills': jd_skills['tech_skills'],
                'soft_skills': jd_skills['soft_skills'],
                'total_skills': len(jd_skills['all_skills'])
            },
            'skill_matching': {
                'total_jd_skills': keyword_details['total_jd_skills'],
                'matched_skills': keyword_details['matched'],
                'missing_skills': keyword_details['missing'],
                'match_percentage': round(keyword_score, 1)
            },
            'formatting_check': {
                'email_present': formatting_analysis['email_found'],
                'phone_present': formatting_analysis['phone_found'],
                'essential_sections_found': formatting_analysis['sections_found'],
                'issues': formatting_analysis['issues']
            },
            'experience_metrics': {
                'action_verbs_count': experience_analysis['action_verb_count'],
                'quantified_achievements': experience_analysis['quantified_achievements']
            },
            'extra_sections': {
                'sections_found': extra_sections_analysis['sections_found'],
                'count': extra_sections_analysis['sections_count']
            }
        }

    def generate_ats_report(self, resume_path: str, jd_path: str) -> Dict:
        """Generate complete ATS report."""

        # Extract text from PDFs
        resume_text = self.extract_text_from_file(resume_path)
        jd_text = self.extract_text_from_file(jd_path)

        # Use the text-based report generation
        return self.generate_ats_report_from_text(resume_text, jd_text)


def main():
    """Example usage with PDF files"""
    ats_scorer = CleanATSScorer()

    try:
        # Generate report from PDF files
        report = ats_scorer.generate_ats_report('/Volumes/Crucible/ATS/ATS/Garv.pdf', 'jd.txt')
        # report = ats_scorer.generate_ats_report('/Volumes/Crucible/ATS/ATS/Garv.pdf', 'jd.pdf')


        # Display results
        print("=" * 60)
        print("ATS SCORE REPORT")
        print("=" * 60)
        print(f"\n🎯 ATS SCORE: {report['ats_score']}/100")

        if report.get('domain_penalty') and report['domain_penalty'] != "None":
            print(f"   ⚠️  Domain Experience Penalty Applied: {report['domain_penalty']}")

        print(f"\n📊 SCORE BREAKDOWN:")
        for component, score in report['breakdown'].items():
            component_name = component.replace('_', ' ').title()
            print(f"   • {component_name:<25}: {score}")

        print(f"\n  JD SKILLS ANALYSIS:")
        print(f"   • Tech Skills ({len(report['jd_skills']['tech_skills'])}): {', '.join(report['jd_skills']['tech_skills'][:10])}")
        if len(report['jd_skills']['tech_skills']) > 10:
            print(f"     ... and {len(report['jd_skills']['tech_skills']) - 10} more")
        print(f"   • Soft Skills ({len(report['jd_skills']['soft_skills'])}): {', '.join(report['jd_skills']['soft_skills'])}")

        print(f"\n  SKILL MATCHING:")
        print(f"   • Total JD Skills: {report['skill_matching']['total_jd_skills']}")
        print(f"   • Skills Matched: {len(report['skill_matching']['matched_skills'])}")
        print(f"   • Match Percentage: {report['skill_matching']['match_percentage']}%")
        print(f"   • Matched: {', '.join(report['skill_matching']['matched_skills'][:15])}")
        if len(report['skill_matching']['matched_skills']) > 15:
            print(f"     ... and {len(report['skill_matching']['matched_skills']) - 15} more")

        if report['skill_matching']['missing_skills']:
            print(f"   • Missing: {', '.join(report['skill_matching']['missing_skills'][:15])}")
            if len(report['skill_matching']['missing_skills']) > 15:
                print(f"     ... and {len(report['skill_matching']['missing_skills']) - 15} more")

        print(f"\n  FORMATTING CHECK:")
        print(f"   • Email Present: {'✅' if report['formatting_check']['email_present'] else '❌'}")
        print(f"   • Phone Present: {'✅' if report['formatting_check']['phone_present'] else '❌'}")
        print(f"   • Essential Sections: {report['formatting_check']['essential_sections_found']}/3")
        if report['formatting_check']['issues']:
            print(f"   • Issues: {', '.join(report['formatting_check']['issues'])}")

        print(f"\n  EXTRA SECTIONS:")
        if report['extra_sections']['sections_found']:
            print(f"   • Found ({report['extra_sections']['count']}): {', '.join(report['extra_sections']['sections_found'])}")
        else:
            print(f"   • None found")

        print(f"\n  EXPERIENCE METRICS:")
        print(f"   • Action Verbs: {report['experience_metrics']['action_verbs_count']}")
        print(f"   • Quantified Achievements: {report['experience_metrics']['quantified_achievements']}")

        print("\n" + "=" * 60)

        # Save detailed report
        with open('ats_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("📄 Detailed report saved to 'ats_report.json'")

        # Save simple JD keywords JSON
        jd_keywords = {
            "tech_skills": sorted(report['jd_skills']['tech_skills']),
            "soft_skills": sorted(report['jd_skills']['soft_skills']),
            "all_skills": sorted(report['jd_skills']['tech_skills'] + report['jd_skills']['soft_skills']),
            "total_count": len(report['jd_skills']['tech_skills'] + report['jd_skills']['soft_skills'])
        }

        with open('jd_keywords.json', 'w') as f:
            json.dump(jd_keywords, f, indent=2)
        print("  JD keywords saved to 'jd_keywords.json'")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
   main()