'''
ML-Enhanced ATS Scorer using Sentence Transformers
Code written by Arreyan Hamid
ML/NLP integration documented by Claude for readability
'''

import pypdf, os
import re
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import hashlib


class MLATSScorer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize ML ATS Scorer with sentence transformer model."""
        self.weights = {
            'keyword_match': 30,        # 30% - JD extraction + alignment (with semantic understanding)
            'skills_relevance': 25,     # 25% - Skills relevance to role (enhanced with embeddings)
            'experience_achievements': 25,  # 25% - Action verbs + metrics (NLP-enhanced)
            'formatting_compliance': 10,    # 10% - ATS-friendly formatting (rule-based)
            'extra_sections': 10        # 10% - Certifications, projects, awards (rule-based)
        }

        # Load the sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")

        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}

        # Comprehensive skill database (same as original)
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

    def _get_text_hash(self, text: str) -> str:
        """Generate hash of text for caching."""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        text_hash = self._get_text_hash(text)
        return self.embedding_cache.get(text_hash)

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding for future use."""
        text_hash = self._get_text_hash(text)
        self.embedding_cache[text_hash] = embedding

    def get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding with caching."""
        # Check cache first
        cached = self._get_cached_embedding(text)
        if cached is not None:
            return cached

        # Generate new embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        self._cache_embedding(text, embedding)
        return embedding

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Get embeddings
        embedding1 = self.get_embedding(text1).reshape(1, -1)
        embedding2 = self.get_embedding(text2).reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)

    def extract_text_from_file(self, file_path):
        """Extracts text from either PDF (.pdf) or plain text (.txt)."""
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

    def calculate_ml_keyword_match(self, resume_text: str, jd_text: str, resume_skills: List[str], jd_skills: List[str]) -> Tuple[float, Dict]:
        """Calculate keyword match using both exact matching and semantic similarity."""
        # First, do exact matching (same as original)
        semantic_inference_map = {
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

        matched_skills = []
        missing_skills = []

        for jd_skill in jd_skills_lower:
            skill_matched = False

            # Direct match
            if jd_skill in resume_skills_lower:
                matched_skills.append(jd_skill)
                skill_matched = True

            # Cloud platform inference
            elif jd_skill in cloud_equivalents or jd_skill in ['cloud platforms', 'cloud computing']:
                if any(cloud_skill in resume_skills_lower for cloud_skill in cloud_equivalents):
                    matched_skills.append(jd_skill)
                    skill_matched = True

            # Semantic inference
            elif jd_skill in semantic_inference_map:
                specific_skills = [s.lower() for s in semantic_inference_map[jd_skill]]
                if any(specific_skill in resume_skills_lower for specific_skill in specific_skills):
                    matched_skills.append(jd_skill)
                    skill_matched = True

            if not skill_matched:
                missing_skills.append(jd_skill)

        # Calculate exact match percentage
        exact_match_percentage = (len(matched_skills) / len(jd_skills_lower)) * 100 if jd_skills_lower else 100.0

        # Now calculate semantic similarity for the overall texts
        semantic_score = self.calculate_semantic_similarity(resume_text, jd_text) * 100

        # Combine exact matching and semantic similarity (70% exact, 30% semantic)
        combined_score = (exact_match_percentage * 0.7) + (semantic_score * 0.3)

        return combined_score, {
            'matched': matched_skills,
            'missing': missing_skills,
            'total_jd_skills': len(jd_skills_lower),
            'total_matched': len(matched_skills),
            'exact_match_percentage': exact_match_percentage,
            'semantic_similarity': semantic_score,
            'ml_contribution': 30  # 30% from ML, 70% from rules
        }

    def calculate_ml_skills_relevance(self, resume_text: str, jd_text: str, resume_skills: List[str], jd_skills: List[str]) -> float:
        """Calculate skills relevance using enhanced ML matching."""
        if not jd_skills:
            return 100.0

        # Get individual skill embeddings for better matching
        skill_similarities = []

        for jd_skill in jd_skills:
            best_match_score = 0

            # Check against each resume skill
            for resume_skill in resume_skills:
                # Multiple context embeddings for better matching
                contexts = [
                    f"experience with {jd_skill}",
                    f"proficient in {jd_skill}",
                    f"skilled in {jd_skill}",
                    f"expert in {jd_skill}",
                    jd_skill  # Just the skill itself
                ]

                resume_contexts = [
                    f"experience with {resume_skill}",
                    f"proficient in {resume_skill}",
                    f"skilled in {resume_skill}",
                    f"expert in {resume_skill}",
                    resume_skill
                ]

                # Calculate similarity for all context pairs
                for jd_ctx in contexts:
                    for resume_ctx in resume_contexts:
                        jd_embedding = self.get_embedding(jd_ctx)
                        resume_embedding = self.get_embedding(resume_ctx)
                        similarity = float(cosine_similarity(jd_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0][0])
                        best_match_score = max(best_match_score, similarity)

            # Check for skill variations and related terms
            skill_variations = self._get_skill_variations(jd_skill)
            for variation in skill_variations:
                if variation.lower() in resume_text.lower():
                    best_match_score = max(best_match_score, 0.7)  # Variation match gets good score

            # Also check if the skill appears directly in resume text
            if jd_skill.lower() in resume_text.lower():
                best_match_score = max(best_match_score, 0.8)  # Direct match gets high score

            skill_similarities.append(best_match_score)

        # Calculate average similarity with bonus for high matches
        avg_similarity = float(np.mean(skill_similarities)) * 100

        # Calculate overall text similarity with enhanced method
        overall_similarity = self._calculate_enhanced_semantic_similarity(resume_text, jd_text) * 100

        # Calculate skill density bonus (more skills mentioned in context = higher score)
        skill_density_bonus = min(5, len(resume_skills) / 10) * 2  # Up to 5% bonus

        # Combine scores with adjusted weights
        relevance_score = (avg_similarity * 0.5) + (overall_similarity * 0.45) + skill_density_bonus

        return min(100.0, relevance_score)

    def _get_skill_variations(self, skill: str) -> List[str]:
        """Get common variations of skills."""
        variations = [skill]

        # Add common variations
        if 'machine learning' in skill.lower():
            variations.extend(['ml', 'artificial intelligence', 'ai'])
        elif 'artificial intelligence' in skill.lower():
            variations.extend(['ai', 'machine learning'])
        elif 'python' in skill.lower():
            variations.extend(['python programming', 'python development'])
        elif 'cloud' in skill.lower():
            variations.extend(['cloud computing', 'cloud platforms'])
        elif 'data' in skill.lower():
            variations.extend(['data science', 'data analysis'])

        return variations

    def _calculate_enhanced_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate enhanced semantic similarity using multiple approaches."""
        # Split into chunks for better comparison
        chunks1 = self._split_into_chunks(text1)
        chunks2 = self._split_into_chunks(text2)

        max_similarities = []

        # Compare each chunk from text1 with each chunk from text2
        for chunk1 in chunks1:
            chunk_similarities = []
            for chunk2 in chunks2:
                if chunk1.strip() and chunk2.strip():
                    embedding1 = self.get_embedding(chunk1)
                    embedding2 = self.get_embedding(chunk2)
                    similarity = float(cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0])
                    chunk_similarities.append(similarity)

            if chunk_similarities:
                max_similarities.append(max(chunk_similarities))

        # Return the average of maximum similarities
        if max_similarities:
            return np.mean(max_similarities)
        else:
            return self.calculate_semantic_similarity(text1, text2)

    def _split_into_chunks(self, text: str, chunk_size: int = 100) -> List[str]:
        """Split text into chunks for better semantic comparison."""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def analyze_experience_achievements_ml(self, resume_text: str) -> Dict:
        """Analyze experience section using enhanced NLP for better achievement detection."""
        resume_lower = resume_text.lower()

        # Count action verbs
        action_verb_count = 0
        found_action_verbs = []

        for verb in self.action_verbs:
            if re.search(r'\b' + re.escape(verb.lower()) + r'\b', resume_lower):
                action_verb_count += 1
                found_action_verbs.append(verb)

        # Enhanced metric patterns using NLP
        metric_patterns = [
            r'\d+%',  # percentages
            r'\$\d+(?:\.\d{2})?',  # dollar amounts with cents
            r'\d+\s*(?:million|thousand|k|m|billion|b)',  # large numbers
            r'\d+\s*(?:hours?|days?|weeks?|months?|years?)',  # time metrics
            r'(?:increased|decreased|reduced|improved|grew|saved).*?\d+',  # improvement metrics
            r'\d+\+\s*(?:users?|clients?|customers?|students?|projects?|employees?)',  # scale metrics
            r'from\s+\d+\s+to\s+\d+',  # range improvements
            r'by\s+\d+%?',  # percentage improvements
        ]

        quantified_achievements = 0
        metric_details = []

        for pattern in metric_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            quantified_achievements += len(matches)
            if matches:
                metric_details.extend(matches)

        # Enhanced achievement detection with multiple prototypes
        sentences = re.split(r'[.!?]+', resume_text)
        achievement_sentences = 0

        # Multiple achievement prototypes for better detection
        achievement_prototypes = [
            "Successfully completed a project with measurable results",
            "Achieved significant improvements in business performance",
            "Delivered exceptional results through technical innovation",
            "Enhanced operational efficiency with quantifiable outcomes",
            "Generated substantial business value through strategic initiatives",
            "Improved key metrics through data-driven solutions",
            "Led transformational projects resulting in measurable success"
        ]

        for sentence in sentences:
            if sentence.strip():
                # Enhanced achievement indicators
                achievement_indicators = [
                    'achieved', 'resulted in', 'led to', 'delivered', 'produced', 'enhanced',
                    'improved', 'increased', 'decreased', 'reduced', 'optimized', 'transformed',
                    'generated', 'saved', 'gained', 'secured', 'established'
                ]

                # Check if sentence starts with achievement language
                starts_achievement = any(
                    sentence.strip().lower().startswith(indicator)
                    for indicator in ['achieved', 'resulted', 'led', 'delivered', 'enhanced', 'improved']
                )

                # Check for achievement indicators in sentence
                has_indicators = any(indicator in sentence.lower() for indicator in achievement_indicators)

                if starts_achievement or has_indicators:
                    # Get embedding for the sentence
                    sentence_embedding = self.get_embedding(sentence)

                    # Compare with all achievement prototypes
                    max_similarity = 0
                    for proto in achievement_prototypes:
                        proto_embedding = self.get_embedding(proto)
                        similarity = float(cosine_similarity(
                            sentence_embedding.reshape(1, -1),
                            proto_embedding.reshape(1, -1)
                        )[0][0])
                        max_similarity = max(max_similarity, similarity)

                    # Lower threshold for sentences starting with achievement words
                    threshold = 0.4 if starts_achievement else 0.5

                    if max_similarity > threshold:
                        achievement_sentences += 1

        # Enhanced curved scoring
        # Action verbs: curved scoring
        if action_verb_count >= 10:
            action_verb_score = 100
        elif action_verb_count >= 8:
            action_verb_score = 90
        elif action_verb_count >= 6:
            action_verb_score = 75
        elif action_verb_count >= 4:
            action_verb_score = 60
        else:
            action_verb_score = (action_verb_count / 12) * 100

        # Metrics: curved scoring with bonus for high numbers
        if quantified_achievements >= 30:
            metrics_score = 100  # Bonus for exceptional metrics
        elif quantified_achievements >= 20:
            metrics_score = 95
        elif quantified_achievements >= 15:
            metrics_score = 90
        elif quantified_achievements >= 10:
            metrics_score = 85
        elif quantified_achievements >= 8:
            metrics_score = 80
        else:
            metrics_score = min(100, (quantified_achievements / 8) * 100)

        # Achievement sentences: more generous scoring
        if achievement_sentences >= 8:
            achievement_score = 100
        elif achievement_sentences >= 6:
            achievement_score = 90
        elif achievement_sentences >= 4:
            achievement_score = 80
        elif achievement_sentences >= 2:
            achievement_score = 70
        else:
            achievement_score = min(100, (achievement_sentences / 10) * 100)

        # Weighted combination with adjusted weights
        overall_score = (action_verb_score * 0.35) + (metrics_score * 0.45) + (achievement_score * 0.2)

        return {
            'action_verb_count': action_verb_count,
            'quantified_achievements': quantified_achievements,
            'achievement_sentences': achievement_sentences,
            'overall_score': overall_score,
            'ml_enhanced': True,
            'metric_examples': metric_details[:5],  # Show first 5 metric examples
            'scoring_breakdown': {
                'action_verb_score': action_verb_score,
                'metrics_score': metrics_score,
                'achievement_score': achievement_score
            }
        }

    def calculate_domain_experience_penalty(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        """Calculate penalty for missing critical application domains (same as original)."""
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

    def analyze_formatting(self, resume_text: str) -> Dict:
        """Analyze formatting compliance (same as original)."""
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
        """Analyze extra sections with improved detection patterns."""
        resume_lower = resume_text.lower()

        extra_sections = {
            'certifications': [r'\b[Cc]ertifications?\b', r'\b[Cc]ertificate\b', r'\b[Cc]ertified\b', r'\blicense\b'],
            'projects': [r'\bprojects?\b', r'\bportfolio\b', r'\bwork sample\b', r'\bgit(hub)?\b'],
            'summary': [r'\bsummary\b', r'\bprofile\b', r'\bobjective\b', r'\babout\b'],
            'technologies': [r'\btechnologies?\b', r'\btechnical skills?\b', r'\bskills?\b', r'\btech stack\b'],
            'awards': [r'\baward\b', r'\bachievement\b', r'\brecognition\b', r'\bhonor\b'],
            'publications': [r'\bpublications?\b', r'\bpublished\s+papers?\b', r'\bresearch\s+publications?\b', r'\bjournal\s+articles?\b'],
            'volunteering': [r'\bvolunteer\b', r'\bcommunity\b', r'\bsocial\b', r'\bleadership\b'],
        }

        sections_found = []

        for section_name, patterns in extra_sections.items():
            # Use regex patterns with word boundaries to avoid false positives
            section_found = any(re.search(pattern, resume_lower) for pattern in patterns)
            if section_found:
                sections_found.append(section_name)

        # Updated scoring: 4+ sections = excellent, 3 = good, 2 = acceptable, 1 = poor
        if len(sections_found) >= 4:
            sections_score = 100  # 4+ sections = excellent
        elif len(sections_found) == 3:
            sections_score = 90   # 3 sections = very good
        elif len(sections_found) == 2:
            sections_score = 75   # 2 sections = good
        elif len(sections_found) == 1:
            sections_score = 50   # 1 section = poor
        else:
            sections_score = 0    # 0 sections = very poor

        return {
            'sections_found': sections_found,
            'sections_count': len(sections_found),
            'overall_score': sections_score
        }

    def generate_ml_ats_report_from_text(self, resume_text: str, jd_text: str) -> Dict:
        """Generate ML-enhanced ATS report from text inputs."""
        start_time = time.time()

        # Extract skills
        jd_skills = self.extract_jd_skills(jd_text)
        resume_skills = self.extract_resume_skills(resume_text)

        # Calculate ML-enhanced scores
        keyword_score, keyword_details = self.calculate_ml_keyword_match(
            resume_text, jd_text, resume_skills['all_skills'], jd_skills['all_skills']
        )

        skills_relevance_score = self.calculate_ml_skills_relevance(
            resume_text, jd_text, resume_skills['all_skills'], jd_skills['all_skills']
        )

        experience_analysis = self.analyze_experience_achievements_ml(resume_text)
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

        # Apply realistic scoring cap
        total_score = min(95.0, total_score)

        # Calculate processing time
        processing_time = time.time() - start_time

        return {
            'ats_score': round(total_score, 1),
            'ml_enhanced': True,
            'processing_time': round(processing_time, 2),
            'model_used': 'all-MiniLM-L6-v2',
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
                'match_percentage': round(keyword_score, 1),
                'ml_contribution': f"{keyword_details['ml_contribution']}%",
                'semantic_similarity': round(keyword_details['semantic_similarity'], 1)
            },
            'formatting_check': {
                'email_present': formatting_analysis['email_found'],
                'phone_present': formatting_analysis['phone_found'],
                'essential_sections_found': formatting_analysis['sections_found'],
                'issues': formatting_analysis['issues']
            },
            'experience_metrics': {
                'action_verbs_count': experience_analysis['action_verb_count'],
                'quantified_achievements': experience_analysis['quantified_achievements'],
                'achievement_sentences': experience_analysis['achievement_sentences'],
                'metric_examples': experience_analysis['metric_examples'],
                'ml_enhanced': experience_analysis['ml_enhanced']
            },
            'extra_sections': {
                'sections_found': extra_sections_analysis['sections_found'],
                'count': extra_sections_analysis['sections_count']
            }
        }

    def generate_ml_ats_report(self, resume_path: str, jd_path: str) -> Dict:
        """Generate complete ML-enhanced ATS report."""
        # Extract text from PDFs
        resume_text = self.extract_text_from_file(resume_path)
        jd_text = self.extract_text_from_file(jd_path)

        # Use the text-based report generation
        return self.generate_ml_ats_report_from_text(resume_text, jd_text)