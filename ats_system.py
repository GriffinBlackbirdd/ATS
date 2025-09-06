
import pypdf
import re
from collections import defaultdict
import json
from typing import Dict, List, Tuple

class ATSScorer:
    def __init__(self):
        """Initialize the ATS Scorer with optimized weightage system (no grammar checking)."""
        self.weights = {
            'keyword_match': 35,        # 35% - JD extraction + alignment
            'skills_relevance': 25,     # 25% - Skills relevance to role
            'experience_achievements': 25,  # 25% - Action verbs + metrics
            'formatting_compliance': 15,    # 15% - ATS-friendly formatting
            'extra_sections': 15        # 15% - Certifications, projects, summary
            # Note: grammar_consistency REMOVED - LLM content assumed error-free
        }

        # Comprehensive skill database
        self.tech_skills = {
            'programming': ['python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'scala', 'r'],
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
                     'version control', 'version control systems']
        }

        # Action verbs for experience analysis
        self.action_verbs = [
            'achieved', 'analyzed', 'built', 'created', 'designed', 'developed', 'engineered',
            'established', 'generated', 'implemented', 'improved', 'increased', 'led', 'managed',
            'optimized', 'reduced', 'streamlined', 'transformed', 'delivered', 'executed',
            'automated', 'collaborated', 'coordinated', 'facilitated', 'initiated', 'launched',
            'maintained', 'monitored', 'operated', 'organized', 'supervised', 'trained',
            'leveraged', 'utilized', 'deployed', 'integrated', 'migrated', 'scaled'
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def _get_context(self, text: str, match) -> str:
        """Helper method to extract context around a regex match."""
        start_pos = max(0, match.start() - 30)
        end_pos = min(len(text), match.end() + 30)
        return text[start_pos:end_pos].replace('\n', ' ')

    def _extract_section(self, text: str, keywords: List[str]) -> str:
        """Extract specific sections from text based on keywords."""
        text_lines = text.split('\n')
        section_text = ""
        capturing = False

        for line in text_lines:
            line_lower = line.lower().strip()

            if any(keyword in line_lower for keyword in keywords):
                capturing = True
                section_text += line + "\n"
                continue

            if capturing and any(stop_word in line_lower for stop_word in
                               ['education', 'experience', 'contact', 'about', 'summary']):
                if not any(keyword in line_lower for keyword in keywords):
                    break

            if capturing:
                section_text += line + "\n"

        return section_text.strip()

    def parse_job_requirements(self, jd_text: str) -> Dict:
        """Parse job description to extract requirements and keywords with enhanced extraction."""
        self._original_jd_text = jd_text
        jd_lower = jd_text.lower()

        sections = {
            'required': self._extract_section(jd_text, ['requirements', 'required', 'must have', 'responsibilities']),
            'preferred': self._extract_section(jd_text, ['preferred', 'nice to have', 'bonus', 'plus']),
            'qualifications': self._extract_section(jd_text, ['qualifications', 'skills', 'experience'])
        }

        print(f"\n=== JOB DESCRIPTION ANALYSIS ===")

        # Extract all skills from JD
        all_skills = []
        required_skills = []
        preferred_skills = []

        # Enhanced skill extraction with context analysis
        import re

        for category, skills_list in self.tech_skills.items():
            for skill in skills_list:
                pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')
                match = pattern.search(jd_lower)

                if match:
                    all_skills.append(skill)
                    context = self._get_context(jd_lower, match)

                    # Determine if required or preferred based on context
                    if any(req_indicator in context.lower() for req_indicator in
                          ['required', 'must', 'essential', 'mandatory', 'need']):
                        required_skills.append(skill)
                        print(f"✅ Required skill: {skill}")
                    elif any(pref_indicator in context.lower() for pref_indicator in
                            ['preferred', 'nice', 'bonus', 'plus', 'advantageous']):
                        preferred_skills.append(skill)
                        print(f"⭐ Preferred skill: {skill}")
                    else:
                        # Default to required if found in main content
                        required_skills.append(skill)
                        print(f"✅ Skill (default required): {skill}")

        # Extract key phrases and industry-specific terms
        key_phrases = self._extract_key_phrases(jd_text)

        return {
            'required_skills': list(set(required_skills)),
            'preferred_skills': list(set(preferred_skills)),
            'all_skills': list(set(all_skills)),
            'key_phrases': key_phrases,
            'sections': sections,
            'total_keywords': len(set(all_skills + key_phrases))
        }

    def _extract_key_phrases(self, jd_text: str) -> List[str]:
        """Extract only meaningful domain-specific terms from JD, excluding common business phrases."""
        import re

        # Only highly specific domain terms that are truly meaningful
        healthcare_terms = ['healthcare', 'clinical', 'medical', 'patient', 'diagnosis', 'treatment',
                        'pharmaceutical', 'biotech', 'radiology', 'pathology', 'oncology', 'cardiology']

        fintech_terms = ['fintech', 'blockchain', 'cryptocurrency', 'trading', 'investment', 'banking',
                        'compliance', 'regulatory', 'aml', 'kyc', 'securities']

        # Very specific technical terms only
        advanced_tech_terms = ['microservices', 'serverless', 'containerization', 'orchestration',
                            'infrastructure', 'architecture', 'scalability', 'distributed systems',
                            'real-time', 'streaming', 'etl', 'data pipeline', 'data warehouse']

        # Exclude common business buzzwords that appear everywhere
        excluded_generic_terms = [
            'efficiency', 'deployment', 'optimization', 'strategy', 'framework', 'pipeline',
            'automation', 'scalable', 'performance', 'quality', 'experience', 'development',
            'management', 'analysis', 'solution', 'system', 'platform', 'technology',
            'innovation', 'collaboration', 'communication', 'problem-solving', 'teamwork',
            'leadership', 'project', 'business', 'operations', 'processes', 'implementation',
            'improvement', 'growth', 'success', 'results', 'goals', 'objectives', 'requirements',
            'design', 'build', 'create', 'develop', 'manage', 'lead', 'support', 'maintain',
            'monitor', 'track', 'report', 'document', 'research', 'investigate', 'evaluate',
            'assess', 'review', 'analyze', 'test', 'validate', 'verify', 'ensure', 'deliver',
            'execute', 'coordinate', 'facilitate', 'organize', 'plan', 'schedule', 'prioritize'
        ]

        key_phrases = []
        jd_lower = jd_text.lower()

        # Extract only highly specific domain terms
        for term_list in [healthcare_terms, fintech_terms, advanced_tech_terms]:
            for term in term_list:
                if re.search(r'\b' + re.escape(term) + r'\b', jd_lower):
                    key_phrases.append(term)
                    print(f"  🎯 Found meaningful domain term: {term}")

        # Filter out any accidentally included generic terms
        key_phrases = [phrase for phrase in key_phrases if phrase.lower() not in excluded_generic_terms]

        print(f"  📝 Total meaningful key phrases: {len(key_phrases)}")
        return key_phrases

    def analyze_resume(self, resume_text: str) -> Dict:
        """Comprehensive resume analysis with optimized scoring criteria."""
        print(f"\n=== COMPREHENSIVE RESUME ANALYSIS ===")

        # 1. Extract skills
        skills_analysis = self._analyze_skills(resume_text)

        # 2. Analyze experience and achievements
        experience_analysis = self._analyze_experience_achievements(resume_text)

        # 3. Check formatting compliance
        formatting_analysis = self._analyze_formatting_compliance(resume_text)

        # 4. Check extra sections
        extra_sections_analysis = self._analyze_extra_sections(resume_text)

        return {
            'skills_analysis': skills_analysis,
            'experience_analysis': experience_analysis,
            'formatting_analysis': formatting_analysis,
            'extra_sections_analysis': extra_sections_analysis,
            'all_skills': skills_analysis['all_skills']
        }

    def _analyze_skills(self, resume_text: str) -> Dict:
        """Analyze skills in resume with relevance scoring."""
        print(f"\n--- Skills Analysis ---")

        resume_lower = resume_text.lower()
        found_skills = defaultdict(list)

        import re

        for category, skills_list in self.tech_skills.items():
            for skill in skills_list:
                # Handle problematic single-character skills
                if skill.lower() in ['r', 'c', 'go']:
                    if skill.lower() == 'r':
                        r_contexts = [r'\br\s+programming\b', r'\br\s+language\b', r'\brstudio\b']
                        if any(re.search(pattern, resume_lower, re.IGNORECASE) for pattern in r_contexts):
                            found_skills[category].append(skill)
                            print(f"  ✅ Found R programming language")
                    continue

                pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')
                if pattern.search(resume_lower):
                    found_skills[category].append(skill)
                    context = self._get_context(resume_lower, pattern.search(resume_lower))
                    print(f"  ✅ Found skill: {skill} in context: ...{context[:50]}...")

        all_skills = [skill for skills_list in found_skills.values() for skill in skills_list]

        return {
            'skills_by_category': dict(found_skills),
            'all_skills': all_skills,
            'total_skills_count': len(all_skills)
        }

    def _analyze_experience_achievements(self, resume_text: str) -> Dict:
        """Analyze experience section for action verbs and quantified achievements."""
        print(f"\n--- Experience & Achievements Analysis ---")

        resume_lower = resume_text.lower()

        # Count action verbs
        action_verb_count = 0
        found_action_verbs = []

        for verb in self.action_verbs:
            if re.search(r'\b' + re.escape(verb.lower()) + r'\b', resume_lower):
                action_verb_count += 1
                found_action_verbs.append(verb)

        print(f"  ✅ Action verbs found: {action_verb_count} ({', '.join(found_action_verbs[:5])}...)")

        # Look for quantified achievements (numbers, percentages, metrics)
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
        achievement_examples = []

        for pattern in metric_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            quantified_achievements += len(matches)
            achievement_examples.extend(matches[:2])  # Keep first 2 examples per pattern

        print(f"  ✅ Quantified achievements: {quantified_achievements}")
        if achievement_examples:
            print(f"    Examples: {', '.join(achievement_examples[:3])}")

        # Calculate experience score
        action_verb_score = min(100, (action_verb_count / 10) * 100)  # Max at 10 action verbs
        metrics_score = min(100, (quantified_achievements / 5) * 100)  # Max at 5 metrics
        overall_experience_score = (action_verb_score * 0.6) + (metrics_score * 0.4)

        return {
            'action_verb_count': action_verb_count,
            'found_action_verbs': found_action_verbs,
            'quantified_achievements': quantified_achievements,
            'achievement_examples': achievement_examples,
            'action_verb_score': action_verb_score,
            'metrics_score': metrics_score,
            'overall_score': overall_experience_score
        }

    def _analyze_formatting_compliance(self, resume_text: str) -> Dict:
        """Analyze ATS-friendly formatting compliance."""
        print(f"\n--- Formatting Compliance Analysis ---")

        formatting_score = 0
        issues = []

        # Check for contact information
        email_found = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text))
        phone_found = bool(re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b', resume_text))

        if email_found:
            formatting_score += 20
            print(f"  ✅ Email address found")
        else:
            issues.append("Missing email address")
            print(f"  ❌ No email address found")

        if phone_found:
            formatting_score += 20
            print(f"  ✅ Phone number found")
        else:
            issues.append("Missing phone number")
            print(f"  ❌ No phone number found")

        # Check for essential sections
        essential_sections = ['experience', 'education', 'skills']
        sections_found = 0

        for section in essential_sections:
            if re.search(r'\b' + section + r'\b', resume_text.lower()):
                sections_found += 1
                print(f"  ✅ {section.title()} section found")
            else:
                issues.append(f"Missing {section} section")
                print(f"  ❌ {section.title()} section not found")

        formatting_score += (sections_found / len(essential_sections)) * 40

        # Check for consistent formatting patterns
        bullet_consistency = len(re.findall(r'^[\s]*[•▪▫◦‣⁃]\s', resume_text, re.MULTILINE))
        if bullet_consistency >= 3:
            formatting_score += 10
            print(f"  ✅ Consistent bullet points found ({bullet_consistency})")
        else:
            issues.append("Inconsistent or missing bullet points")

        # Check for proper spacing and structure
        proper_spacing = len(re.findall(r'\n\s*\n', resume_text)) >= 2
        if proper_spacing:
            formatting_score += 10
            print(f"  ✅ Proper section spacing")
        else:
            issues.append("Poor section spacing")

        return {
            'overall_score': min(100, formatting_score),
            'email_found': email_found,
            'phone_found': phone_found,
            'sections_found': sections_found,
            'total_sections': len(essential_sections),
            'issues': issues
        }

    def _analyze_extra_sections(self, resume_text: str) -> Dict:
        """Analyze extra sections like certifications, projects, summary with increased weight."""
        print(f"\n--- Extra Sections Analysis ---")

        resume_lower = resume_text.lower()
        extra_sections = {
            'certifications': ['certification', 'certificate', 'certified', 'license'],
            'projects': ['project', 'portfolio', 'work sample', 'github'],
            'summary': ['summary', 'profile', 'objective', 'about'],
            'awards': ['award', 'achievement', 'recognition', 'honor'],
            'publications': ['publication', 'paper', 'article', 'research'],
            'volunteering': ['volunteer', 'community', 'social'],
        }

        sections_found = []
        sections_score = 0

        for section_name, keywords in extra_sections.items():
            section_found = any(keyword in resume_lower for keyword in keywords)
            if section_found:
                sections_found.append(section_name)
                sections_score += 15  # **REDUCED from 20 to 15 points per section**
                print(f"  ✅ {section_name.title()} section found")

        # **FIXED: More reasonable bonus system**
        if len(sections_found) >= 4:
            sections_score += 15  # **REDUCED bonus**
            print(f"  🎉 Excellent: 4+ extra sections ({len(sections_found)})")
        elif len(sections_found) >= 3:
            sections_score += 10  # **REDUCED bonus**
            print(f"  🎉 Great: 3+ extra sections ({len(sections_found)})")
        elif len(sections_found) >= 2:
            sections_score += 5   # **REDUCED bonus**
            print(f"  🎉 Good: 2+ extra sections ({len(sections_found)})")

        # **FIX: Ensure score never exceeds 100**
        sections_score = min(100, sections_score)

        return {
            'sections_found': sections_found,
            'sections_count': len(sections_found),
            'overall_score': sections_score,
            'available_sections': list(extra_sections.keys())
        }
    def _calculate_keyword_match(self, resume_skills: List[str], jd_requirements: Dict) -> Tuple[float, Dict]:
        """Calculate keyword alignment focusing only on technical skills and meaningful terms."""
        print(f"\n--- Smart Keyword Match Calculation (Technical Skills Only) ---")

        # Enhanced semantic inference mapping (same as before)
        semantic_inference_map = {
            'version control': ['git', 'github', 'gitlab', 'svn', 'mercurial', 'bitbucket'],
            'version control systems': ['git', 'github', 'gitlab', 'svn', 'mercurial', 'bitbucket'],
            'cloud platforms': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
            'cloud computing': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
            'machine learning': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'ml'],
            'deep learning': ['tensorflow', 'pytorch', 'keras', 'neural networks'],
            'ai frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'langchain', 'crewai'],
            'data analysis': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi'],
            'data visualization': ['matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi'],
            'data manipulation': ['pandas', 'numpy', 'sql', 'data engineering'],
            'web development': ['react', 'nodejs', 'html', 'css', 'javascript', 'fastapi', 'flask', 'django'],
            'api development': ['fastapi', 'flask', 'rest api', 'restful api', 'graphql'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'ansible', 'terraform'],
            'containerization': ['docker', 'kubernetes'],
            'ai orchestration': ['langchain', 'crewai', 'workflow orchestration', 'orchestration frameworks'],
            'orchestration frameworks': ['langchain', 'crewai', 'ai orchestration', 'workflow orchestration'],
            'agentic systems': ['ai agents', 'agents', 'agentic workflows', 'multi-agent systems'],
            'ai agents': ['agentic systems', 'agentic workflows', 'agents', 'multi-agent systems'],
            # Add more meaningful mappings
            'microservices': ['api development', 'distributed systems', 'docker', 'kubernetes'],
            'real-time': ['streaming', 'kafka', 'redis', 'websockets'],
            'data warehouse': ['sql', 'etl', 'data engineering', 'spark', 'hadoop'],
            'infrastructure': ['aws', 'azure', 'gcp', 'terraform', 'ansible', 'docker', 'kubernetes'],
        }

        cloud_equivalents = ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services']

        # Focus only on technical skills and meaningful domain terms
        jd_technical_skills = set([skill.lower() for skill in jd_requirements['all_skills']])
        jd_meaningful_phrases = set([phrase.lower() for phrase in jd_requirements.get('key_phrases', [])])

        # Combine only meaningful terms
        all_meaningful_jd_terms = jd_technical_skills.union(jd_meaningful_phrases)

        resume_terms = set([skill.lower() for skill in resume_skills])

        if not all_meaningful_jd_terms:
            print("No meaningful JD terms found - returning 100%")
            return 100.0, {'matched': [], 'missing': []}

        print(f"📊 FILTERING RESULTS:")
        print(f"   • Technical Skills from JD: {len(jd_technical_skills)}")
        print(f"   • Meaningful Domain Phrases: {len(jd_meaningful_phrases)}")
        print(f"   • Total Meaningful Terms: {len(all_meaningful_jd_terms)}")

        # Smart matching with inference (same logic as before)
        matched_keywords = []
        missing_keywords = []
        inference_matches = []

        for jd_term in all_meaningful_jd_terms:
            term_matched = False
            match_method = ""

            # 1. Direct match
            if jd_term in resume_terms:
                matched_keywords.append(jd_term)
                term_matched = True
                match_method = "direct"

            # 2. Smart cloud platform inference
            elif jd_term in cloud_equivalents:
                context_analysis = self._analyze_jd_skill_context(jd_term, self._original_jd_text)

                if context_analysis['is_generic_requirement']:
                    user_cloud_skills = [skill for skill in resume_terms if skill in cloud_equivalents]
                    if user_cloud_skills:
                        matched_keywords.append(jd_term)
                        term_matched = True
                        match_method = f"cloud_equivalent_via_{user_cloud_skills[0]}"
                        inference_matches.append(f"{jd_term} ← {user_cloud_skills[0]} (cloud equivalency)")
                else:
                    if jd_term not in resume_terms:
                        missing_keywords.append(jd_term)
                        print(f"  ❌ MISSING (Specific): '{jd_term}' - specific platform required")

            # 3. Semantic inference for generic terms
            elif jd_term in semantic_inference_map:
                specific_skills = semantic_inference_map[jd_term]
                found_specific = [skill for skill in specific_skills if skill in resume_terms]

                if found_specific:
                    matched_keywords.append(jd_term)
                    term_matched = True
                    match_method = f"inferred_from_{found_specific[0]}"
                    inference_matches.append(f"{jd_term} ← {', '.join(found_specific[:2])} (semantic inference)")

            if not term_matched:
                missing_keywords.append(jd_term)

        # Remove duplicates
        matched_keywords = list(dict.fromkeys(matched_keywords))
        missing_keywords = list(dict.fromkeys(missing_keywords))

        match_percentage = (len(matched_keywords) / len(all_meaningful_jd_terms)) * 100

        print(f"📊 SMART MATCHING RESULTS (Meaningful Terms Only):")
        print(f"   • Meaningful JD Terms: {len(all_meaningful_jd_terms)}")
        print(f"   • Direct Matches: {len([k for k in matched_keywords if k in resume_terms])}")
        print(f"   • Inference Matches: {len(inference_matches)}")
        print(f"   • Total Matched: {len(matched_keywords)}")
        print(f"   • Missing: {len(missing_keywords)}")
        print(f"   • Match Rate: {match_percentage:.1f}%")

        if inference_matches:
            print(f"   🧠 SMART INFERENCES:")
            for inference in inference_matches[:5]:
                print(f"      • {inference}")

        # Separate analysis
        matched_skills = [term for term in matched_keywords if term in jd_requirements['all_skills']]
        missing_skills = [term for term in missing_keywords if term in jd_requirements['all_skills']]
        matched_phrases = [term for term in matched_keywords if term in jd_requirements.get('key_phrases', [])]
        missing_phrases = [term for term in missing_keywords if term in jd_requirements.get('key_phrases', [])]

        return match_percentage, {
            'matched_keywords': matched_keywords,
            'missing_keywords': missing_keywords,
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'matched_phrases': matched_phrases,
            'missing_phrases': missing_phrases,
            'inference_matches': inference_matches,
            'total_jd_terms': len(all_meaningful_jd_terms),
            'total_matches': len(matched_keywords),
            'match_rate': match_percentage,
            'excluded_generic_terms': len(jd_technical_skills) + len(jd_meaningful_phrases) - len(all_meaningful_jd_terms)
        }

    def _analyze_jd_skill_context(self, skill: str, jd_text: str) -> Dict:
        """Analyze if a skill mention in JD is generic or specific requirement."""
        import re

        jd_lower = jd_text.lower()
        skill_lower = skill.lower()

        # Find contexts where the skill is mentioned
        pattern = re.compile(r'.{0,50}\b' + re.escape(skill_lower) + r'\b.{0,50}')
        contexts = pattern.findall(jd_lower)

        is_generic_requirement = False
        is_specific_requirement = False

        for context in contexts:
            # Patterns indicating generic cloud requirement
            generic_patterns = [
                r'cloud\s+platforms?',
                r'any\s+cloud',
                r'major\s+cloud',
                r'public\s+cloud',
                r'cloud\s+providers?',
                r'such\s+as.*' + re.escape(skill_lower),
                r'like.*' + re.escape(skill_lower),
                r'including.*' + re.escape(skill_lower),
                r'e\.?g\.?.*' + re.escape(skill_lower),
                r'\(' + re.escape(skill_lower) + r'.*\)',
            ]

            for pattern in generic_patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    is_generic_requirement = True
                    break

            # Patterns indicating specific requirement
            specific_patterns = [
                r'experience\s+with\s+' + re.escape(skill_lower),
                r'proficiency\s+in\s+' + re.escape(skill_lower),
                r'knowledge\s+of\s+' + re.escape(skill_lower),
                r'must\s+have.*' + re.escape(skill_lower),
                r'required.*' + re.escape(skill_lower),
                r'expertise\s+in\s+' + re.escape(skill_lower),
            ]

            for pattern in specific_patterns:
                if re.search(pattern, context, re.IGNORECASE) and not is_generic_requirement:
                    is_specific_requirement = True
                    break

        return {
            'is_generic_requirement': is_generic_requirement,
            'is_specific_requirement': is_specific_requirement,
            'contexts': contexts
        }


    def calculate_ats_score(self, resume_analysis: Dict, jd_requirements: Dict) -> Dict:
        """Calculate comprehensive ATS score with updated weightage system (no grammar checking)."""
        print(f"\n=== CALCULATING ATS SCORES ===")

        # 1. Keyword Match (35%) - Enhanced with detailed analysis
        keyword_score_pct, keyword_details = self._calculate_keyword_match(
            resume_analysis['all_skills'], jd_requirements
        )

        # 2. Skills Relevance (25%)
        skills_relevance_pct = self._calculate_skills_relevance(
            resume_analysis['skills_analysis'], jd_requirements
        )

        # 3. Experience & Achievements (25%)
        experience_score_pct = resume_analysis['experience_analysis']['overall_score']

        # 4. Formatting Compliance (15%)
        formatting_score_pct = resume_analysis['formatting_analysis']['overall_score']

        # 5. Extra Sections (15%)
        extra_sections_pct = resume_analysis['extra_sections_analysis']['overall_score']

        # **FIX: Cap individual component scores at 100%**
        keyword_score_pct = min(100, keyword_score_pct)
        skills_relevance_pct = min(100, skills_relevance_pct)
        experience_score_pct = min(100, experience_score_pct)
        formatting_score_pct = min(100, formatting_score_pct)
        extra_sections_pct = min(100, extra_sections_pct)  # This was likely >100%

        # Calculate weighted scores (no grammar component)
        keyword_weighted = (keyword_score_pct / 100) * self.weights['keyword_match']
        skills_weighted = (skills_relevance_pct / 100) * self.weights['skills_relevance']
        experience_weighted = (experience_score_pct / 100) * self.weights['experience_achievements']
        formatting_weighted = (formatting_score_pct / 100) * self.weights['formatting_compliance']
        extra_weighted = (extra_sections_pct / 100) * self.weights['extra_sections']

        total_score = (keyword_weighted + skills_weighted + experience_weighted +
                    formatting_weighted + extra_weighted)

        # **FIX: Ensure total score never exceeds 100**
        total_score = min(100.0, total_score)

        print(f"\n--- UPDATED SCORE BREAKDOWN ---")
        print(f"Keyword Match: {keyword_score_pct:.1f}% → {keyword_weighted:.1f}/{self.weights['keyword_match']}")
        print(f"Skills Relevance: {skills_relevance_pct:.1f}% → {skills_weighted:.1f}/{self.weights['skills_relevance']}")
        print(f"Experience & Achievements: {experience_score_pct:.1f}% → {experience_weighted:.1f}/{self.weights['experience_achievements']}")
        print(f"Formatting: {formatting_score_pct:.1f}% → {formatting_weighted:.1f}/{self.weights['formatting_compliance']}")
        print(f"Extra Sections: {extra_sections_pct:.1f}% → {extra_weighted:.1f}/{self.weights['extra_sections']}")
        print(f"TOTAL SCORE: {total_score:.1f}/100")

        # **DEBUG: Show if any component exceeded 100%**
        if resume_analysis['extra_sections_analysis']['overall_score'] > 100:
            print(f"⚠️  DEBUG: Extra sections score was {resume_analysis['extra_sections_analysis']['overall_score']:.1f}% (capped at 100%)")


        return {
            'total_score': round(total_score, 1),
            'component_scores': {
                'keyword_match': f"{keyword_weighted:.1f}/{self.weights['keyword_match']}",
                'skills_relevance': f"{skills_weighted:.1f}/{self.weights['skills_relevance']}",
                'experience_achievements': f"{experience_weighted:.1f}/{self.weights['experience_achievements']}",
                'formatting_compliance': f"{formatting_weighted:.1f}/{self.weights['formatting_compliance']}",
                'extra_sections': f"{extra_weighted:.1f}/{self.weights['extra_sections']}"
            },
            'component_percentages': {
                'keyword_match': round(keyword_score_pct, 1),
                'skills_relevance': round(skills_relevance_pct, 1),
                'experience_achievements': round(experience_score_pct, 1),
                'formatting_compliance': round(formatting_score_pct, 1),
                'extra_sections': round(extra_sections_pct, 1)
            },
            'keyword_analysis': keyword_details,  # NEW: Detailed keyword breakdown
            'detailed_analysis': {
                'resume_analysis': resume_analysis,
                'jd_requirements': jd_requirements
            }
        }

    def _calculate_skills_relevance(self, skills_analysis: Dict, jd_requirements: Dict) -> float:
        """Calculate skills relevance with smart inference."""
        print(f"\n--- Smart Skills Relevance Calculation ---")

        resume_skills = [skill.lower() for skill in skills_analysis['all_skills']]
        required_skills = [skill.lower() for skill in jd_requirements['required_skills']]
        preferred_skills = [skill.lower() for skill in jd_requirements['preferred_skills']]

        if not required_skills and not preferred_skills:
            return 100.0

        # Use same semantic inference for skills relevance
        semantic_inference_map = {
            'version control': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
            'version control systems': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
            'cloud platforms': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure'],
            'cloud computing': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure'],
            'machine learning': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'ml'],
            'data analysis': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly'],
            'ai orchestration': ['langchain', 'crewai', 'workflow orchestration'],
            'agentic systems': ['ai agents', 'agents', 'agentic workflows'],
        }

        cloud_equivalents = ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure']

        # Smart matching for required skills
        required_matches = 0
        for skill in required_skills:
            if skill in resume_skills:
                required_matches += 1  # Direct match
            elif skill in cloud_equivalents:
                # Check for cloud equivalency
                if any(cloud_skill in resume_skills for cloud_skill in cloud_equivalents):
                    required_matches += 1
            elif skill in semantic_inference_map:
                # Check for semantic inference
                specific_skills = semantic_inference_map[skill]
                if any(specific_skill in resume_skills for specific_skill in specific_skills):
                    required_matches += 1

        # Smart matching for preferred skills
        preferred_matches = 0
        for skill in preferred_skills:
            if skill in resume_skills:
                preferred_matches += 1  # Direct match
            elif skill in cloud_equivalents:
                if any(cloud_skill in resume_skills for cloud_skill in cloud_equivalents):
                    preferred_matches += 1
            elif skill in semantic_inference_map:
                specific_skills = semantic_inference_map[skill]
                if any(specific_skill in resume_skills for specific_skill in specific_skills):
                    preferred_matches += 1

        required_score = (required_matches / len(required_skills) * 100) if required_skills else 100
        preferred_score = (preferred_matches / len(preferred_skills) * 100) if preferred_skills else 100

        # Weighted average: 70% required, 30% preferred
        relevance_score = (required_score * 0.7) + (preferred_score * 0.3)

        print(f"   • Required Skills: {required_matches}/{len(required_skills)} = {required_score:.1f}% (with inference)")
        print(f"   • Preferred Skills: {preferred_matches}/{len(preferred_skills)} = {preferred_score:.1f}% (with inference)")
        print(f"   • Overall Relevance: {relevance_score:.1f}%")

        return relevance_score

    def generate_report(self, resume_path: str, jd_path: str) -> Dict:
        """Generate comprehensive ATS report with optimized scoring system."""

        # Extract text from PDFs
        resume_text = self.extract_text_from_pdf(resume_path)
        jd_text = self.extract_text_from_pdf(jd_path)

        # Analyze resume and job requirements
        resume_analysis = self.analyze_resume(resume_text)
        jd_requirements = self.parse_job_requirements(jd_text)

        # Calculate ATS score
        ats_results = self.calculate_ats_score(resume_analysis, jd_requirements)

        return {
            'ats_score': ats_results['total_score'],
            'component_scores': ats_results['component_scores'],
            'component_percentages': ats_results['component_percentages'],
            'keyword_analysis': ats_results['keyword_analysis'],  # NEW: Detailed keyword analysis
            'detailed_analysis': ats_results['detailed_analysis'],
            'recommendations': self._generate_comprehensive_recommendations(resume_analysis, jd_requirements, ats_results['keyword_analysis']),
            'summary_stats': {
                'total_resume_skills': len(resume_analysis['all_skills']),
                'total_jd_keywords': jd_requirements['total_keywords'],
                'keywords_matched': ats_results['keyword_analysis']['total_matches'],
                'keywords_missing': len(ats_results['keyword_analysis']['missing_keywords']),
                'match_rate': ats_results['keyword_analysis']['match_rate'],
                'action_verbs_used': resume_analysis['experience_analysis']['action_verb_count'],
                'quantified_achievements': resume_analysis['experience_analysis']['quantified_achievements'],
                'extra_sections_count': resume_analysis['extra_sections_analysis']['sections_count']
            }
        }

    def _generate_comprehensive_recommendations(self, resume_analysis: Dict, jd_requirements: Dict, keyword_analysis: Dict) -> List[str]:
        """Generate comprehensive improvement recommendations (no grammar recommendations)."""
        recommendations = []

        # Priority recommendations based on missing keywords
        if keyword_analysis['missing_skills']:
            top_missing_skills = keyword_analysis['missing_skills'][:5]
            recommendations.append(f"🎯 HIGH PRIORITY: Add missing JD skills: {', '.join(top_missing_skills)}")

        if keyword_analysis['missing_phrases']:
            top_missing_phrases = keyword_analysis['missing_phrases'][:3]
            recommendations.append(f"📝 Add missing key phrases: {', '.join(top_missing_phrases)}")

        # Skills relevance recommendations
        resume_skills_set = set([skill.lower() for skill in resume_analysis['all_skills']])
        missing_required = set([skill.lower() for skill in jd_requirements['required_skills']]) - resume_skills_set
        if missing_required:
            recommendations.append(f"⚠️ CRITICAL: Include required skills: {', '.join(list(missing_required)[:3])}")

        # Experience & achievements recommendations
        if resume_analysis['experience_analysis']['action_verb_count'] < 5:
            recommendations.append("💪 Use more action verbs to start bullet points (developed, implemented, created, etc.)")

        if resume_analysis['experience_analysis']['quantified_achievements'] < 3:
            recommendations.append("📊 Add quantified achievements with specific numbers, percentages, or metrics")

        # Formatting recommendations
        formatting_issues = resume_analysis['formatting_analysis']['issues']
        if formatting_issues:
            recommendations.extend([f"🔧 Fix formatting: {issue}" for issue in formatting_issues[:2]])

        # Extra sections recommendations (enhanced weight)
        if resume_analysis['extra_sections_analysis']['sections_count'] < 3:
            available = resume_analysis['extra_sections_analysis']['available_sections']
            found = resume_analysis['extra_sections_analysis']['sections_found']
            missing_sections = [s for s in available if s not in found]
            recommendations.append(f"📚 Consider adding more sections for higher score: {', '.join(missing_sections[:3])}")

        # General recommendations
        recommendations.extend([
            "🔄 Mirror the job description language and terminology",
            "🎯 Use industry-specific keywords naturally throughout the resume",
                    "✨ Ensure consistent formatting and professional presentation"
                ])

        return recommendations[:10]  # Limit to top 10 recommendations


def main():
   """Example usage of the optimized ATS Scorer with detailed keyword analysis."""

   # Initialize the ATS scorer
   ats_scorer = ATSScorer()

   try:
       # Generate comprehensive ATS report
       report = ats_scorer.generate_report(
           '/Users/arreyanhamid/Developer/ai-resume/resumes/g.pdf',
           '/Users/arreyanhamid/Developer/ai-resume/JD/AIJD.pdf'
       )
       print(f"\n🎯 KEYWORD FILTERING ANALYSIS:")
       keyword_analysis = report['keyword_analysis']
       if 'excluded_generic_terms' in keyword_analysis:
            print(f"   • ✅ Focused on meaningful terms only")
            print(f"   • ❌ Excluded generic business buzzwords")
            print(f"   • 📊 Quality over quantity approach")
       print(f"\n📝 MEANINGFUL TERMS BREAKDOWN:")
       if keyword_analysis['matched_skills']:
            print(f"   ✅ Technical Skills Matched ({len(keyword_analysis['matched_skills'])}):")
            for skill in keyword_analysis['matched_skills'][:10]:
                print(f"      • {skill}")

       if keyword_analysis['matched_phrases']:
            print(f"   ✅ Domain-Specific Terms Matched ({len(keyword_analysis['matched_phrases'])}):")
            for phrase in keyword_analysis['matched_phrases']:
                print(f"      • {phrase}")

       print(f"\n❌ MISSING MEANINGFUL TERMS:")
       if keyword_analysis['missing_skills']:
            print(f"   🚨 Technical Skills to Add ({len(keyword_analysis['missing_skills'])}):")
            for skill in keyword_analysis['missing_skills'][:15]:
                print(f"      • {skill}")

       if keyword_analysis['missing_phrases']:
            print(f"   🎯 Domain Terms to Consider ({len(keyword_analysis['missing_phrases'])}):")
            for phrase in keyword_analysis['missing_phrases']:
                print(f"      • {phrase}")
       # Display comprehensive results
       print("\n" + "="*120)
       print(f"{'OPTIMIZED ATS COMPREHENSIVE ANALYSIS REPORT':^120}")
       print("="*120)

       print(f"\n🎯 OVERALL ATS SCORE: {report['ats_score']}/100")

       print("\n" + "-"*120)
       print(f"{'OPTIMIZED WEIGHTAGE SYSTEM BREAKDOWN (NO GRAMMAR CHECKING)':^120}")
       print("-"*120)

       weightage_info = [
           ("Keyword Match (JD extraction + alignment)", "35%", report['component_scores']['keyword_match'], report['component_percentages']['keyword_match']),
           ("Skills Relevance", "25%", report['component_scores']['skills_relevance'], report['component_percentages']['skills_relevance']),
           ("Experience & Achievements (action verbs + metrics)", "25%", report['component_scores']['experience_achievements'], report['component_percentages']['experience_achievements']),
           ("Formatting Compliance", "15%", report['component_scores']['formatting_compliance'], report['component_percentages']['formatting_compliance']),
           ("Extra Sections (Certifications, Projects, etc.)", "15%", report['component_scores']['extra_sections'], report['component_percentages']['extra_sections'])
       ]

       for component, weight, score, percentage in weightage_info:
           print(f"📊 {component:<45} | Weight: {weight:>4} | Score: {score:>6} | Performance: {percentage:>6.1f}%")

       print("\n" + "-"*120)
       print(f"{'DETAILED KEYWORD ANALYSIS':^120}")
       print("-"*120)

       keyword_analysis = report['keyword_analysis']

       print(f"📈 KEYWORD MATCHING SUMMARY:")
       print(f"   • Total JD Keywords: {keyword_analysis['total_jd_terms']}")
       print(f"   • Keywords Matched: {keyword_analysis['total_matches']}")
       print(f"   • Keywords Missing: {len(keyword_analysis['missing_keywords'])}")
       print(f"   • Match Rate: {keyword_analysis['match_rate']:.1f}%")

       print(f"\n✅ MATCHED KEYWORDS ({keyword_analysis['total_matches']} total):")
       if keyword_analysis['matched_skills']:
           print(f"   📋 Matched Skills ({len(keyword_analysis['matched_skills'])}):")
           for i, skill in enumerate(keyword_analysis['matched_skills'][:15], 1):
               print(f"      {i:2d}. {skill}")
           if len(keyword_analysis['matched_skills']) > 15:
               print(f"      ... and {len(keyword_analysis['matched_skills']) - 15} more")

       if keyword_analysis['matched_phrases']:
           print(f"   🏷️  Matched Key Phrases ({len(keyword_analysis['matched_phrases'])}):")
           for i, phrase in enumerate(keyword_analysis['matched_phrases'], 1):
               print(f"      {i:2d}. {phrase}")

       print(f"\n❌ MISSING KEYWORDS ({len(keyword_analysis['missing_keywords'])} total):")
       if keyword_analysis['missing_skills']:
           print(f"   🚨 Missing Skills ({len(keyword_analysis['missing_skills'])}) - HIGH PRIORITY:")
           for i, skill in enumerate(keyword_analysis['missing_skills'][:20], 1):
               print(f"      {i:2d}. {skill}")
           if len(keyword_analysis['missing_skills']) > 20:
               print(f"      ... and {len(keyword_analysis['missing_skills']) - 20} more")

       if keyword_analysis['missing_phrases']:
           print(f"   📝 Missing Key Phrases ({len(keyword_analysis['missing_phrases'])}):")
           for i, phrase in enumerate(keyword_analysis['missing_phrases'], 1):
               print(f"      {i:2d}. {phrase}")

       print("\n" + "-"*120)
       print(f"{'DETAILED PERFORMANCE ANALYSIS':^120}")
       print("-"*120)

       # Summary statistics
       stats = report['summary_stats']
       print(f"📈 RESUME METRICS:")
       print(f"   • Total Skills Detected: {stats['total_resume_skills']}")
       print(f"   • Action Verbs Used: {stats['action_verbs_used']}")
       print(f"   • Quantified Achievements: {stats['quantified_achievements']}")
       print(f"   • Extra Sections: {stats['extra_sections_count']}")

       print(f"\n📋 JOB DESCRIPTION METRICS:")
       print(f"   • Total Keywords Required: {stats['total_jd_keywords']}")
       print(f"   • Required Skills: {len(report['detailed_analysis']['jd_requirements']['required_skills'])}")
       print(f"   • Preferred Skills: {len(report['detailed_analysis']['jd_requirements']['preferred_skills'])}")

       print(f"\n🎯 KEYWORD MATCHING PERFORMANCE:")
       print(f"   • Keywords Matched: {stats['keywords_matched']}/{stats['total_jd_keywords']}")
       print(f"   • Keywords Missing: {stats['keywords_missing']}")
       print(f"   • Overall Match Rate: {stats['match_rate']:.1f}%")

       print("\n" + "-"*120)
       print(f"{'IMPROVEMENT RECOMMENDATIONS':^120}")
       print("-"*120)

       print(f"🎯 PRIORITY IMPROVEMENTS:")
       for i, recommendation in enumerate(report['recommendations'], 1):
           print(f"   {i:2d}. {recommendation}")

       print("\n" + "-"*120)
       print(f"{'COMPONENT DETAILS':^120}")
       print("-"*120)

       # Experience Analysis Details
       exp_analysis = report['detailed_analysis']['resume_analysis']['experience_analysis']
       print(f"📝 EXPERIENCE & ACHIEVEMENTS BREAKDOWN:")
       print(f"   • Action Verbs: {exp_analysis['action_verb_count']} (Examples: {', '.join(exp_analysis['found_action_verbs'][:5])})")
       print(f"   • Quantified Results: {exp_analysis['quantified_achievements']} metrics found")
       if exp_analysis['achievement_examples']:
           print(f"   • Achievement Examples: {', '.join(exp_analysis['achievement_examples'][:3])}")

       # Extra Sections Details
       extra_analysis = report['detailed_analysis']['resume_analysis']['extra_sections_analysis']
       print(f"\n📚 EXTRA SECTIONS ANALYSIS:")
       print(f"   • Sections Found: {', '.join(extra_analysis['sections_found']) if extra_analysis['sections_found'] else 'None'}")
       print(f"   • Score Impact: {extra_analysis['overall_score']:.1f}/100")

       # Formatting Issues
       format_analysis = report['detailed_analysis']['resume_analysis']['formatting_analysis']
       if format_analysis['issues']:
           print(f"\n🔧 FORMATTING ISSUES TO FIX:")
           for issue in format_analysis['issues']:
               print(f"   • {issue}")

       print("\n" + "-"*120)
       print(f"{'ACTIONABLE KEYWORD RECOMMENDATIONS':^120}")
       print("-"*120)

       # Provide specific actionable recommendations for missing keywords
       missing_skills = keyword_analysis['missing_skills']
       if missing_skills:
           print(f"🎯 TOP MISSING SKILLS TO ADD (High Impact):")
           for i, skill in enumerate(missing_skills[:10], 1):
               print(f"   {i:2d}. '{skill}' - Add to skills section or experience descriptions")

           if len(missing_skills) > 10:
               print(f"   💡 Focus on the top 10 first for maximum ATS improvement")

       missing_phrases = keyword_analysis['missing_phrases']
       if missing_phrases:
           print(f"\n📝 MISSING KEY PHRASES TO INCORPORATE:")
           for i, phrase in enumerate(missing_phrases[:5], 1):
               print(f"   {i:2d}. '{phrase}' - Integrate naturally into experience bullet points")

       print(f"\n💡 KEYWORD OPTIMIZATION STRATEGY:")
       match_rate = keyword_analysis['match_rate']
       if match_rate < 60:
           print(f"   🚨 CRITICAL: {match_rate:.1f}% match rate is too low. Target 70%+ for better ATS performance.")
           print(f"   📋 Action: Add at least {len(missing_skills[:5])} critical missing skills immediately.")
       elif match_rate < 80:
           print(f"   ⚠️  MODERATE: {match_rate:.1f}% match rate is acceptable but can improve.")
           print(f"   📋 Action: Add {len(missing_skills[:3])} more relevant skills to reach 80%+ target.")
       else:
           print(f"   ✅ EXCELLENT: {match_rate:.1f}% match rate shows strong keyword alignment!")
           print(f"   📋 Action: Focus on other components like achievements and formatting.")

       print("\n" + "="*120)
       print(f"{'OPTIMIZED ATS SYSTEM - NO GRAMMAR CHECKING':^120}")
       print("="*120)

       # Verify updated weightage system
       total_weight = sum(ats_scorer.weights.values())
       print(f"✅ Total Weight Verification: {total_weight}% (should be 100%)")

       print(f"\n📊 FINAL WEIGHTAGE DISTRIBUTION:")
       for component, weight in ats_scorer.weights.items():
           component_name = component.replace('_', ' ').title()
           print(f"   • {component_name:<35}: {weight:>3}%")

       print(f"\n🎯 KEY OPTIMIZATIONS:")
       print(f"   • ✅ Removed grammar checking (LLM-generated content)")
       print(f"   • ✅ Enhanced keyword analysis with missing/matched breakdown")
       print(f"   • ✅ Increased keyword match weight (30% → 35%)")
       print(f"   • ✅ Increased skills relevance (20% → 25%)")
       print(f"   • ✅ Increased experience weight (20% → 25%)")
       print(f"   • ✅ Increased formatting importance (10% → 15%)")
       print(f"   • ✅ Enhanced extra sections scoring (10% → 15%)")
       print(f"   • ✅ Detailed missing keywords analysis for actionable insights")

       print("\n" + "="*120)
       print(f"{'END OF COMPREHENSIVE ANALYSIS WITH KEYWORD BREAKDOWN':^120}")
       print("="*120)

       # Save detailed report with keyword analysis
       with open('optimized_ats_report_with_keywords.json', 'w') as f:
           json.dump(report, f, indent=2)
       print(f"\n💾 Detailed JSON report with keyword analysis saved to 'optimized_ats_report_with_keywords.json'")

       # Additional CSV export for missing keywords (for easy reference)
       if keyword_analysis['missing_skills'] or keyword_analysis['missing_phrases']:
           missing_keywords_data = {
               'missing_skills': keyword_analysis['missing_skills'],
               'missing_phrases': keyword_analysis['missing_phrases'],
               'matched_skills': keyword_analysis['matched_skills'],
               'matched_phrases': keyword_analysis['matched_phrases']
           }

           with open('missing_keywords_analysis.json', 'w') as f:
               json.dump(missing_keywords_data, f, indent=2)
           print(f"📝 Missing keywords analysis saved to 'missing_keywords_analysis.json'")

   except Exception as e:
       print(f"❌ Error: {e}")
       print("Please ensure PDF files exist and are readable.")


if __name__ == "__main__":
   main()

# Installation requirements:
"""
pypdf==3.0.1
pandas==2.0.3
numpy==1.24.3
"""