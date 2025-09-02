# import pypdf
# import re
# from collections import defaultdict
# import json
# from typing import Dict, List, Tuple

# class ATSScorer:
#     def __init__(self):
#         """Initialize the ATS Scorer with predefined skill categories and weights."""
#         self.weights = {
#             'required_skills': 50,      # 50% weight for required skills
#             'preferred_skills': 25,     # 25% weight for preferred skills
#             'keyword_match': 20,        # 20% weight for keyword matching
#             'semantic_similarity': 5    # 5% weight for semantic similarity
#         }

#         # Comprehensive skill database
#         self.tech_skills = {
#             'programming': ['python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'scala', 'r'],
#             'ai_ml': ['machine learning', 'deep learning', 'neural networks', 'tensorflow', 'pytorch',
#                      'scikit-learn', 'keras', 'opencv', 'nlp', 'computer vision', 'reinforcement learning',
#                      'generative ai', 'langchain', 'crewai', 'llm', 'transformers', 'hugging face',
#                      'ai agents', 'agentic systems', 'agentic workflows', 'ai orchestration',
#                      'orchestration frameworks', 'agents', 'multi-agent systems', 'autonomous agents',
#                      'intelligent agents', 'agent frameworks', 'workflow orchestration', 'llm orchestration'],
#             'data': ['sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'pandas', 'numpy', 'matplotlib',
#                     'seaborn', 'plotly', 'tableau', 'power bi', 'data analysis', 'data engineering',
#                     'data preprocessing', 'etl', 'data warehouse', 'spark', 'hadoop', 'data pipelines',
#                     'data manipulation', 'data science', 'analytics'],
#             'cloud': ['aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'jenkins',
#                      'terraform', 'ansible', 'devops', 'ci/cd', 'cloud computing', 'cloud platforms',
#                      'microsoft azure', 'amazon web services'],
#             'web': ['fastapi', 'flask', 'django', 'react', 'nodejs', 'express', 'html', 'css',
#                    'rest api', 'restful api', 'graphql', 'microservices', 'api development'],
#             'tools': ['git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'vs code', 'jupyter',
#                      'version control', 'version control systems']
#         }

#     def extract_text_from_pdf(self, pdf_path: str) -> str:
#         """Extract text content from PDF file."""
#         try:
#             with open(pdf_path, 'rb') as file:
#                 reader = pypdf.PdfReader(file)
#                 text = ""
#                 for page in reader.pages:
#                     text += page.extract_text()
#                 return text.strip()
#         except Exception as e:
#             raise Exception(f"Error reading PDF: {str(e)}")

#     def _get_context(self, text: str, match) -> str:
#         """Helper method to extract context around a regex match."""
#         start_pos = max(0, match.start() - 30)
#         end_pos = min(len(text), match.end() + 30)
#         return text[start_pos:end_pos].replace('\n', ' ')

#     def _extract_section(self, text: str, keywords: List[str]) -> str:
#         """Extract specific sections from text based on keywords."""
#         text_lines = text.split('\n')
#         section_text = ""
#         capturing = False

#         for line in text_lines:
#             line_lower = line.lower().strip()

#             # Check if this line starts a target section
#             if any(keyword in line_lower for keyword in keywords):
#                 capturing = True
#                 section_text += line + "\n"
#                 continue

#             # Stop capturing if we hit another major section
#             if capturing and any(stop_word in line_lower for stop_word in
#                                ['education', 'experience', 'contact', 'about', 'summary']):
#                 if not any(keyword in line_lower for keyword in keywords):
#                     break

#             if capturing:
#                 section_text += line + "\n"

#         return section_text.strip()

#     def parse_job_requirements(self, jd_text: str) -> Dict:
#         """Parse job description to extract requirements and keywords."""
#         # Store original JD text for context analysis
#         self._original_jd_text = jd_text
#         jd_lower = jd_text.lower()

#         # Extract sections
#         sections = {
#             'required': self._extract_section(jd_text, ['requirements', 'required', 'must have']),
#             'preferred': self._extract_section(jd_text, ['preferred', 'nice to have', 'bonus']),
#             'responsibilities': self._extract_section(jd_text, ['responsibilities', 'duties', 'role'])
#         }

#         print(f"\nDEBUG - JD Analysis:")
#         print(f"Required section length: {len(sections['required']) if sections['required'] else 0}")
#         print(f"Preferred section length: {len(sections['preferred']) if sections['preferred'] else 0}")

#         # ONLY include skills that are EXPLICITLY mentioned in the JD text using whole-word matching
#         import re

#         # Create expanded skill matching - check for related terms too
#         skill_expansion_map = {
#             'agents': ['ai agents', 'agentic systems', 'agentic workflows', 'intelligent agents'],
#             'agentic systems': ['ai agents', 'agents', 'agentic workflows', 'autonomous agents'],
#             'agentic workflows': ['ai agents', 'agents', 'agentic systems', 'workflow orchestration'],
#             'orchestration frameworks': ['ai orchestration', 'workflow orchestration', 'llm orchestration'],
#             'machine learning': ['ml', 'machine learning', 'predictive modeling'],
#             'artificial intelligence': ['ai', 'artificial intelligence', 'generative ai'],
#         }

#         all_skills = []
#         required_skills = []
#         preferred_skills = []

#         for category, skills_list in self.tech_skills.items():
#             for skill in skills_list:
#                 # Use regex for whole word matching to avoid false positives
#                 pattern = None
#                 if len(skill.split()) > 1:
#                     # Multi-word skill - check for exact phrase
#                     pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')
#                 else:
#                     # Single word skill - use word boundary matching
#                     pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')

#                 match = pattern.search(jd_lower)
#                 skill_found = False

#                 # First try direct match
#                 if match:
#                     skill_found = True
#                     context = self._get_context(jd_lower, match)
#                     print(f"  Found skill '{skill}' directly in context: ...{context}...")

#                 # If not found directly, try expanded/related terms
#                 elif skill in skill_expansion_map:
#                     for related_term in skill_expansion_map[skill]:
#                         related_pattern = re.compile(r'\b' + re.escape(related_term.lower()) + r'\b')
#                         related_match = related_pattern.search(jd_lower)
#                         if related_match:
#                             skill_found = True
#                             context = self._get_context(jd_lower, related_match)
#                             print(f"  Found skill '{skill}' via related term '{related_term}' in context: ...{context}...")
#                             break

#                 if skill_found:
#                     all_skills.append(skill)

#                     # Determine if it's required or preferred based on which section mentions it
#                     skill_in_required = sections['required'] and (
#                         pattern.search(sections['required'].lower()) if match else
#                         any(re.compile(r'\b' + re.escape(term.lower()) + r'\b').search(sections['required'].lower())
#                             for term in skill_expansion_map.get(skill, [skill]))
#                     )
#                     skill_in_preferred = sections['preferred'] and (
#                         pattern.search(sections['preferred'].lower()) if match else
#                         any(re.compile(r'\b' + re.escape(term.lower()) + r'\b').search(sections['preferred'].lower())
#                             for term in skill_expansion_map.get(skill, [skill]))
#                     )

#                     if skill_in_required:
#                         required_skills.append(skill)
#                         print(f"    → Categorized as REQUIRED")
#                     elif skill_in_preferred:
#                         preferred_skills.append(skill)
#                         print(f"    → Categorized as PREFERRED")
#                     else:
#                         # Check responsibilities section
#                         if sections['responsibilities'] and (
#                             pattern.search(sections['responsibilities'].lower()) if match else
#                             any(re.compile(r'\b' + re.escape(term.lower()) + r'\b').search(sections['responsibilities'].lower())
#                                 for term in skill_expansion_map.get(skill, [skill]))
#                         ):
#                             required_skills.append(skill)
#                             print(f"    → Found in responsibilities (marking as REQUIRED)")
#                         else:
#                             # Found in JD but unclear context - be conservative
#                             required_skills.append(skill)
#                             print(f"    → Found in JD (defaulting to REQUIRED)")

#         # Remove duplicates while preserving order
#         all_skills = list(dict.fromkeys(all_skills))
#         required_skills = list(dict.fromkeys(required_skills))
#         preferred_skills = list(dict.fromkeys(preferred_skills))

#         print(f"Total skills mentioned in JD: {len(all_skills)}")
#         print(f"Final required skills: {required_skills}")
#         print(f"Final preferred skills: {preferred_skills}")

#         return {
#             'required_skills': required_skills,
#             'preferred_skills': preferred_skills,
#             'all_skills': all_skills,
#             'sections': sections
#         }

#     def analyze_resume(self, resume_text: str) -> Dict:
#         """Analyze resume to extract skills, experience, and other relevant information."""
#         resume_lower = resume_text.lower()

#         # Extract skills by category using whole-word matching
#         import re
#         found_skills = defaultdict(list)

#         # Special handling for problematic single-character skills
#         problematic_single_chars = ['r', 'c', 'go']

#         # Create expanded skill matching for resume too
#         resume_skill_expansion = {
#             'agentic systems': ['agentic workflows', 'ai agents', 'agents'],
#             'agentic workflows': ['agentic systems', 'ai agents', 'workflow orchestration'],
#             'ai agents': ['agentic systems', 'agentic workflows', 'agents'],
#             'orchestration frameworks': ['ai orchestration', 'workflow orchestration', 'langchain', 'crewai'],
#             'generative ai': ['generative artificial intelligence', 'gen ai', 'ai'],
#         }

#         for category, skills_list in self.tech_skills.items():
#             for skill in skills_list:
#                 # Skip problematic single-character skills
#                 if skill.lower() in problematic_single_chars:
#                     if skill.lower() == 'r':
#                         # Look for R programming context
#                         r_contexts = [
#                             r'\br\s+programming\b', r'\br\s+language\b', r'\br\s+studio\b',
#                             r'\br\s+packages\b', r'\brstudio\b', r'\bggplot\b', r'\bdplyr\b'
#                         ]
#                         found_r = False
#                         for context_pattern in r_contexts:
#                             if re.search(context_pattern, resume_lower, re.IGNORECASE):
#                                 found_r = True
#                                 print(f"  Found R programming language in appropriate context")
#                                 break

#                         if found_r:
#                             found_skills[category].append(skill)
#                         continue
#                     continue

#                 # Use regex for whole word matching
#                 pattern = None
#                 if len(skill.split()) > 1:
#                     pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')
#                 else:
#                     pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')

#                 match = pattern.search(resume_lower)
#                 skill_found = False

#                 # First try direct match
#                 if match:
#                     skill_found = True
#                     context = self._get_context(resume_lower, match)
#                     print(f"  Found resume skill '{skill}' directly in context: ...{context}...")

#                 # If not found directly, try expanded/related terms
#                 elif skill in resume_skill_expansion:
#                     for related_term in resume_skill_expansion[skill]:
#                         related_pattern = re.compile(r'\b' + re.escape(related_term.lower()) + r'\b')
#                         related_match = related_pattern.search(resume_lower)
#                         if related_match:
#                             skill_found = True
#                             context = self._get_context(resume_lower, related_match)
#                             print(f"  Found resume skill '{skill}' via related term '{related_term}' in context: ...{context}...")
#                             break

#                 if skill_found:
#                     found_skills[category].append(skill)

#         # Debug print
#         all_resume_skills = [skill for skills_list in found_skills.values() for skill in skills_list]
#         print(f"\nDEBUG - Resume Skills Found: {all_resume_skills}")

#         return {
#             'skills_by_category': dict(found_skills),
#             'all_skills': all_resume_skills,
#             'projects': [],  # Simplified for now
#             'education': [],  # Simplified for now
#             'certifications': []  # Simplified for now
#         }

#     def _analyze_skill_context_in_jd(self, skill: str, jd_text: str) -> Dict:
#         """Analyze how a skill is mentioned in JD - specifically required or as example/option."""
#         jd_lower = jd_text.lower()
#         skill_lower = skill.lower()

#         # Find all occurrences of the skill with context
#         import re
#         pattern = re.compile(r'\b' + re.escape(skill_lower) + r'\b')
#         contexts = []

#         for match in pattern.finditer(jd_lower):
#             start_pos = max(0, match.start() - 50)
#             end_pos = min(len(jd_lower), match.end() + 50)
#             context = jd_lower[start_pos:end_pos].replace('\n', ' ')
#             contexts.append(context)

#         # Analyze context to determine if it's specific requirement or example/option
#         is_example_context = False
#         is_specific_requirement = False

#         for context in contexts:
#             # Patterns indicating it's an example/option
#             example_patterns = [
#                 r'such as.*' + re.escape(skill_lower),
#                 r'like.*' + re.escape(skill_lower),
#                 r'including.*' + re.escape(skill_lower),
#                 r'e\.g\..*' + re.escape(skill_lower),
#                 r'for example.*' + re.escape(skill_lower),
#                 r'\(' + re.escape(skill_lower) + r'.*?\)',
#                 re.escape(skill_lower) + r'.*,.*\)',
#             ]

#             for pattern in example_patterns:
#                 if re.search(pattern, context):
#                     is_example_context = True
#                     break

#             # Patterns indicating specific requirement
#             specific_patterns = [
#                 r'experience.{0,20}' + re.escape(skill_lower),
#                 r'proficiency.{0,20}' + re.escape(skill_lower),
#                 r'knowledge.{0,20}' + re.escape(skill_lower),
#                 r'must.{0,30}' + re.escape(skill_lower),
#                 r'required.{0,30}' + re.escape(skill_lower),
#                 r'need.{0,20}' + re.escape(skill_lower),
#             ]

#             for pattern in specific_patterns:
#                 if re.search(pattern, context) and not is_example_context:
#                     is_specific_requirement = True
#                     break

#         return {
#             'is_specific_requirement': is_specific_requirement,
#             'is_example_context': is_example_context,
#             'contexts': contexts
#         }

#     def _calculate_skills_match(self, resume_skills: List[str], required_skills: List[str]) -> float:
#         """Calculate percentage of required skills found in resume with semantic matching."""
#         if not required_skills:
#             print("No required skills to match - giving 100% score")
#             return 100

#         print(f"\nDEBUG - Skills Matching:")
#         print(f"Resume skills: {resume_skills}")
#         print(f"Required skills: {required_skills}")

#         # Create semantic inference rules
#         semantic_inference_map = {
#             'cloud platforms': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
#             'cloud computing': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
#             'version control': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
#             'version control systems': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
#             'ai orchestration': ['langchain', 'crewai', 'orchestration frameworks', 'workflow orchestration'],
#             'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'generative ai'],
#             'machine learning': ['ml', 'scikit-learn', 'tensorflow', 'pytorch', 'keras'],
#             'data manipulation': ['pandas', 'numpy', 'sql', 'data analysis', 'data engineering'],
#             'data analysis': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi'],
#             'web development': ['fastapi', 'flask', 'django', 'react', 'nodejs'],
#             'api development': ['fastapi', 'flask', 'rest api', 'restful api'],
#             'devops practices': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'devops'],
#             'containerization': ['docker', 'kubernetes'],
#         }

#         # Cloud platform equivalency
#         cloud_equivalents = ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services']

#         resume_skills_lower = [skill.lower() for skill in resume_skills]
#         matched_skills = []

#         for req_skill in required_skills:
#             skill_matched = False

#             # First try direct match
#             if req_skill.lower() in resume_skills_lower:
#                 matched_skills.append(req_skill)
#                 skill_matched = True
#                 print(f"  ✅ MATCHED (Direct): '{req_skill}' found in resume")

#             # Special handling for cloud platforms
#             elif req_skill.lower() in cloud_equivalents:
#                 context_analysis = self._analyze_skill_context_in_jd(req_skill, self._original_jd_text)

#                 if context_analysis['is_example_context']:
#                     user_cloud_skills = [skill for skill in resume_skills_lower if skill in cloud_equivalents]
#                     if user_cloud_skills:
#                         matched_skills.append(req_skill)
#                         skill_matched = True
#                         print(f"  ✅ MATCHED (Cloud Equivalent): '{req_skill}' satisfied by user's cloud experience: {user_cloud_skills}")
#                     else:
#                         print(f"  ❌ MISSING (Cloud Specific): '{req_skill}' mentioned as example but user has no cloud platforms")
#                 else:
#                     print(f"  ❌ MISSING (Cloud Specific): '{req_skill}' specifically required, not found in resume")

#             # Try semantic inference
#             elif req_skill.lower() in semantic_inference_map:
#                 specific_skills = semantic_inference_map[req_skill.lower()]
#                 found_specific = [skill for skill in specific_skills if skill in resume_skills_lower]

#                 if found_specific:
#                     matched_skills.append(req_skill)
#                     skill_matched = True
#                     print(f"  ✅ MATCHED (Inferred): '{req_skill}' inferred from resume skills: {found_specific}")

#             if not skill_matched:
#                 print(f"  ❌ MISSING: '{req_skill}' NOT found in resume (neither directly nor through inference)")

#         match_percentage = (len(matched_skills) / len(required_skills)) * 100
#         print(f"Skills match: {len(matched_skills)}/{len(required_skills)} = {match_percentage}%")
#         print(f"Final matched skills: {matched_skills}")

#         return match_percentage

#     def _calculate_keyword_coverage(self, resume_skills: List[str], jd_skills: List[str]) -> float:
#         """Calculate keyword coverage based on JD skill requirements only."""
#         if not jd_skills:
#             print("No JD skills to match - giving 100% keyword score")
#             return 100

#         resume_skills_set = set([skill.lower() for skill in resume_skills])
#         jd_skills_set = set([skill.lower() for skill in jd_skills])

#         # Calculate coverage of JD keywords only
#         matches = len(resume_skills_set & jd_skills_set)
#         jd_coverage = (matches / len(jd_skills_set)) * 100

#         print(f"Keyword coverage: {matches}/{len(jd_skills_set)} JD skills = {jd_coverage:.1f}%")

#         return jd_coverage

#     def calculate_ats_score(self, resume_analysis: Dict, jd_requirements: Dict) -> Dict:
#         """Calculate comprehensive ATS score based on multiple factors."""

#         # Calculate base percentages (0-100)
#         required_skills_pct = self._calculate_skills_match(
#             resume_analysis['all_skills'],
#             jd_requirements['required_skills']
#         )

#         preferred_skills_pct = self._calculate_skills_match(
#             resume_analysis['all_skills'],
#             jd_requirements['preferred_skills']
#         )

#         keyword_match_pct = self._calculate_keyword_coverage(
#             resume_analysis['all_skills'],
#             jd_requirements['all_skills']
#         )

#         semantic_similarity_pct = 85  # Improved semantic score baseline

#         # Apply weights correctly (convert weights to fractions)
#         required_skills_score = (required_skills_pct / 100) * self.weights['required_skills']
#         preferred_skills_score = (preferred_skills_pct / 100) * self.weights['preferred_skills']
#         keyword_score = (keyword_match_pct / 100) * self.weights['keyword_match']
#         semantic_score = (semantic_similarity_pct / 100) * self.weights['semantic_similarity']

#         total_score = (required_skills_score + preferred_skills_score +
#                       keyword_score + semantic_score)

#         # Generate detailed breakdown
#         breakdown = self._generate_detailed_breakdown(
#             resume_analysis, jd_requirements,
#             required_skills_pct, preferred_skills_pct,
#             keyword_match_pct, semantic_similarity_pct
#         )

#         return {
#             'total_score': min(100, round(total_score, 1)),
#             'breakdown': breakdown,
#             'component_scores': {
#                 'required_skills': f"{round(required_skills_score, 1)}/{self.weights['required_skills']}",
#                 'preferred_skills': f"{round(preferred_skills_score, 1)}/{self.weights['preferred_skills']}",
#                 'keyword_match': f"{round(keyword_score, 1)}/{self.weights['keyword_match']}",
#                 'semantic_similarity': f"{round(semantic_score, 1)}/{self.weights['semantic_similarity']}"
#             },
#             'component_percentages': {
#                 'required_skills': round(required_skills_pct, 1),
#                 'preferred_skills': round(preferred_skills_pct, 1),
#                 'keyword_match': round(keyword_match_pct, 1),
#                 'semantic_similarity': round(semantic_similarity_pct, 1)
#             }
#         }

#     def _generate_detailed_breakdown(self, resume_analysis: Dict, jd_requirements: Dict,
#                                    req_pct: float, pref_pct: float, kw_pct: float,
#                                    sem_pct: float) -> Dict:
#         """Generate detailed analysis breakdown."""

#         # Find matched and missing skills using semantic inference logic
#         resume_skills_lower = [skill.lower() for skill in resume_analysis['all_skills']]
#         print(f"\nDEBUG - Breakdown Calculation:")
#         print(f"Resume skills (lowercase): {resume_skills_lower}")
#         print(f"JD required skills: {jd_requirements['required_skills']}")

#         # Use semantic inference map
#         semantic_inference_map = {
#             'cloud platforms': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
#             'cloud computing': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
#             'version control': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
#             'version control systems': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
#             'ai orchestration': ['langchain', 'crewai', 'orchestration frameworks', 'workflow orchestration'],
#             'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'generative ai'],
#             'machine learning': ['ml', 'scikit-learn', 'tensorflow', 'pytorch', 'keras'],
#             'data manipulation': ['pandas', 'numpy', 'sql', 'data analysis', 'data engineering'],
#             'data analysis': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi'],
#             'web development': ['fastapi', 'flask', 'django', 'react', 'nodejs'],
#             'api development': ['fastapi', 'flask', 'rest api', 'restful api'],
#             'devops practices': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'devops'],
#             'containerization': ['docker', 'kubernetes'],
#         }

#         cloud_equivalents = ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services']

#         matched_required = []
#         missing_required = []

#         for skill in jd_requirements['required_skills']:
#             skill_matched = False

#             # Try direct match first
#             if skill.lower() in resume_skills_lower:
#                 matched_required.append(skill)
#                 skill_matched = True
#                 print(f"  ✅ Required skill MATCHED (Direct): {skill}")

#             # Special cloud platform handling
#             elif skill.lower() in cloud_equivalents:
#                 context_analysis = self._analyze_skill_context_in_jd(skill, self._original_jd_text)

#                 if context_analysis['is_example_context']:
#                     user_cloud_skills = [s for s in resume_skills_lower if s in cloud_equivalents]
#                     if user_cloud_skills:
#                         matched_required.append(skill)
#                         skill_matched = True
#                         print(f"  ✅ Required skill MATCHED (Cloud Equivalent): {skill} satisfied by {user_cloud_skills}")
#                     else:
#                         print(f"  ❌ Required skill MISSING (Cloud): {skill} - user has no cloud platforms")
#                 else:
#                     print(f"  ❌ Required skill MISSING (Cloud Specific): {skill} - specifically required")

#             # Try semantic inference
#             elif skill.lower() in semantic_inference_map:
#                 specific_skills = semantic_inference_map[skill.lower()]
#                 found_specific = [s for s in specific_skills if s in resume_skills_lower]

#                 if found_specific:
#                     matched_required.append(skill)
#                     skill_matched = True
#                     print(f"  ✅ Required skill MATCHED (Inferred): {skill} from {found_specific}")

#             if not skill_matched:
#                 missing_required.append(skill)
#                 print(f"  ❌ Required skill MISSING: {skill}")

#         # Same logic for preferred skills
#         matched_preferred = []
#         missing_preferred = []

#         for skill in jd_requirements['preferred_skills']:
#             skill_matched = False

#             if skill.lower() in resume_skills_lower:
#                 matched_preferred.append(skill)
#                 skill_matched = True
#             elif skill.lower() in cloud_equivalents:
#                 context_analysis = self._analyze_skill_context_in_jd(skill, self._original_jd_text)
#                 if context_analysis['is_example_context']:
#                     user_cloud_skills = [s for s in resume_skills_lower if s in cloud_equivalents]
#                     if user_cloud_skills:
#                         matched_preferred.append(skill)
#                         skill_matched = True
#             elif skill.lower() in semantic_inference_map:
#                 specific_skills = semantic_inference_map[skill.lower()]
#                 found_specific = [s for s in specific_skills if s in resume_skills_lower]
#                 if found_specific:
#                     matched_preferred.append(skill)
#                     skill_matched = True

#             if not skill_matched:
#                 missing_preferred.append(skill)

#         print(f"Final matched required: {matched_required}")
#         print(f"Final missing required: {missing_required}")

#         return {
#             'strong_matches': matched_required + matched_preferred,
#             'missing_required': missing_required,
#             'missing_preferred': missing_preferred,
#             'skill_analysis': {
#                 'total_resume_skills': len(resume_analysis['all_skills']),
#                 'total_jd_skills': len(jd_requirements['all_skills']),
#                 'skills_matched': len(matched_required + matched_preferred)
#             },
#             'recommendations': self._generate_recommendations(missing_required, missing_preferred)
#         }

#     def _generate_recommendations(self, missing_required: List[str], missing_preferred: List[str]) -> List[str]:
#         """Generate improvement recommendations."""
#         recommendations = []

#         if missing_required:
#             recommendations.append(f"Add missing required skills: {', '.join(missing_required[:3])}")

#         if missing_preferred:
#             recommendations.append(f"Consider adding preferred skills: {', '.join(missing_preferred[:3])}")

#         recommendations.extend([
#             "Use exact keywords from the job description",
#             "Quantify achievements with specific metrics",
#             "Tailor your summary to match the role requirements",
#             "Include relevant project details that demonstrate required skills"
#         ])

#         return recommendations

#     def generate_report(self, resume_path: str, jd_path: str) -> Dict:
#         """Generate comprehensive ATS report for resume and job description."""

#         # Extract text from PDFs
#         resume_text = self.extract_text_from_pdf(resume_path)
#         jd_text = self.extract_text_from_pdf(jd_path)

#         # Analyze resume and job requirements
#         resume_analysis = self.analyze_resume(resume_text)
#         jd_requirements = self.parse_job_requirements(jd_text)

#         # Calculate ATS score
#         ats_results = self.calculate_ats_score(resume_analysis, jd_requirements)

#         return {
#             'ats_score': ats_results['total_score'],
#             'component_scores': ats_results['component_scores'],
#             'detailed_analysis': ats_results['breakdown'],
#             'resume_summary': {
#                 'total_skills': len(resume_analysis['all_skills']),
#                 'skills_by_category': resume_analysis['skills_by_category'],
#                 'experience_months': 0,
#                 'projects_count': len(resume_analysis['projects']),
#                 'certifications_count': len(resume_analysis['certifications'])
#             },
#             'job_summary': {
#                 'required_skills_count': len(jd_requirements['required_skills']),
#                 'preferred_skills_count': len(jd_requirements['preferred_skills']),
#                 'total_keywords': len(jd_requirements['all_skills']),
#                 'experience_required_months': 0
#             },
#             'jd_keyword_analysis': {
#                 'required_skills': jd_requirements['required_skills'],
#                 'required_count': len(jd_requirements['required_skills']),
#                 'preferred_skills': jd_requirements['preferred_skills'],
#                 'preferred_count': len(jd_requirements['preferred_skills']),
#                 'all_keywords': jd_requirements['all_skills']
#             },
#             'matching_details': {
#                 'matched_keywords': ats_results['breakdown']['strong_matches'],
#                 'missing_required': ats_results['breakdown']['missing_required'],
#                 'missing_preferred': ats_results['breakdown']['missing_preferred'],
#                 'keyword_match_rate': (len(ats_results['breakdown']['strong_matches']) / len(jd_requirements['all_skills']) * 100) if jd_requirements['all_skills'] else 0
#             }
#         }


# # Enhanced main function with clean output
# def main():
#     """Example usage of the ATS Scorer with clean output."""

#     # Initialize the ATS scorer
#     ats_scorer = ATSScorer()

#     # Example with file paths
#     try:
#         # Generate comprehensive ATS report
#         report = ats_scorer.generate_report("/Users/arreyanhamid/Developer/ai-resume/rendercv_output/Arreyan_Hamid_CV.pdf", '/Users/arreyanhamid/Developer/ai-resume/JD/JD_Genzeon.pdf')


#         # Clean formatted output
#         jd_analysis = report.get('jd_keyword_analysis', {})

#         print("\n" + "="*80)
#         print(f"ATS ANALYSIS REPORT")
#         print("="*80)
#         print(f"OVERALL ATS SCORE: {report['ats_score']}/100")
#         print(f"COMPONENT BREAKDOWN:")
#         for component, score in report['component_scores'].items():
#             print(f"  {component.replace('_', ' ').title()}: {score}")

#         print(f"\nSUMMARY STATISTICS:")
#         print(f"  Keywords in JD: {report['job_summary']['total_keywords']}")
#         print(f"  Keywords in Resume: {report['resume_summary']['total_skills']}")
#         print(f"  Keywords Matched: {len(report['detailed_analysis']['strong_matches'])}")
#         print(f"  Match Rate: {(len(report['detailed_analysis']['strong_matches']) / report['job_summary']['total_keywords'] * 100):.1f}%")

#         print(f"\nJOB DESCRIPTION KEYWORDS:")
#         if jd_analysis:
#             print(f"  Required Skills: {jd_analysis.get('required_count', 0)} keywords")
#             for skill in jd_analysis.get('required_skills', [])[:10]:
#                 print(f"    • {skill}")

#             print(f"  Preferred Skills: {jd_analysis.get('preferred_count', 0)} keywords")
#             for skill in jd_analysis.get('preferred_skills', [])[:10]:
#                 print(f"    • {skill}")

#         print(f"\nRESUME KEYWORDS DETECTED:")
#         resume_skills = report['resume_summary'].get('skills_by_category', {})
#         for category, skills in resume_skills.items():
#             if skills:
#                 print(f"  {category.replace('_', ' ').title()}: {len(skills)} keywords")
#                 for skill in skills[:5]:
#                     print(f"    • {skill}")

#         print(f"\nMATCHED KEYWORDS:")
#         matched = report['detailed_analysis']['strong_matches']
#         print(f"  Total Matches: {len(matched)}")
#         for i, match in enumerate(matched, 1):
#             print(f"    {i:2d}. {match}")

#         print(f"\nMISSING KEYWORDS (from JD):")
#         missing_req = report['detailed_analysis']['missing_required']
#         missing_pref = report['detailed_analysis']['missing_preferred']

#         if missing_req:
#             print(f"  Missing Required ({len(missing_req)}):")
#             for i, missing in enumerate(missing_req, 1):
#                 print(f"    {i:2d}. {missing}")
#         else:
#             print(f"  Missing Required: None ✓")

#         if missing_pref:
#             print(f"  Missing Preferred ({len(missing_pref)}):")
#             for i, missing in enumerate(missing_pref, 1):
#                 print(f"    {i:2d}. {missing}")
#         else:
#             print(f"  Missing Preferred: None ✓")

#         print(f"\nUNMATCHED RESUME KEYWORDS:")
#         # Find keywords in resume that weren't in JD
#         jd_keywords = set([skill.lower() for skill in jd_analysis.get('all_keywords', [])])

#         # Get all resume skills flattened
#         all_resume_skills = []
#         for category_skills in report['resume_summary']['skills_by_category'].values():
#             all_resume_skills.extend(category_skills)

#         unmatched_resume = [skill for skill in all_resume_skills
#                            if skill.lower() not in jd_keywords]

#         if unmatched_resume:
#             print(f"  Resume Keywords NOT in JD ({len(unmatched_resume)}):")
#             for i, skill in enumerate(unmatched_resume, 1):
#                 print(f"    {i:2d}. {skill}")
#         else:
#             print(f"  Resume Keywords NOT in JD: None (perfect JD alignment!)")

#         print(f"\nSKILL COVERAGE ANALYSIS:")
#         skill_analysis = report['detailed_analysis']['skill_analysis']
#         print(f"  Resume Skills: {skill_analysis['total_resume_skills']}")
#         print(f"  JD Skills: {skill_analysis['total_jd_skills']}")
#         print(f"  Skills Matched: {skill_analysis['skills_matched']}")
#         print(f"  Coverage Rate: {(skill_analysis['skills_matched'] / skill_analysis['total_jd_skills'] * 100):.1f}%")

#         print(f"\nRECOMMENDATIONS:")
#         for i, rec in enumerate(report['detailed_analysis']['recommendations'][:5], 1):
#             print(f"  {i}. {rec}")

#         print("\n" + "="*80)

#         # Save detailed report to JSON
#         with open('ats_report.json', 'w') as f:
#             json.dump(report, f, indent=2)
#         print("Detailed report saved to 'ats_report.json'")

#     except Exception as e:
#         print(f"Error: {e}")
#         print("Please ensure PDF files exist and are readable.")

# if __name__ == "__main__":
#     main()

# # Installation requirements (add to requirements.txt):
# """
# pypdf==3.0.1
# scikit-learn==1.3.0
# pandas==2.0.3
# numpy==1.24.3
# """

import pypdf
import re
from collections import defaultdict
import json
from typing import Dict, List, Tuple

class ATSScorer:
    def __init__(self):
        """Initialize the ATS Scorer with predefined skill categories and weights."""
        self.weights = {
            'required_skills': 40,      # 40% weight for required skills
            'preferred_skills': 20,     # 20% weight for preferred skills
            'keyword_match': 20,        # 20% weight for keyword matching
            'contact_formatting': 10,   # 10% weight for contact info and formatting
            'semantic_similarity': 10   # 10% weight for semantic similarity
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

    def _analyze_contact_and_formatting(self, resume_text: str) -> Dict:
        """Analyze contact information and resume formatting."""
        import re

        resume_lower = resume_text.lower()
        contact_analysis = {
            'email': False,
            'phone': False,
            'linkedin': False,
            'sections_present': [],
            'formatting_score': 0
        }

        # Check for email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, resume_text, re.IGNORECASE):
            contact_analysis['email'] = True
            print("  ✅ Email address found")
        else:
            print("  ❌ No email address found")

        # FIXED PHONE DETECTION - specifically for Indian numbers starting with 0
        phone_patterns = [
            r'0\d{5}\s*\d{2}\s*\d{3}',         # Your format: 098218 70330 (optional spaces)
            r'0\d{5}\s+\d{2}\s+\d{3}',         # Your format: 098218 70330 (required spaces)
            r'0\d{5}\s{1,5}\d{2}\s{1,5}\d{3}', # Handle multiple spaces (PDF parsing issue)
            r'0\d{2,5}[-\s]\d{2,4}[-\s]\d{3,4}', # Flexible Indian landline format
            r'[6-9]\d{9}',                      # Indian mobile (10 digits)
            r'[6-9]\d{4}\s*\d{5}',             # Indian mobile with space
            r'\d{6}\s+\d{2}\s+\d{3}',          # 6-2-3 format (any starting digit)
            r'\d{10}',                          # Any 10-digit number
            r'\+\d{1,3}[-\s]?\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}', # International
            r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',    # US format
            r'\(\d{3}\)\s?\d{3}[-.\s]\d{4}',   # US format with parentheses
        ]

        for pattern in phone_patterns:
            if re.search(pattern, resume_text):
                contact_analysis['phone'] = True
                print("  ✅ Phone number found")
                break

        if not contact_analysis['phone']:
            print("  ❌ No phone number found")

        # Check for LinkedIn
        linkedin_indicators = ['linkedin', 'linkedin.com', 'in/', '/in/', 'linkedin profile']
        for indicator in linkedin_indicators:
            if indicator in resume_lower:
                contact_analysis['linkedin'] = True
                print("  ✅ LinkedIn profile found")
                break
        if not contact_analysis['linkedin']:
            print("  ❌ No LinkedIn profile found")

        # Rest of the function remains the same...
        section_keywords = {
            'experience': ['experience', 'work experience', 'employment', 'work history', 'professional experience'],
            'education': ['education', 'academic', 'degree', 'university', 'college'],
            'skills': ['skills', 'technical skills', 'abilities', 'competencies', 'technologies'],
            'projects': ['projects', 'personal projects', 'portfolio', 'work samples'],
            'summary': ['summary', 'profile', 'objective', 'about', 'professional summary']
        }

        for section_name, keywords in section_keywords.items():
            section_found = any(keyword in resume_lower for keyword in keywords)
            if section_found:
                contact_analysis['sections_present'].append(section_name)
                print(f"  ✅ {section_name.title()} section found")
            else:
                print(f"  ❌ {section_name.title()} section not found")

        # Calculate formatting score
        contact_score = 0
        if contact_analysis['email']: contact_score += 15
        if contact_analysis['phone']: contact_score += 15
        if contact_analysis['linkedin']: contact_score += 10

        section_score = 0
        essential_sections = ['experience', 'education', 'skills']
        for section in essential_sections:
            if section in contact_analysis['sections_present']:
                section_score += 20

        bonus_sections = ['projects', 'summary']
        for section in bonus_sections:
            if section in contact_analysis['sections_present']:
                section_score += 5

        contact_analysis['formatting_score'] = min(100, contact_score + section_score)

        return contact_analysis
    def parse_job_requirements(self, jd_text: str) -> Dict:
        """Parse job description to extract requirements and keywords."""
        self._original_jd_text = jd_text
        jd_lower = jd_text.lower()

        sections = {
            'required': self._extract_section(jd_text, ['requirements', 'required', 'must have']),
            'preferred': self._extract_section(jd_text, ['preferred', 'nice to have', 'bonus']),
            'responsibilities': self._extract_section(jd_text, ['responsibilities', 'duties', 'role'])
        }

        print(f"\nDEBUG - JD Analysis:")
        print(f"Required section length: {len(sections['required']) if sections['required'] else 0}")
        print(f"Preferred section length: {len(sections['preferred']) if sections['preferred'] else 0}")

        import re

        skill_expansion_map = {
            'agents': ['ai agents', 'agentic systems', 'agentic workflows', 'intelligent agents'],
            'agentic systems': ['ai agents', 'agents', 'agentic workflows', 'autonomous agents'],
            'agentic workflows': ['ai agents', 'agents', 'agentic systems', 'workflow orchestration'],
            'orchestration frameworks': ['ai orchestration', 'workflow orchestration', 'llm orchestration'],
            'machine learning': ['ml', 'machine learning', 'predictive modeling'],
            'artificial intelligence': ['ai', 'artificial intelligence', 'generative ai'],
        }

        all_skills = []
        required_skills = []
        preferred_skills = []

        for category, skills_list in self.tech_skills.items():
            for skill in skills_list:
                pattern = None
                if len(skill.split()) > 1:
                    pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')
                else:
                    pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')

                match = pattern.search(jd_lower)
                skill_found = False

                if match:
                    skill_found = True
                    context = self._get_context(jd_lower, match)
                    print(f"  Found skill '{skill}' directly in context: ...{context}...")

                elif skill in skill_expansion_map:
                    for related_term in skill_expansion_map[skill]:
                        related_pattern = re.compile(r'\b' + re.escape(related_term.lower()) + r'\b')
                        related_match = related_pattern.search(jd_lower)
                        if related_match:
                            skill_found = True
                            context = self._get_context(jd_lower, related_match)
                            print(f"  Found skill '{skill}' via related term '{related_term}' in context: ...{context}...")
                            break

                if skill_found:
                    all_skills.append(skill)

                    skill_in_required = sections['required'] and (
                        pattern.search(sections['required'].lower()) if match else
                        any(re.compile(r'\b' + re.escape(term.lower()) + r'\b').search(sections['required'].lower())
                            for term in skill_expansion_map.get(skill, [skill]))
                    )
                    skill_in_preferred = sections['preferred'] and (
                        pattern.search(sections['preferred'].lower()) if match else
                        any(re.compile(r'\b' + re.escape(term.lower()) + r'\b').search(sections['preferred'].lower())
                            for term in skill_expansion_map.get(skill, [skill]))
                    )

                    if skill_in_required:
                        required_skills.append(skill)
                        print(f"    → Categorized as REQUIRED")
                    elif skill_in_preferred:
                        preferred_skills.append(skill)
                        print(f"    → Categorized as PREFERRED")
                    else:
                        if sections['responsibilities'] and (
                            pattern.search(sections['responsibilities'].lower()) if match else
                            any(re.compile(r'\b' + re.escape(term.lower()) + r'\b').search(sections['responsibilities'].lower())
                                for term in skill_expansion_map.get(skill, [skill]))
                        ):
                            required_skills.append(skill)
                            print(f"    → Found in responsibilities (marking as REQUIRED)")
                        else:
                            required_skills.append(skill)
                            print(f"    → Found in JD (defaulting to REQUIRED)")

        all_skills = list(dict.fromkeys(all_skills))
        required_skills = list(dict.fromkeys(required_skills))
        preferred_skills = list(dict.fromkeys(preferred_skills))

        print(f"Total skills mentioned in JD: {len(all_skills)}")
        print(f"Final required skills: {required_skills}")
        print(f"Final preferred skills: {preferred_skills}")

        return {
            'required_skills': required_skills,
            'preferred_skills': preferred_skills,
            'all_skills': all_skills,
            'sections': sections
        }

    def analyze_resume(self, resume_text: str) -> Dict:
        """Analyze resume to extract skills, experience, and other relevant information."""
        resume_lower = resume_text.lower()

        print(f"\n=== RESUME ANALYSIS ===")
        print(f"Contact Information & Formatting:")

        contact_formatting = self._analyze_contact_and_formatting(resume_text)

        print(f"\nSkill Detection:")

        import re
        found_skills = defaultdict(list)

        problematic_single_chars = ['r', 'c', 'go']

        resume_skill_expansion = {
            'agentic systems': ['agentic workflows', 'ai agents', 'agents'],
            'agentic workflows': ['agentic systems', 'ai agents', 'workflow orchestration'],
            'ai agents': ['agentic systems', 'agentic workflows', 'agents'],
            'orchestration frameworks': ['ai orchestration', 'workflow orchestration', 'langchain', 'crewai'],
            'generative ai': ['generative artificial intelligence', 'gen ai', 'ai'],
        }

        for category, skills_list in self.tech_skills.items():
            for skill in skills_list:
                if skill.lower() in problematic_single_chars:
                    if skill.lower() == 'r':
                        r_contexts = [
                            r'\br\s+programming\b', r'\br\s+language\b', r'\br\s+studio\b',
                            r'\br\s+packages\b', r'\brstudio\b', r'\bggplot\b', r'\bdplyr\b'
                        ]
                        found_r = False
                        for context_pattern in r_contexts:
                            if re.search(context_pattern, resume_lower, re.IGNORECASE):
                                found_r = True
                                print(f"  Found R programming language in appropriate context")
                                break

                        if found_r:
                            found_skills[category].append(skill)
                        continue
                    continue

                pattern = None
                if len(skill.split()) > 1:
                    pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')
                else:
                    pattern = re.compile(r'\b' + re.escape(skill.lower()) + r'\b')

                match = pattern.search(resume_lower)
                skill_found = False

                if match:
                    skill_found = True
                    context = self._get_context(resume_lower, match)
                    print(f"  Found resume skill '{skill}' directly in context: ...{context}...")

                elif skill in resume_skill_expansion:
                    for related_term in resume_skill_expansion[skill]:
                        related_pattern = re.compile(r'\b' + re.escape(related_term.lower()) + r'\b')
                        related_match = related_pattern.search(resume_lower)
                        if related_match:
                            skill_found = True
                            context = self._get_context(resume_lower, related_match)
                            print(f"  Found resume skill '{skill}' via related term '{related_term}' in context: ...{context}...")
                            break

                if skill_found:
                    found_skills[category].append(skill)

        all_resume_skills = [skill for skills_list in found_skills.values() for skill in skills_list]
        print(f"\nDEBUG - Resume Skills Found: {all_resume_skills}")

        return {
            'skills_by_category': dict(found_skills),
            'all_skills': all_resume_skills,
            'contact_formatting': contact_formatting,
            'projects': [],
            'education': [],
            'certifications': []
        }

    def _analyze_skill_context_in_jd(self, skill: str, jd_text: str) -> Dict:
        """Analyze how a skill is mentioned in JD."""
        jd_lower = jd_text.lower()
        skill_lower = skill.lower()

        import re
        pattern = re.compile(r'\b' + re.escape(skill_lower) + r'\b')
        contexts = []

        for match in pattern.finditer(jd_lower):
            start_pos = max(0, match.start() - 50)
            end_pos = min(len(jd_lower), match.end() + 50)
            context = jd_lower[start_pos:end_pos].replace('\n', ' ')
            contexts.append(context)

        is_example_context = False
        is_specific_requirement = False

        for context in contexts:
            example_patterns = [
                r'such as.*' + re.escape(skill_lower),
                r'like.*' + re.escape(skill_lower),
                r'including.*' + re.escape(skill_lower),
                r'e\.g\..*' + re.escape(skill_lower),
                r'for example.*' + re.escape(skill_lower),
                r'\(' + re.escape(skill_lower) + r'.*?\)',
                re.escape(skill_lower) + r'.*,.*\)',
            ]

            for pattern in example_patterns:
                if re.search(pattern, context):
                    is_example_context = True
                    break

            specific_patterns = [
                r'experience.{0,20}' + re.escape(skill_lower),
                r'proficiency.{0,20}' + re.escape(skill_lower),
                r'knowledge.{0,20}' + re.escape(skill_lower),
                r'must.{0,30}' + re.escape(skill_lower),
                r'required.{0,30}' + re.escape(skill_lower),
                r'need.{0,20}' + re.escape(skill_lower),
            ]

            for pattern in specific_patterns:
                if re.search(pattern, context) and not is_example_context:
                    is_specific_requirement = True
                    break

        return {
            'is_specific_requirement': is_specific_requirement,
            'is_example_context': is_example_context,
            'contexts': contexts
        }

    def _calculate_skills_match(self, resume_skills: List[str], required_skills: List[str]) -> float:
        """Calculate percentage of required skills found in resume with semantic matching."""
        if not required_skills:
            print("No required skills to match - giving 100% score")
            return 100

        print(f"\nDEBUG - Skills Matching:")
        print(f"Resume skills: {resume_skills}")
        print(f"Required skills: {required_skills}")

        semantic_inference_map = {
            'cloud platforms': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
            'cloud computing': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
            'version control': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
            'version control systems': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
            'ai orchestration': ['langchain', 'crewai', 'orchestration frameworks', 'workflow orchestration'],
            'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'generative ai'],
            'machine learning': ['ml', 'scikit-learn', 'tensorflow', 'pytorch', 'keras'],
            'data manipulation': ['pandas', 'numpy', 'sql', 'data analysis', 'data engineering'],
            'data analysis': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi'],
            'web development': ['fastapi', 'flask', 'django', 'react', 'nodejs'],
            'api development': ['fastapi', 'flask', 'rest api', 'restful api'],
            'devops practices': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'devops'],
            'containerization': ['docker', 'kubernetes'],
        }

        cloud_equivalents = ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services']

        resume_skills_lower = [skill.lower() for skill in resume_skills]
        matched_skills = []

        for req_skill in required_skills:
            skill_matched = False

            if req_skill.lower() in resume_skills_lower:
                matched_skills.append(req_skill)
                skill_matched = True
                print(f"  ✅ MATCHED (Direct): '{req_skill}' found in resume")

            elif req_skill.lower() in cloud_equivalents:
                context_analysis = self._analyze_skill_context_in_jd(req_skill, self._original_jd_text)

                if context_analysis['is_example_context']:
                    user_cloud_skills = [skill for skill in resume_skills_lower if skill in cloud_equivalents]
                    if user_cloud_skills:
                        matched_skills.append(req_skill)
                        skill_matched = True
                        print(f"  ✅ MATCHED (Cloud Equivalent): '{req_skill}' satisfied by user's cloud experience: {user_cloud_skills}")
                    else:
                        print(f"  ❌ MISSING (Cloud Specific): '{req_skill}' mentioned as example but user has no cloud platforms")
                else:
                    print(f"  ❌ MISSING (Cloud Specific): '{req_skill}' specifically required, not found in resume")

            elif req_skill.lower() in semantic_inference_map:
                specific_skills = semantic_inference_map[req_skill.lower()]
                found_specific = [skill for skill in specific_skills if skill in resume_skills_lower]

                if found_specific:
                    matched_skills.append(req_skill)
                    skill_matched = True
                    print(f"  ✅ MATCHED (Inferred): '{req_skill}' inferred from resume skills: {found_specific}")

            if not skill_matched:
                print(f"  ❌ MISSING: '{req_skill}' NOT found in resume (neither directly nor through inference)")

        match_percentage = (len(matched_skills) / len(required_skills)) * 100
        print(f"Skills match: {len(matched_skills)}/{len(required_skills)} = {match_percentage}%")
        print(f"Final matched skills: {matched_skills}")

        return match_percentage

    def _calculate_keyword_coverage(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        """Calculate keyword coverage based on JD skill requirements only."""
        if not jd_skills:
            print("No JD skills to match - giving 100% keyword score")
            return 100

        resume_skills_set = set([skill.lower() for skill in resume_skills])
        jd_skills_set = set([skill.lower() for skill in jd_skills])

        matches = len(resume_skills_set & jd_skills_set)
        jd_coverage = (matches / len(jd_skills_set)) * 100

        print(f"Keyword coverage: {matches}/{len(jd_skills_set)} JD skills = {jd_coverage:.1f}%")

        return jd_coverage

    def calculate_ats_score(self, resume_analysis: Dict, jd_requirements: Dict) -> Dict:
        """Calculate comprehensive ATS score based on multiple factors."""

        required_skills_pct = self._calculate_skills_match(
            resume_analysis['all_skills'],
            jd_requirements['required_skills']
        )

        preferred_skills_pct = self._calculate_skills_match(
            resume_analysis['all_skills'],
            jd_requirements['preferred_skills']
        )

        keyword_match_pct = self._calculate_keyword_coverage(
            resume_analysis['all_skills'],
            jd_requirements['all_skills']
        )

        contact_formatting_pct = resume_analysis['contact_formatting']['formatting_score']

        semantic_similarity_pct = 85

        required_skills_score = (required_skills_pct / 100) * self.weights['required_skills']
        preferred_skills_score = (preferred_skills_pct / 100) * self.weights['preferred_skills']
        keyword_score = (keyword_match_pct / 100) * self.weights['keyword_match']
        contact_formatting_score = (contact_formatting_pct / 100) * self.weights['contact_formatting']
        semantic_score = (semantic_similarity_pct / 100) * self.weights['semantic_similarity']

        total_score = (required_skills_score + preferred_skills_score +
                      keyword_score + contact_formatting_score + semantic_score)

        breakdown = self._generate_detailed_breakdown(
            resume_analysis, jd_requirements,
            required_skills_pct, preferred_skills_pct,
            keyword_match_pct, contact_formatting_pct, semantic_similarity_pct
        )

        return {
            'total_score': min(100, round(total_score, 1)),
            'breakdown': breakdown,
            'component_scores': {
                'required_skills': f"{round(required_skills_score, 1)}/{self.weights['required_skills']}",
                'preferred_skills': f"{round(preferred_skills_score, 1)}/{self.weights['preferred_skills']}",
                'keyword_match': f"{round(keyword_score, 1)}/{self.weights['keyword_match']}",
                'contact_formatting': f"{round(contact_formatting_score, 1)}/{self.weights['contact_formatting']}",
                'semantic_similarity': f"{round(semantic_score, 1)}/{self.weights['semantic_similarity']}"
            },
            'component_percentages': {
                'required_skills': round(required_skills_pct, 1),
                'preferred_skills': round(preferred_skills_pct, 1),
                'keyword_match': round(keyword_match_pct, 1),
                'contact_formatting': round(contact_formatting_pct, 1),
                'semantic_similarity': round(semantic_similarity_pct, 1)
            }
        }

    def _generate_detailed_breakdown(self, resume_analysis: Dict, jd_requirements: Dict,
                                   req_pct: float, pref_pct: float, kw_pct: float,
                                   cf_pct: float, sem_pct: float) -> Dict:
        """Generate detailed analysis breakdown."""

        resume_skills_lower = [skill.lower() for skill in resume_analysis['all_skills']]
        print(f"\nDEBUG - Breakdown Calculation:")
        print(f"Resume skills (lowercase): {resume_skills_lower}")
        print(f"JD required skills: {jd_requirements['required_skills']}")

        semantic_inference_map = {
            'cloud platforms': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
            'cloud computing': ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services'],
            'version control': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
            'version control systems': ['git', 'github', 'gitlab', 'svn', 'mercurial'],
            'ai orchestration': ['langchain', 'crewai', 'orchestration frameworks', 'workflow orchestration'],
            'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'generative ai'],
            'machine learning': ['ml', 'scikit-learn', 'tensorflow', 'pytorch', 'keras'],
            'data manipulation': ['pandas', 'numpy', 'sql', 'data analysis', 'data engineering'],
            'data analysis': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi'],
            'web development': ['fastapi', 'flask', 'django', 'react', 'nodejs'],
            'api development': ['fastapi', 'flask', 'rest api', 'restful api'],
            'devops practices': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'devops'],
            'containerization': ['docker', 'kubernetes'],
        }

        cloud_equivalents = ['aws', 'azure', 'google cloud', 'gcp', 'microsoft azure', 'amazon web services']

        matched_required = []
        missing_required = []

        for skill in jd_requirements['required_skills']:
            skill_matched = False

            if skill.lower() in resume_skills_lower:
                matched_required.append(skill)
                skill_matched = True
                print(f"  ✅ Required skill MATCHED (Direct): {skill}")

            elif skill.lower() in cloud_equivalents:
                context_analysis = self._analyze_skill_context_in_jd(skill, self._original_jd_text)

                if context_analysis['is_example_context']:
                    user_cloud_skills = [s for s in resume_skills_lower if s in cloud_equivalents]
                    if user_cloud_skills:
                        matched_required.append(skill)
                        skill_matched = True
                        print(f"  ✅ Required skill MATCHED (Cloud Equivalent): {skill} satisfied by {user_cloud_skills}")
                    else:
                        print(f"  ❌ Required skill MISSING (Cloud): {skill} - user has no cloud platforms")
                else:
                    print(f"  ❌ Required skill MISSING (Cloud Specific): {skill} - specifically required")

            elif skill.lower() in semantic_inference_map:
                specific_skills = semantic_inference_map[skill.lower()]
                found_specific = [s for s in specific_skills if s in resume_skills_lower]

                if found_specific:
                    matched_required.append(skill)
                    skill_matched = True
                    print(f"  ✅ Required skill MATCHED (Inferred): {skill} from {found_specific}")

            if not skill_matched:
                missing_required.append(skill)
                print(f"  ❌ Required skill MISSING: {skill}")

        # Same logic for preferred skills
        matched_preferred = []
        missing_preferred = []

        for skill in jd_requirements['preferred_skills']:
            skill_matched = False

            if skill.lower() in resume_skills_lower:
                matched_preferred.append(skill)
                skill_matched = True
            elif skill.lower() in cloud_equivalents:
                context_analysis = self._analyze_skill_context_in_jd(skill, self._original_jd_text)
                if context_analysis['is_example_context']:
                    user_cloud_skills = [s for s in resume_skills_lower if s in cloud_equivalents]
                    if user_cloud_skills:
                        matched_preferred.append(skill)
                        skill_matched = True
            elif skill.lower() in semantic_inference_map:
                specific_skills = semantic_inference_map[skill.lower()]
                found_specific = [s for s in specific_skills if s in resume_skills_lower]
                if found_specific:
                    matched_preferred.append(skill)
                    skill_matched = True

            if not skill_matched:
                missing_preferred.append(skill)

        print(f"Final matched required: {matched_required}")
        print(f"Final missing required: {missing_required}")

        return {
            'strong_matches': matched_required + matched_preferred,
            'missing_required': missing_required,
            'missing_preferred': missing_preferred,
            'contact_formatting_analysis': resume_analysis['contact_formatting'],
            'skill_analysis': {
                'total_resume_skills': len(resume_analysis['all_skills']),
                'total_jd_skills': len(jd_requirements['all_skills']),
                'skills_matched': len(matched_required + matched_preferred)
            },
            'recommendations': self._generate_recommendations(missing_required, missing_preferred, resume_analysis['contact_formatting'])
        }

    def _generate_recommendations(self, missing_required: List[str], missing_preferred: List[str], contact_formatting: Dict) -> List[str]:
        """Generate improvement recommendations based on actual gaps."""
        recommendations = []

        # Only add skill recommendations if there are actually missing skills
        if missing_required:
            recommendations.append(f"Add missing required skills: {', '.join(missing_required[:3])}")

        if missing_preferred:
            recommendations.append(f"Consider adding preferred skills: {', '.join(missing_preferred[:3])}")

        # Contact/formatting recommendations - only if actually missing
        contact_issues = []
        if not contact_formatting['email']:
            contact_issues.append("email address")
        if not contact_formatting['phone']:
            contact_issues.append("phone number")
        if not contact_formatting['linkedin']:
            contact_issues.append("LinkedIn profile")

        if contact_issues:
            recommendations.append(f"Add missing contact info: {', '.join(contact_issues)}")

        # Section recommendations - only if actually missing
        missing_sections = []
        essential_sections = ['experience', 'education', 'skills']
        for section in essential_sections:
            if section not in contact_formatting['sections_present']:
                missing_sections.append(section)

        if missing_sections:
            recommendations.append(f"Add missing essential sections: {', '.join(missing_sections)}")

        # If no major issues, provide advanced recommendations
        if not missing_required and not missing_preferred and not contact_issues and not missing_sections:
            recommendations.extend([
                "Excellent keyword alignment! Consider adding more specific technology versions (e.g., 'Python 3.9+', 'TensorFlow 2.x')",
                "Add more quantified achievements with specific metrics",
                "Include relevant certifications or training programs",
                "Consider adding a brief professional summary if not present",
                "Ensure consistent verb tenses throughout (past tense for previous roles)",
                "Add action verbs to bullet points (Developed, Implemented, Optimized, etc.)"
            ])
        else:
            # Standard recommendations for improvement
            recommendations.extend([
                "Use action verbs to start bullet points (Developed, Implemented, Created)",
                "Ensure consistent formatting and verb tenses throughout resume",
                "Consider adding relevant certifications or professional development"
            ])

        return recommendations[:8]  # Limit to top 8 recommendations
    def generate_report(self, resume_path: str, jd_path: str) -> Dict:
        """Generate comprehensive ATS report for resume and job description."""

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
            'detailed_analysis': ats_results['breakdown'],
            'resume_summary': {
                'total_skills': len(resume_analysis['all_skills']),
                'skills_by_category': resume_analysis['skills_by_category'],
                'contact_formatting': resume_analysis['contact_formatting'],
                'experience_months': 0,
                'projects_count': len(resume_analysis['projects']),
                'certifications_count': len(resume_analysis['certifications'])
            },
            'job_summary': {
                'required_skills_count': len(jd_requirements['required_skills']),
                'preferred_skills_count': len(jd_requirements['preferred_skills']),
                'total_keywords': len(jd_requirements['all_skills']),
                'experience_required_months': 0
            },
            'jd_keyword_analysis': {
                'required_skills': jd_requirements['required_skills'],
                'required_count': len(jd_requirements['required_skills']),
                'preferred_skills': jd_requirements['preferred_skills'],
                'preferred_count': len(jd_requirements['preferred_skills']),
                'all_keywords': jd_requirements['all_skills']
            },
            'matching_details': {
                'matched_keywords': ats_results['breakdown']['strong_matches'],
                'missing_required': ats_results['breakdown']['missing_required'],
                'missing_preferred': ats_results['breakdown']['missing_preferred'],
                'keyword_match_rate': (len(ats_results['breakdown']['strong_matches']) / len(jd_requirements['all_skills']) * 100) if jd_requirements['all_skills'] else 0
            }
        }


# Enhanced main function with comprehensive clean output
def main():
    """Example usage of the ATS Scorer with detailed, clean output."""

    # Initialize the ATS scorer
    ats_scorer = ATSScorer()

    try:
        # Generate comprehensive ATS report
        report = ats_scorer.generate_report('/Users/arreyanhamid/Developer/ai-resume/rendercv_output/Garv_Khurana_CV.pdf', '/Users/arreyanhamid/Developer/ai-resume/JD/AIJD.pdf')

        # Clean formatted output
        jd_analysis = report.get('jd_keyword_analysis', {})
        contact_analysis = report['detailed_analysis']['contact_formatting_analysis']

        print("\n" + "="*100)
        print(f"{'ATS COMPREHENSIVE ANALYSIS REPORT':^100}")
        print("="*100)

        print(f"\n📊 OVERALL ATS SCORE: {report['ats_score']}/100")
        print("\n" + "-"*100)
        print(f"{'COMPONENT BREAKDOWN':^100}")
        print("-"*100)
        for component, score in report['component_scores'].items():
            component_name = component.replace('_', ' ').title()
            print(f"  {component_name:<25}: {score}")

        print("\n" + "-"*100)
        print(f"{'CONTACT INFORMATION & FORMATTING ANALYSIS':^100}")
        print("-"*100)
        print(f"📧 Email Address        : {'✅ Present' if contact_analysis['email'] else '❌ Missing'}")
        print(f"📱 Phone Number        : {'✅ Present' if contact_analysis['phone'] else '❌ Missing'}")
        print(f"🔗 LinkedIn Profile    : {'✅ Present' if contact_analysis['linkedin'] else '❌ Missing'}")

        print(f"\n📋 Resume Sections:")
        essential_sections = ['experience', 'education', 'skills']
        bonus_sections = ['projects', 'summary']

        for section in essential_sections:
            status = '✅ Present' if section in contact_analysis['sections_present'] else '❌ Missing (CRITICAL)'
            print(f"  {section.title():<15}: {status}")

        for section in bonus_sections:
            if section in contact_analysis['sections_present']:
                print(f"  {section.title():<15}: ✅ Present (Bonus)")

        print(f"\n📈 Formatting Score    : {contact_analysis['formatting_score']}/100")

        print("\n" + "-"*100)
        print(f"{'JOB DESCRIPTION REQUIREMENTS ANALYSIS':^100}")
        print("-"*100)

        if jd_analysis:
            print(f"🎯 Required Skills ({jd_analysis.get('required_count', 0)} total):")
            for i, skill in enumerate(jd_analysis.get('required_skills', []), 1):
                status = '✅ MATCHED' if skill in report['detailed_analysis']['strong_matches'] else '❌ MISSING'
                print(f"  {i:2d}. {skill:<25} : {status}")

            print(f"\n⭐ Preferred Skills ({jd_analysis.get('preferred_count', 0)} total):")
            for i, skill in enumerate(jd_analysis.get('preferred_skills', []), 1):
                status = '✅ MATCHED' if skill in report['detailed_analysis']['strong_matches'] else '❌ MISSING'
                print(f"  {i:2d}. {skill:<25} : {status}")

        print("\n" + "-"*100)
        print(f"{'RESUME SKILLS DETECTED BY CATEGORY':^100}")
        print("-"*100)
        resume_skills = report['resume_summary'].get('skills_by_category', {})
        for category, skills in resume_skills.items():
            if skills:
                print(f"🔧 {category.replace('_', ' ').title()} ({len(skills)} skills):")
                for skill in skills:
                    match_status = '✅' if skill in report['detailed_analysis']['strong_matches'] else '⚪'
                    print(f"    {match_status} {skill}")
                print()

        print("-"*100)
        print(f"{'SKILL MATCHING SUMMARY':^100}")
        print("-"*100)
        skill_analysis = report['detailed_analysis']['skill_analysis']
        print(f"📊 Total Resume Skills : {skill_analysis['total_resume_skills']}")
        print(f"🎯 Total JD Skills     : {skill_analysis['total_jd_skills']}")
        print(f"✅ Skills Matched      : {skill_analysis['skills_matched']}")
        print(f"📈 Coverage Rate       : {(skill_analysis['skills_matched'] / skill_analysis['total_jd_skills'] * 100):.1f}%")

        # Show unmatched resume skills
        jd_keywords = set([skill.lower() for skill in jd_analysis.get('all_keywords', [])])
        all_resume_skills = []
        for category_skills in report['resume_summary']['skills_by_category'].values():
            all_resume_skills.extend(category_skills)

        unmatched_resume = [skill for skill in all_resume_skills
                           if skill.lower() not in jd_keywords]

        if unmatched_resume:
            print(f"\n⚪ Additional Skills Not in JD ({len(unmatched_resume)}):")
            for skill in unmatched_resume[:15]:  # Show first 15
                print(f"    • {skill}")
            if len(unmatched_resume) > 15:
                print(f"    ... and {len(unmatched_resume) - 15} more")

        print("\n" + "-"*100)
        print(f"{'RECOMMENDATIONS FOR IMPROVEMENT':^100}")
        print("-"*100)
        for i, rec in enumerate(report['detailed_analysis']['recommendations'], 1):
            print(f"  {i:2d}. {rec}")

        print("\n" + "="*100)
        print(f"{'END OF REPORT':^100}")
        print("="*100)

        # Save detailed report to JSON
        with open('ats_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Detailed JSON report saved to 'ats_report.json'")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure PDF files exist and are readable.")

if __name__ == "__main__":
    main()

# Installation requirements (add to requirements.txt):
"""
pypdf==3.0.1
pandas==2.0.3
numpy==1.24.3
"""