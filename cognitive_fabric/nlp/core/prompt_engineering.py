from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class PromptTemplate:
    name: str
    template: str
    variables: List[str]
    description: str

class PromptEngineer:
    """
    Advanced prompt engineering for optimal AI responses
    """
    
    def __init__(self):
        self.templates = self._load_default_templates()
        self.prompt_optimization_rules = self._load_optimization_rules()
    
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """Load default prompt templates"""
        return {
            'research_query': PromptTemplate(
                name='research_query',
                template="""You are a research assistant AI. Your task is to provide comprehensive, accurate, and well-structured information about the following topic.

TOPIC: {query}

Please provide:
1. A clear overview of the topic
2. Key concepts and definitions
3. Current state of research/knowledge
4. Important applications or implications
5. Reputable sources or references when available

Ensure your response is factual, balanced, and cites reliable sources when making specific claims.""",
                variables=['query'],
                description='Template for research-oriented queries'
            ),
            
            'technical_explanation': PromptTemplate(
                name='technical_explanation',
                template="""You are a technical expert AI. Explain the following technical concept in clear, accessible language.

CONCEPT: {query}

Your explanation should include:
- A simple definition anyone can understand
- How it works at a high level
- Key components or mechanisms
- Practical examples or analogies
- Common applications or use cases

Avoid unnecessary jargon and focus on making complex concepts understandable.""",
                variables=['query'],
                description='Template for technical explanations'
            ),
            
            'verification_focused': PromptTemplate(
                name='verification_focused',
                template="""You are a verified information AI. Your responses must be based ONLY on verified, reliable information.

QUERY: {query}

CONTEXT FROM VERIFIED SOURCES:
{context}

RESPONSE GUIDELINES:
1. Base your answer SOLELY on the verified context provided
2. If the context doesn't contain relevant information, explicitly state this
3. Do not add any external knowledge or assumptions
4. Cite specific sources when making claims
5. If uncertain, acknowledge the limitations

Your response will be evaluated based on its verifiability against the provided context.""",
                variables=['query', 'context'],
                description='Template for responses requiring high verification'
            ),
            
            'collaborative_synthesis': PromptTemplate(
                name='collaborative_synthesis',
                template="""You are participating in a multi-agent collaborative analysis. Synthesize the following perspectives into a comprehensive response.

ORIGINAL QUERY: {query}

AGENT PERSPECTIVES:
{perspectives}

SYNTHESIS TASK:
- Identify common themes and agreements
- Note important differences or contradictions
- Create a balanced, comprehensive response
- Acknowledge uncertainties or limitations
- Provide the most accurate overall understanding

Your response should represent the collective intelligence of all perspectives.""",
                variables=['query', 'perspectives'],
                description='Template for synthesizing multiple agent perspectives'
            )
        }
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load prompt optimization rules"""
        return {
            'clarity': [
                "Use clear, direct language",
                "Avoid ambiguous terms",
                "Define technical terms when first used"
            ],
            'structure': [
                "Organize information logically",
                "Use headings and bullet points when helpful",
                "Progress from general to specific"
            ],
            'completeness': [
                "Address all aspects of the query",
                "Provide context when necessary",
                "Acknowledge limitations or uncertainties"
            ],
            'verifiability': [
                "Cite sources for factual claims",
                "Distinguish between facts and opinions",
                "Note when information cannot be verified"
            ]
        }
    
    def create_prompt(self, 
                     template_name: str, 
                     variables: Dict[str, str],
                     optimization_level: str = 'standard') -> str:
        """Create a prompt using a template"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        
        # Validate all required variables are provided
        missing_vars = [var for var in template.variables if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing template variables: {missing_vars}")
        
        # Apply template
        prompt = template.template
        for var_name, var_value in variables.items():
            prompt = prompt.replace(f"{{{var_name}}}", var_value)
        
        # Apply optimizations
        if optimization_level != 'none':
            prompt = self._optimize_prompt(prompt, optimization_level)
        
        return prompt
    
    def _optimize_prompt(self, prompt: str, optimization_level: str) -> str:
        """Optimize prompt based on rules"""
        optimized_prompt = prompt
        
        if optimization_level == 'high':
            # Add explicit instructions for high-quality responses
            optimization_suffix = """
            
ADDITIONAL INSTRUCTIONS FOR HIGH-QUALITY RESPONSE:
- Be thorough and comprehensive
- Use evidence-based reasoning
- Consider multiple perspectives
- Acknowledge complexity where it exists
- Provide actionable insights when possible
"""
            optimized_prompt += optimization_suffix
        
        return optimized_prompt
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity to select appropriate template"""
        word_count = len(query.split())
        sentence_count = query.count('.') + query.count('!') + query.count('?')
        
        # Detect question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'describe']
        has_question = any(word in query.lower() for word in question_words)
        
        # Detect technical terms (simplified)
        technical_indicators = ['algorithm', 'protocol', 'architecture', 'framework', 'API', 'database']
        is_technical = any(term in query.lower() for term in technical_indicators)
        
        # Detect research terms
        research_indicators = ['research', 'study', 'analysis', 'evidence', 'data', 'statistics']
        is_research = any(term in query.lower() for term in research_indicators)
        
        complexity_score = min(1.0, (word_count / 50) + (sentence_count / 5) * 0.1)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'has_question': has_question,
            'is_technical': is_technical,
            'is_research': is_research,
            'complexity_score': complexity_score,
            'recommended_template': self._recommend_template(has_question, is_technical, is_research, complexity_score)
        }
    
    def _recommend_template(self, has_question: bool, is_technical: bool, is_research: bool, complexity: float) -> str:
        """Recommend appropriate template based on query analysis"""
        if is_research or complexity > 0.7:
            return 'research_query'
        elif is_technical:
            return 'technical_explanation'
        elif has_question and complexity > 0.3:
            return 'verification_focused'
        else:
            return 'research_query'  # Default
    
    def create_custom_template(self, 
                             name: str, 
                             template: str, 
                             variables: List[str],
                             description: str) -> PromptTemplate:
        """Create a custom prompt template"""
        new_template = PromptTemplate(
            name=name,
            template=template,
            variables=variables,
            description=description
        )
        
        self.templates[name] = new_template
        return new_template
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        
        return {
            'name': template.name,
            'description': template.description,
            'variables': template.variables,
            'example': self._create_template_example(template)
        }
    
    def _create_template_example(self, template: PromptTemplate) -> str:
        """Create an example using the template"""
        example_variables = {}
        for var in template.variables:
            example_variables[var] = f"[Example {var.replace('_', ' ').title()}]"
        
        return self.create_prompt(template.name, example_variables, 'none')
    
    def list_all_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        return [
            {
                'name': template.name,
                'description': template.description,
                'variables': template.variables
            }
            for template in self.templates.values()
        ]

# Global prompt engineer instance
prompt_engineer = PromptEngineer()