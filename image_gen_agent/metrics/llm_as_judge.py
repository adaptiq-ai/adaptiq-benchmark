from openai import OpenAI
import json
import re
from typing import Dict, Any
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class CriteriaScore(BaseModel):
    """Individual criteria scoring details"""
    adaptiq_rating: str = Field(..., description="Rating for Adaptiq method (High/Medium/Low)")
    adaptiq_reasoning: str = Field(..., description="Detailed reasoning for Adaptiq rating")
    gpt_rating: str = Field(..., description="Rating for GPT method (High/Medium/Low)")
    gpt_reasoning: str = Field(..., description="Detailed reasoning for GPT rating")
    winner: str = Field(..., description="Winner for this criteria (Adaptiq/GPT)")
    winner_justification: str = Field(..., description="Justification for the winner")

class OverallScores(BaseModel):
    """Overall scores for both methods"""
    adaptiq_score: float = Field(..., ge=0, le=100, description="Adaptiq overall score (0-100)")
    gpt_score: float = Field(..., ge=0, le=100, description="GPT overall score (0-100)")

class EvaluationResult(BaseModel):
    """Structure to hold evaluation results"""
    criteria_scores: Dict[str, CriteriaScore]
    overall_scores: OverallScores
    winner: str = Field(..., description="Overall winner (Adaptiq/GPT)")
    detailed_table: str = Field(..., description="Formatted evaluation table")
    summary: Optional[str] = Field(None, description="Summary of evaluation")

class ErrorHandlingScore(BaseModel):
    """Error handling assessment for a method"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    identified_vulnerabilities: List[str]
    strengths: List[str]

class OutputQuality(BaseModel):
    """Output quality metrics"""
    consistency_score: float = Field(..., ge=0, le=100)
    instruction_following_score: float = Field(..., ge=0, le=100)
    completeness_score: float = Field(..., ge=0, le=100)
    coherence_score: float = Field(..., ge=0, le=100)
    analysis: str

class MethodErrorAssessment(BaseModel):
    """Complete error assessment for one method"""
    prompt_error_handling: ErrorHandlingScore
    output_quality: OutputQuality
    error_patterns: List[str]
    overall_reliability_score: float = Field(..., ge=0, le=100)

class ComparativeAnalysis(BaseModel):
    """Comparative analysis between methods"""
    winner: str
    error_rate_difference: float
    key_differences: List[str]
    recommendations: List[str]

class ErrorSummary(BaseModel):
    """Summary of error assessment"""
    adaptiq_error_rate: float = Field(..., ge=0, le=100)
    gpt_error_rate: float = Field(..., ge=0, le=100)
    most_reliable_method: str
    critical_findings: str

class StructuredErrorScores(BaseModel):
    """Structured scores for error assessment"""
    adaptiq: Dict[str, float]
    gpt: Dict[str, float]
    winner: str
    winner_by_reliability: str

class ErrorAssessmentResult(BaseModel):
    """Complete error assessment result"""
    error_assessment: Dict[str, MethodErrorAssessment]
    comparative_analysis: ComparativeAnalysis
    summary: ErrorSummary
    formatted_report: str
    error_scores: StructuredErrorScores

class LLMAsJudgeScore:
    """
    LLM-as-Judge scoring system for prompt enhancement evaluation.
    Compares Adaptiq vs GPT methods across multiple criteria.
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1"):
        """
        Initialize the LLM Judge with GPT model.
        
        Args:
            api_key: API key for OpenAI
            model_name: Gemini model name (default: "gpt-4.1")
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name  # Store model name for later use
        
        # Evaluation criteria with descriptions
        self.evaluation_criteria = {
            "Prompt Refinement Precision": {
                "description": "How precisely the method incorporates feedback into specific parts of the prompt",
                "weight": 0.25
            },
            "Structural Integrity & Reliability": {
                "description": "How well-structured and reliable the prompt format is for consistent outputs",
                "weight": 0.25
            },
            "Learning Efficiency Metric": {
                "description": "How efficiently the method learns and converges to optimal prompts",
                "weight": 0.20
            },
            "Adaptability & Debuggability": {
                "description": "How easy it is to debug and adapt the prompt when issues arise",
                "weight": 0.15
            },
            "Final Result Effectiveness": {
                "description": "How effectively the refined prompt produces the desired outcome",
                "weight": 0.15
            }
        }
    
    def create_evaluation_prompt(self, original_prompt: str, adaptiq_prompt: str, 
                                gpt_prompt: str, enhancement_feedback: str) -> str:
        """
        Create a comprehensive evaluation prompt for the LLM judge.
        
        Args:
            original_prompt: The initial prompt before optimization
            adaptiq_prompt: Prompt optimized using Adaptiq method
            gpt_prompt: Prompt optimized using GPT method
            enhancement_feedback: The feedback/requirements for enhancement
            
        Returns:
            Formatted evaluation prompt
        """
        
        criteria_descriptions = "\n".join([
            f"- **{criteria}**: {info['description']}"
            for criteria, info in self.evaluation_criteria.items()
        ])
        
        evaluation_prompt = f"""
        You are an expert LLM evaluator tasked with comparing two prompt optimization methods: Adaptiq vs GPT method.

        **ORIGINAL PROMPT:**
        ```
        {original_prompt}
        ```

        **ENHANCEMENT FEEDBACK GIVEN:**
        ```
        {enhancement_feedback}
        ```

        **ADAPTIQ OPTIMIZED PROMPT:**
        ```
        {adaptiq_prompt}
        ```

        **GPT METHOD OPTIMIZED PROMPT:**
        ```
        {gpt_prompt}
        ```

        **EVALUATION CRITERIA:**
        {criteria_descriptions}

        **INSTRUCTIONS:**
        1. Evaluate both methods across all criteria listed above
        2. For each criterion, provide:
        - Detailed assessment of Adaptiq method (High/Medium/Low rating + reasoning)
        - Detailed assessment of GPT method (High/Medium/Low rating + reasoning)  
        - Winner determination with clear justification
        3. Provide overall scores (0-100) for each method
        4. Determine the overall winner

        **OUTPUT FORMAT:**
        Return your evaluation in the following JSON structure:

        ```json
        {{
            "evaluation_table": {{
                "Prompt Refinement Precision": {{
                    "adaptiq_rating": "High/Medium/Low",
                    "adaptiq_reasoning": "Detailed explanation...",
                    "gpt_rating": "High/Medium/Low", 
                    "gpt_reasoning": "Detailed explanation...",
                    "winner": "Adaptiq/GPT",
                    "winner_justification": "Why this method wins..."
                }},
                "Structural Integrity & Reliability": {{
                    "adaptiq_rating": "High/Medium/Low",
                    "adaptiq_reasoning": "Detailed explanation...",
                    "gpt_rating": "High/Medium/Low",
                    "gpt_reasoning": "Detailed explanation...", 
                    "winner": "Adaptiq/GPT",
                    "winner_justification": "Why this method wins..."
                }},
                "Learning Efficiency Metric": {{
                    "adaptiq_rating": "High/Medium/Low",
                    "adaptiq_reasoning": "Detailed explanation...",
                    "gpt_rating": "High/Medium/Low",
                    "gpt_reasoning": "Detailed explanation...",
                    "winner": "Adaptiq/GPT", 
                    "winner_justification": "Why this method wins..."
                }},
                "Adaptability & Debuggability": {{
                    "adaptiq_rating": "High/Medium/Low",
                    "adaptiq_reasoning": "Detailed explanation...",
                    "gpt_rating": "High/Medium/Low",
                    "gpt_reasoning": "Detailed explanation...",
                    "winner": "Adaptiq/GPT",
                    "winner_justification": "Why this method wins..."
                }},
                "Final Result Effectiveness": {{
                    "adaptiq_rating": "High/Medium/Low", 
                    "adaptiq_reasoning": "Detailed explanation...",
                    "gpt_rating": "High/Medium/Low",
                    "gpt_reasoning": "Detailed explanation...",
                    "winner": "Adaptiq/GPT",
                    "winner_justification": "Why this method wins..."
                }}
            }},
            "overall_scores": {{
                "adaptiq_score": 85,
                "gpt_score": 72
            }},
            "overall_winner": "Adaptiq/GPT",
            "summary": "Brief summary of why the winner is better overall..."
        }}
        ```

        Be thorough, objective, and provide specific examples from the prompts to support your assessments.
        """
        return evaluation_prompt
    
    def prompt_quality_score(self, original_prompt: str, adaptiq_prompt: str, gpt_prompt: str, enhancement_feedback: str) -> EvaluationResult:
        """
        Main method to evaluate and score prompt quality between Adaptiq and GPT methods.
        
        Args:
            original_prompt: The initial prompt before optimization
            adaptiq_prompt: Prompt optimized using Adaptiq method
            gpt_prompt: Prompt optimized using GPT method
            enhancement_feedback: The feedback/requirements for enhancement
        
        Returns:
            EvaluationResult object containing scores, winner, and detailed analysis
        """
        try:
            # Create evaluation prompt
            eval_prompt = self.create_evaluation_prompt(
                original_prompt, adaptiq_prompt, gpt_prompt, enhancement_feedback
            )
            
            # Get evaluation from Gemini
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": eval_prompt}]  # or error_evaluation_prompt
            )
                
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                evaluation_data = json.loads(json_match.group(1))
                
                # Convert to Pydantic models
                criteria_scores = {
                    criteria: CriteriaScore(**details)
                    for criteria, details in evaluation_data["evaluation_table"].items()
                }
                overall_scores = OverallScores(**evaluation_data["overall_scores"])
            else:
                print("Could not extract JSON from LLM response")
                return None
                
            # Generate detailed table
            detailed_table = self._generate_detailed_table(evaluation_data)
            
            # Create result object
            result = EvaluationResult(
                criteria_scores=criteria_scores,
                overall_scores=overall_scores,
                winner=evaluation_data["overall_winner"],
                detailed_table=detailed_table,
                summary=evaluation_data.get("summary")
            )
            
            return result
            
        except Exception as e:
            print(f"Error in prompt quality evaluation: {str(e)}")
            return None
    
    def _generate_detailed_table(self, evaluation_data: Dict) -> str:
        """
        Generate a formatted table from evaluation data.
        
        Args:
            evaluation_data: Parsed JSON evaluation data
            
        Returns:
            Formatted table string
        """
        table_lines = []
        table_lines.append("=" * 120)
        table_lines.append("PROMPT OPTIMIZATION EVALUATION: ADAPTIQ VS GPT METHOD")
        table_lines.append("=" * 120)
        table_lines.append("")
        
        # Header
        table_lines.append(f"{'Criteria & Metric':<35} {'Adaptiq':<25} {'GPT':<25} {'Winner':<15}")
        table_lines.append("-" * 120)
        
        # Evaluation table
        eval_table = evaluation_data["evaluation_table"]
        for criteria, details in eval_table.items():
            adaptiq_summary = f"{details['adaptiq_rating']}: {details['adaptiq_reasoning'][:50]}..."
            gpt_summary = f"{details['gpt_rating']}: {details['gpt_reasoning'][:50]}..."
            
            table_lines.append(f"{criteria:<35} {adaptiq_summary:<25} {gpt_summary:<25} {details['winner']:<15}")
            table_lines.append("")
        
        # Overall scores
        table_lines.append("-" * 120)
        scores = evaluation_data["overall_scores"]
        table_lines.append(f"{'OVERALL SCORES':<35} {scores['adaptiq_score']:<25} {scores['gpt_score']:<25} {evaluation_data['overall_winner']:<15}")
        table_lines.append("=" * 120)
        
        # Summary
        table_lines.append(f"\nSUMMARY: {evaluation_data.get('summary', 'N/A')}")
        
        return "\n".join(table_lines)
    
    def get_scoring_breakdown(self, result: EvaluationResult) -> Dict[str, Any]:
        """
        Get detailed scoring breakdown with weights applied.
        
        Args:
            result: EvaluationResult from prompt_quality_score
            
        Returns:
            Dictionary with weighted scores and breakdown
        """
        breakdown = {
            "criteria_breakdown": {},
            "weighted_scores": {"adaptiq": 0, "gpt": 0},
            "winner_analysis": {}
        }
        
        # Convert ratings to numeric scores
        rating_scores = {"High": 85, "Medium": 65, "Low": 40}
        
        adaptiq_wins = 0
        gpt_wins = 0
        
        for criteria, details in result.criteria_scores.items():
            adaptiq_score = rating_scores.get(details["adaptiq_rating"], 50)
            gpt_score = rating_scores.get(details["gpt_rating"], 50)
            weight = self.evaluation_criteria[criteria]["weight"]
            
            breakdown["criteria_breakdown"][criteria] = {
                "adaptiq_score": adaptiq_score,
                "gpt_score": gpt_score, 
                "weight": weight,
                "winner": details["winner"]
            }
            
            # Apply weights
            breakdown["weighted_scores"]["adaptiq"] += adaptiq_score * weight
            breakdown["weighted_scores"]["gpt"] += gpt_score * weight
            
            # Count wins
            if details["winner"] == "Adaptiq":
                adaptiq_wins += 1
            else:
                gpt_wins += 1
        
        breakdown["winner_analysis"] = {
            "adaptiq_criteria_wins": adaptiq_wins,
            "gpt_criteria_wins": gpt_wins,
            "overall_winner": result.winner
        }
        
        return breakdown
    
