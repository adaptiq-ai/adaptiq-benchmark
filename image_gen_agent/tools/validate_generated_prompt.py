from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
import re
from openai import OpenAI
from settings import settings

# Input schema
class PromptValidationInput(BaseModel):
    """Input schema for PromptValidationTool."""
    positive_prompt: str = Field(..., description="positive prompt")
    negative_prompt: str = Field(..., description="negative prompt")

class PromptValidationTool(BaseTool):
    name: str = "validate_prompt"
    description: str = (
        "Validates and sanitizes image generation prompts, removing or rephrasing sensitive terms "
        "to ensure they pass safety filters without losing core creative intent."
    )
    args_schema: Type[BaseModel] = PromptValidationInput

    def _run(self, positive_prompt: str, negative_prompt: str) -> str:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        # GPT system prompt for sanitization
        prompt_template = f"""
        You are a **prompt sanitization engine** for an AI image generator.  
        You must rewrite the positive and negative prompts so they are **guaranteed to pass safety filters**.

        Rules:
        - Remove/rephrase any sexual, nude, erotic, or pornographic content.
        - Remove/rephrase explicit violence, gore, or injury details.
        - Remove/rephrase strong profanity or hate speech.
        - Remove/rephrase political or religiously extreme terms.
        - Keep the artistic style, tone, and overall vibe intact.
        - Err on the side of caution — if a term is borderline, replace it.
        - Replace disallowed terms with safe, thematic synonyms.

        Examples:
        - "naked woman" → "woman in elegant attire"
        - "blood on the floor" → "red paint splattered floor"
        - "sexually suggestive" → "playful pose"
        - "gun pointed at head" → "character holding object"
        - "lolita" → "youthful character"

        Return ONLY in this exact XML format (no commentary, no markdown):
        <prompt>
        <np>{{CLEAN_POSITIVE_PROMPT}}</np>
        <nn>{{CLEAN_NEGATIVE_PROMPT}}</nn>
        </prompt>

        Positive Prompt: {positive_prompt}
        Negative Prompt: {negative_prompt}
        """

        completion = client.responses.create(
            model="gpt-4.1",
            input=prompt_template,
        )

        raw_output = completion.output_text.strip()

        # Regex extraction — tolerant of extra whitespace and line breaks
        match = re.search(
            r"<prompt>\s*<np>(.*?)</np>\s*<nn>(.*?)</nn>\s*</prompt>",
            raw_output,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            clean_positive = match.group(1).strip()
            clean_negative = match.group(2).strip()
            return f"Positive Prompt: {clean_positive}\nNegative Prompt: {clean_negative}"
        else:
            # Fallback if GPT messes up formatting
            return raw_output
