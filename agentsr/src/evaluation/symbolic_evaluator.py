"""
Symbolic Accuracy Evaluator using GPT-4o.

This module implements symbolic equivalence checking between discovered equations
and ground truth equations using GPT-4o as an LLM evaluator, following the approach
from LLM-SRBench.
"""

import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class SymbolicAccuracyEvaluator:
    """
    Evaluator that uses GPT-4o to check symbolic equivalence between equations.

    This evaluator determines if the discovered equation is symbolically equivalent
    to the ground truth equation, possibly with different coefficient values.
    """

    EVALUATION_PROMPT = """Given the ground truth mathematical expression A and the hypothesis B, determine if there exist any constant parameter values that would make the hypothesis equivalent to the given ground truth expression.

Ground Truth Expression (A): {ground_truth}

Hypothesis Expression (B): {hypothesis}

Let's think step by step. Explain your reasoning and then provide the final answer as:
```json
{{
  "reasoning": "brief step-by-step analysis",
  "answer": "yes/no"
}}
```"""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, api_key: Optional[str] = None):
        """
        Initialize the symbolic accuracy evaluator.

        Args:
            model: OpenAI model to use for evaluation (default: gpt-4o)
            temperature: Temperature for LLM sampling (default: 0.0 for deterministic results)
            api_key: OpenAI API key (if None, uses environment variable)
        """
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def evaluate(self, ground_truth: str, hypothesis: str) -> Dict[str, Any]:
        """
        Evaluate symbolic equivalence between ground truth and hypothesis equations.

        Args:
            ground_truth: The ground truth mathematical expression
            hypothesis: The discovered/hypothesis mathematical expression

        Returns:
            Dictionary containing:
                - is_equivalent: bool indicating if equations are symbolically equivalent
                - reasoning: str with step-by-step analysis from GPT-4o
                - raw_response: str with the full LLM response
        """
        prompt = self.EVALUATION_PROMPT.format(
            ground_truth=ground_truth,
            hypothesis=hypothesis
        )

        logger.info(f"Evaluating symbolic equivalence:")
        logger.info(f"  Ground truth: {ground_truth}")
        logger.info(f"  Hypothesis: {hypothesis}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )

            raw_response = response.choices[0].message.content
            logger.debug(f"Raw LLM response: {raw_response}")

            # Parse the JSON response
            result = self._parse_response(raw_response)

            # Add raw response for debugging
            result['raw_response'] = raw_response

            logger.info(f"  Result: {'EQUIVALENT' if result['is_equivalent'] else 'NOT EQUIVALENT'}")
            logger.info(f"  Reasoning: {result['reasoning']}")

            return result

        except Exception as e:
            logger.error(f"Error during symbolic evaluation: {e}")
            return {
                'is_equivalent': False,
                'reasoning': f"Error during evaluation: {str(e)}",
                'raw_response': None,
                'error': str(e)
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract reasoning and answer.

        Args:
            response: Raw response from the LLM

        Returns:
            Dictionary with is_equivalent (bool) and reasoning (str)
        """
        try:
            # Look for JSON code block
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                # Try generic code block
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                # Try to find JSON object directly
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end].strip()

            parsed = json.loads(json_str)

            # Extract answer and reasoning
            answer = parsed.get('answer', '').lower().strip()
            reasoning = parsed.get('reasoning', '')

            # Convert answer to boolean
            is_equivalent = answer in ['yes', 'true', '1']

            return {
                'is_equivalent': is_equivalent,
                'reasoning': reasoning
            }

        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.warning(f"Response was: {response}")

            # Fallback: look for yes/no in the response
            response_lower = response.lower()
            if 'answer: yes' in response_lower or '"answer": "yes"' in response_lower:
                is_equivalent = True
            elif 'answer: no' in response_lower or '"answer": "no"' in response_lower:
                is_equivalent = False
            else:
                # Default to not equivalent if we can't parse
                is_equivalent = False

            return {
                'is_equivalent': is_equivalent,
                'reasoning': 'Failed to parse structured response. Using fallback detection.'
            }
