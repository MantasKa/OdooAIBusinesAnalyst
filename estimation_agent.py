import json
import re
import tiktoken
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner


class OdooEstimationAgent:
    """
    Specialized agent for providing detailed time estimations for Odoo requirements.
    """

    def __init__(self, model_instance, search_tool=None):
        """
        Initialize the estimation agent.

        Args:
            model_instance: The OpenAI model instance to use
            search_tool: Optional search tool for accessing knowledge base
        """
        self.model_instance = model_instance
        self.search_tool = search_tool

        # Initialize tokenizer
        try:
            model_name = getattr(model_instance, 'model', 'gpt-4o-nano')
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except (KeyError, AttributeError):
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Create the agent
        tools = [search_tool] if search_tool else []
        self.agent = Agent(
            name="EstimationSpecialist",
            instructions="""You are a senior Odoo consultant specializing in accurate time estimation.
            You have extensive experience with Odoo implementations and can provide detailed,
            realistic time estimates for various types of requirements.

            When providing estimates:
            1. Break down the work into phases (Analysis, Development, Testing, Deployment)
            2. Consider complexity factors and potential risks
            3. Include confidence levels and assumptions
            4. Provide skill level recommendations
            5. Always respond with valid JSON format

            Expected JSON format:
            {
                "total_estimation_hours": "X-Y hours",
                "confidence_level": "High/Medium/Low",
                "confidence_score": 0.85,
                "complexity_rating": "Simple/Medium/Complex",
                "breakdown": {
                    "analysis_phase": "description and hours",
                    "development_phase": "description and hours",
                    "testing_phase": "description and hours",
                    "deployment_phase": "description and hours"
                },
                "risk_factors": ["risk1", "risk2"],
                "assumptions": ["assumption1", "assumption2"],
                "recommended_approach": "description",
            }
            """,
            model=self.model_instance,
            model_settings=ModelSettings(temperature=0.1),
            tools=tools
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string using tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback: rough estimation (4 chars = 1 token)
            return len(text) // 4

    def _clean_markdown_codeblocks(self, text: str) -> str:
        """
        Remove markdown code block formatting from text.

        Args:
            text (str): Text that may contain markdown code blocks

        Returns:
            str: Cleaned text with markdown formatting removed
        """
        # Remove ```json and ``` markers
        pattern = r'```(?:json)?\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    async def estimate_requirement(self, requirement: str,
                                   implementation_type: str = "unknown") -> dict:
        """
        Provide detailed estimation for a given requirement.

        Args:
            requirement (str): The business requirement to estimate
            implementation_type (str): Type of implementation (configuration/development)

        Returns:
            dict: Detailed estimation breakdown
        """
        estimation_prompt = f"""
        Analyze this Odoo requirement and provide a detailed time estimation:

        Requirement: "{requirement}"
        Implementation Type: {implementation_type}

        Consider the following in your estimation:
        1. Technical complexity and effort required
        2. Integration points with existing Odoo modules
        3. Custom development vs configuration work
        4. Testing requirements (unit tests, integration tests)
        5. Documentation and knowledge transfer
        6. Potential challenges and risks

        Provide a comprehensive estimation breakdown with justifications.
        Include a confidence_score between 0 and 1 (where 1 means very confident, 0 means very uncertain).
        """

        try:
            # Count input tokens
            input_tokens = self._count_tokens(estimation_prompt)

            result = await Runner.run(self.agent, estimation_prompt)

            # Count output tokens
            output_tokens = self._count_tokens(result.final_output)

            print(
                f"Estimation agent tokens: {input_tokens} input, {output_tokens} output")

            # Clean markdown formatting and parse JSON response
            cleaned_output = self._clean_markdown_codeblocks(result.final_output)
            estimation_result = json.loads(cleaned_output)
            return {
                "status": "success",
                "estimation": estimation_result,
                "raw_output": result.final_output,
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            }

        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": "Failed to parse estimation response",
                "raw_output": result.final_output if 'result' in locals() else "No output"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Estimation failed: {str(e)}",
                "raw_output": ""
            }
