import os
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, \
    function_tool, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import asyncio
import json
import glob

load_dotenv()

endpoint_azure = "https://models.inference.ai.azure.com"


class OdooEstimationAgent:
    """
    Specialized agent for providing time and cost estimations for Odoo projects.

    This agent focuses specifically on analyzing requirements and providing
    detailed time estimations, cost breakdowns, and resource planning.
    """

    def __init__(self, model_instance, search_tool=None):
        """
        Initialize the estimation agent.

        Args:
            model_instance: The AI model instance to use
            search_tool: Optional search tool for accessing documentation
        """
        self.model_instance = model_instance
        self.search_tool = search_tool
        self.agent = self._create_agent()

    def _create_agent(self):
        """Create the specialized estimation agent."""
        tools = [self.search_tool] if self.search_tool else []

        return Agent(
            name="OdooEstimationExpert",
            instructions="""You are a senior Odoo estimation expert with extensive experience in project planning and time estimation.

            Your expertise includes:
            - Odoo module development and customization
            - Database schema modifications
            - API integrations and third-party connections
            - User interface development
            - Testing and quality assurance
            - Deployment and maintenance considerations

            When providing estimations:
            1. Break down the work into detailed components
            2. Consider complexity factors (simple, medium, complex)
            3. Include time for testing, debugging, and documentation
            4. Account for potential risks and challenges
            5. Provide realistic time ranges with justifications
            6. Provide confidence level as decimal between 0 and 1 (0=very uncertain, 1=very certain)

            IMPORTANT: Always respond with valid JSON in this exact format:
            {
                "total_estimation_hours": "X-Y hours",
                "confidence_level": "High/Medium/Low",
                "confidence_score": 0.85,
                "breakdown": {
                    "development": "X hours - Description", 
                    "testing": "X hours - Description",
                    "documentation": "X hours - Description",
                    "deployment": "X hours - Description"
                },
                "risk_factors": ["risk1", "risk2"],
                "assumptions": ["assumption1", "assumption2"],
                "complexity_rating": "Simple/Medium/Complex",
                "recommended_approach": "Brief description of recommended implementation approach"
            }
            """,
            model=self.model_instance,
            model_settings=ModelSettings(temperature=0.1),
            tools=tools
        )

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
            result = await Runner.run(self.agent, estimation_prompt)

            # Parse JSON response
            estimation_result = json.loads(result.final_output)
            return {
                "status": "success",
                "estimation": estimation_result,
                "raw_output": result.final_output
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

    async def refine_estimation(self, original_estimation: dict,
                                additional_context: str) -> dict:
        """
        Refine an existing estimation with additional context.

        Args:
            original_estimation (dict): The original estimation result
            additional_context (str): Additional information to consider

        Returns:
            dict: Refined estimation
        """
        refinement_prompt = f"""
        Refine this existing estimation with additional context:

        Original Estimation: {json.dumps(original_estimation, indent=2)}

        Additional Context: "{additional_context}"

        Update the estimation considering this new information. Adjust time estimates, 
        risk factors, assumptions, and confidence_score as needed.
        """

        try:
            result = await Runner.run(self.agent, refinement_prompt)
            refined_estimation = json.loads(result.final_output)

            return {
                "status": "success",
                "estimation": refined_estimation,
                "refinement_notes": additional_context,
                "raw_output": result.final_output
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Estimation refinement failed: {str(e)}",
                "raw_output": ""
            }
