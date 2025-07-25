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

from estimation_agent import OdooEstimationAgent

load_dotenv()

endpoint_azure = "https://models.inference.ai.azure.com"


class OdooGAPAnalyst:
    """
    A comprehensive AI agent for Odoo GAP analysis and design solutions.

    This class handles document loading, vector database management,
    agent creation, and analysis execution for Odoo customization requirements.
    """

    def __init__(self, data_directory="./data", db_directory="./chroma_db"):
        """
        Initialize the Odoo GAP Analyst.

        Args:
            data_directory (str): Directory containing PDF and TXT files
            db_directory (str): Directory for the vector database
        """
        self.data_directory = data_directory
        self.db_directory = db_directory
        self.vstore = None
        self.search_tool = None
        self.estimation_agent = None

        # Initialize OpenAI settings
        self.token = os.getenv("OPENAI_KEY")
        self.endpoint = 'https://models.github.ai/inference'
        self.model_name = 'openai/gpt-4.1-nano'
        self.endpoint_azure = "https://models.inference.ai.azure.com"

        # Setup OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self.token
        )

        # Setup model instance
        self.model_instance = OpenAIChatCompletionsModel(
            model=self.model_name,
            openai_client=self.client,
        )

        set_tracing_disabled(True)

    def load_data_to_database(self):
        """
        Load PDF and TXT data to vector database using PyPDFLoader, TextLoader and RecursiveCharacterTextSplitter.

        Returns:
            Chroma: Vector store instance or None if failed
        """
        try:
            # Initialize embeddings - use standard OpenAI endpoint for embeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                base_url=self.endpoint_azure,
                api_key=self.token
            )

            # Create vector store
            self.vstore = Chroma(
                persist_directory=self.db_directory,
                embedding_function=embeddings
            )

            # Check if the database already exists
            try:
                collection = self.vstore._collection
                if collection.count() > 0:
                    print(
                        f"Found existing database with {collection.count()} documents.")
                else:
                    print("No existing data found. Loading documents...")
            except Exception:
                print("No existing data found. Loading documents...")

            all_docs = []

            # Get list of already loaded files from metadata if available
            loaded_files = set()
            try:
                if collection.count() > 0:
                    all_metadata = collection.get()["metadatas"]
                    for metadata in all_metadata:
                        if metadata and "source" in metadata:
                            loaded_files.add(metadata["source"])
                    print(
                        f"Found {len(loaded_files)} already loaded files: {loaded_files}")
            except Exception as e:
                print(f"Error retrieving loaded files: {e}")
                loaded_files = set()

            # Load PDF files
            pdf_files = glob.glob(f"{self.data_directory}/*.pdf")
            for pdf_file in pdf_files:
                if pdf_file in loaded_files:
                    print(f"Skipping already loaded PDF: {pdf_file}")
                    continue

                print(f"Loading PDF: {pdf_file}")
                pdf_loader = PyPDFLoader(pdf_file)
                splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                          chunk_overlap=64)
                docs_from_pdf = pdf_loader.load_and_split(text_splitter=splitter)

                # Add source metadata to each document
                for doc in docs_from_pdf:
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = pdf_file

                # Add documents to vector store immediately
                if docs_from_pdf:
                    self.vstore.add_documents(docs_from_pdf)
                    print(
                        f"Loaded and inserted {len(docs_from_pdf)} chunks from {pdf_file}")
                    loaded_files.add(pdf_file)

                all_docs.extend(docs_from_pdf)

            # Load TXT files
            txt_files = glob.glob(f"{self.data_directory}/*.txt")
            for txt_file in txt_files:
                if txt_file in loaded_files:
                    print(f"Skipping already loaded TXT: {txt_file}")
                    continue

                print(f"Loading TXT: {txt_file}")
                txt_loader = TextLoader(txt_file, encoding='utf-8')
                splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                          chunk_overlap=64)
                docs_from_txt = txt_loader.load_and_split(text_splitter=splitter)

                # Add source metadata to each document
                for doc in docs_from_txt:
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = txt_file

                # Add documents to vector store immediately
                if docs_from_txt:
                    self.vstore.add_documents(docs_from_txt)
                    print(
                        f"Loaded and inserted {len(docs_from_txt)} chunks from {txt_file}")
                    loaded_files.add(txt_file)

                all_docs.extend(docs_from_txt)

            if not all_docs and not loaded_files:
                print(f"No PDF or TXT files found in {self.data_directory} directory.")
            else:
                print(f"Total documents in database: {collection.count()}.")

            return self.vstore

        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            return None
        except Exception as e:
            print(f"Error loading document data: {e}")
            return None

    def create_search_tool(self):
        """
        Create a search tool that the agent can use to query the PDF data.

        Returns:
            function: Search tool function
        """
        if not self.vstore:
            raise ValueError(
                "Vector store not initialized. Call load_data_to_database() first.")

        @function_tool
        def search_pdf_knowledge(query: str) -> str:
            """Search for relevant information from the loaded PDF documents.

            Args:
                query: The search query to find relevant information

            Returns:
                Relevant text passages from the PDF
            """
            try:
                # Perform similarity search
                results = self.vstore.similarity_search(query, k=3)

                if not results:
                    return "No relevant information found in the PDF documents."

                # Format the results
                formatted_results = []
                for i, doc in enumerate(results, 1):
                    content = doc.page_content.strip()
                    formatted_results.append(f"Source {i}:\n{content}\n")

                # Store the results for later use in sources
                search_pdf_knowledge.last_results = results

                return "\n".join(formatted_results)

            except Exception as e:
                search_pdf_knowledge.last_results = []
                return f"Error searching PDF knowledge: {str(e)}"

        self.search_tool = search_pdf_knowledge
        return search_pdf_knowledge

    def create_estimation_agent(self):
        """
        Create the specialized estimation agent.

        Returns:
            OdooEstimationAgent: Configured estimation agent
        """
        if not self.estimation_agent:
            self.estimation_agent = OdooEstimationAgent(
                model_instance=self.model_instance,
                search_tool=self.search_tool
            )
        return self.estimation_agent

    def create_gap_analysis_agent(self):
        """
        Create the main GAP analysis agent.

        Returns:
            Agent: Configured agent for GAP analysis
        """
        if not self.search_tool:
            raise ValueError("Search tool not created. Call create_search_tool() first.")

        return Agent(
            name="Assistant",
            instructions="""You are a helpful assistant that can make a good Odoo GAP analysis.
            You have access to PDF documents that may contain relevant information.
            On the input will receive a business need.
            When answering questions:
             1. search for relevant information using the search_pdf_knowledge tool,
             2. decide if its a standard odoo customization or a custom development,
             3. provide a comprehensive answer based on both the retrieved information and your knowledge.
             4. if question is not clear enough or you need some additional info, ask for user.
             5. Provide a confidence score between 0 and 1 (where 1 means very confident, 0 means very uncertain)

             IMPORTANT: Always respond with valid JSON in this exact format:
                {
                    "executive_summary": "brief overview here",
                    "recommendations": "rec1. rec2.",
                    "implementation_steps": "step1, step2.", 
                    "implementation_type": "odoo configuration" or "development",
                    "confidence_score": 0.85
                }
             Note: Time estimation will be handled by a separate specialized agent.
             """,
            model=self.model_instance,
            model_settings=ModelSettings(temperature=0.1),
            tools=[self.search_tool]
        )

    def create_design_agent(self):
        """
        Create the design architect agent for development solutions.

        Returns:
            Agent: Configured agent for design solutions
        """
        if not self.search_tool:
            raise ValueError("Search tool not created. Call create_search_tool() first.")

        return Agent(
            name="DesignArchitect",
            instructions="""You are a senior Odoo technical architect specializing in custom development solutions.""",
            model=self.model_instance,
            model_settings=ModelSettings(temperature=0.2),
            tools=[self.search_tool]
        )

    async def create_design_solution(self, requirement):
        """
        Create a detailed design solution for development requirements.

        Args:
            requirement (str): The business requirement

        Returns:
            str: Design solution output
        """
        design_agent = self.create_design_agent()

        design_prompt = f"""
        Based on the requirement: "{requirement}"

        Create a comprehensive design solution that includes:
        1. **Technical Architecture**: Overall system design and components
        2. **Design solution**: How it will be implemented, including database schema changes, API endpoints, user interface considerations, integration points with existing Odoo modules.
        3. **Possible challenges implementing this solution**: What might be the most difficult parts of the implementation, and how to overcome them.
        4. **Code snippet**: Provide a code snippet that can be used to implement the solution.

        Note: Time estimation will be handled separately by a specialized estimation agent.
        """

        design_result = await Runner.run(design_agent, design_prompt)
        return design_result.final_output

    def get_sources_from_search(self):
        """
        Extract sources from the last search results.

        Returns:
            list: List of source dictionaries
        """
        sources = []
        if hasattr(self.search_tool, 'last_results') and self.search_tool.last_results:
            for doc in self.search_tool.last_results:
                source_path = doc.metadata.get("source", "Unknown")
                filename = source_path.split("/")[
                    -1] if "/" in source_path else source_path

                source_entry = {
                    "content": doc.page_content.strip(),
                    "metadata": {
                        "title": f"{filename} (Page {doc.metadata.get('page', 'N/A')})",
                        "url": source_path,
                        "source": source_path,
                        "page": doc.metadata.get("page", "")
                    }
                }
                sources.append(source_entry)

        if not sources:
            sources = [{
                "content": "No source documents were retrieved during this query",
                "metadata": {"title": "N/A", "url": "N/A"}
            }]

        return sources

    async def analyze_requirement(self, user_requirement):
        """
        Perform complete GAP analysis for a given requirement.

        Args:
            user_requirement (str): The business requirement to analyze

        Returns:
            dict: Complete analysis result with sources and estimation
        """
        # Ensure everything is set up
        if not self.vstore:
            print("Loading PDF data to vector database...")
            self.vstore = self.load_data_to_database()
            if not self.vstore:
                raise RuntimeError("Failed to load PDF data.")
            print("PDF data loaded successfully!")

        if not self.search_tool:
            self.create_search_tool()

        # Create and run GAP analysis agent
        agent = self.create_gap_analysis_agent()
        result = await Runner.run(agent, user_requirement)

        print("=== GAP Analysis Result ===")
        print(result.final_output)

        # Validate JSON response
        try:
            json.loads(result.final_output)
        except (json.JSONDecodeError, TypeError):
            return result.final_output

        # Parse the JSON response
        analysis_result = json.loads(result.final_output)
        implementation_type = analysis_result.get("implementation_type", "").lower()

        response_data = {
            "analysis_result": analysis_result,
            "implementation_type": implementation_type,
        }

        # Get time estimation from specialized agent
        print("\n=== Creating Time Estimation ===")
        estimation_agent = self.create_estimation_agent()
        estimation_result = await estimation_agent.estimate_requirement(
            user_requirement,
            implementation_type
        )

        if estimation_result["status"] == "success":
            response_data["time_estimation"] = estimation_result["estimation"]
            print("Time estimation completed successfully!")
        else:
            print(
                f"Time estimation failed: {estimation_result.get('message', 'Unknown error')}")
            response_data["time_estimation"] = {
                "error": estimation_result.get("message", "Estimation failed")
            }

        # If implementation type is development, create design solution
        if "development" in implementation_type:
            print("\n=== Creating Design Solution ===")
            print("Implementation type is 'development'. Generating design solution...")

            design_solution = await self.create_design_solution(user_requirement)
            response_data["design_solution"] = design_solution

        return {
            "response": response_data,
            "sources": self.get_sources_from_search()
        }
