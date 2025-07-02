
# Odoo GAP Analysis Assistant

A Streamlit application that helps analyze business requirements for Odoo ERP implementation, determining whether they require standard Odoo configuration or custom development. The application also provides time and cost estimations for implementing the requirements.

## 🌟 Features

- **GAP Analysis**: Analyze business requirements against Odoo's standard capabilities
- **Time Estimation**: Get detailed time estimates for implementing requirements
- **Solution Design**: Receive suggestions for implementation approaches
- **Knowledge Base**: Utilizes Odoo documentation for informed analysis
- **Export Options**: Export analysis results in various formats
- **History Tracking**: Keep track of previous analyses

## 📋 Project Description

The Odoo GAP Analysis Assistant is an AI-powered tool that helps Odoo consultants and project managers analyze business requirements and determine the best implementation approach. It leverages OpenAI models and a knowledge base of Odoo documentation to provide accurate analyses and estimations.

The application can:
- Determine if a requirement can be implemented using standard Odoo features
- Identify when custom development is needed
- Provide detailed time and cost estimations
- Suggest implementation approaches
- Reference relevant Odoo documentation

## 🚀 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key or Azure OpenAI service access

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd OdooAIBusinesAnalyst
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install streamlit openai langchain langchain-openai langchain-community python-dotenv chromadb openai-agents pypdf

   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   # Or for Azure OpenAI:
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   ```

## 🖥️ Usage

1. Run the application:
   ```bash
   streamlit run main.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Load the knowledge base by clicking the "Load Knowledge Base" button in the sidebar

4. Enter your business requirement in the text area and click "Analyze Requirement"

5. View the analysis results in the tabs:
   - GAP Analysis
   - Estimation
   - Design Solution
   - Sources

6. Export the results using the export options at the bottom of the analysis

## 📁 Project Structure

```
Streamlit-App/
├── main.py                 # Entry point for the application
├── streamlit_app.py        # Main Streamlit application code
├── analyst_agent.py        # GAP analysis agent implementation
├── estimation_agent.py     # Estimation agent implementation
├── data/                   # Knowledge base data
│   ├── Odoo 18 Sales Module User Manual.pdf
│   ├── Odoo 18 Inventory Module User Manual.pdf
│   ├── Odoo 18 Purchase Module User Manual.pdf
│   ├── Odoo 18 Manufacturing Module User Manual.pdf
│   ├── implementation_methodology.pdf
│   ├── odoo_sale.txt
│   ├── odoo_inventory.txt
│   ├── odoo_purchase.txt
│   └── odoo_mrp.txt
└── chroma_db/              # Vector database storage
```

## 🔧 Configuration

The application uses environment variables for configuration. You can modify the `.env` file to change:

- OpenAI API key or Azure OpenAI service credentials
- Model settings (temperature, etc.)
- Other configuration parameters

## 🧩 Dependencies

- **streamlit**: Web application framework
- **openai**: OpenAI API client
- **langchain**: Framework for LLM applications
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-community**: Community components for LangChain
- **python-dotenv**: Environment variable management
- **chromadb**: Vector database for document storage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Specify the license under which the project is released]

## 🙏 Acknowledgements

- OpenAI for providing the AI models
- Streamlit for the web application framework
- LangChain for the LLM application framework
- Odoo for the ERP system and documentation