import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from analyst_agent import OdooGAPAnalyst
from login import render_login_ui
from token_counter import TokenCounter
from analysis_history import AnalysisHistory


class StreamlitOdooGAPApp:
    """
    Streamlit application class for Odoo GAP Analysis.

    This class manages the Streamlit interface, session state, and integrates
    with the OdooGAPAnalyst for performing business requirement analysis.
    """

    def __init__(self):
        """Initialize the Streamlit application."""
        self.setup_page_config()
        self.initialize_session_state()
        self.analyst = self._get_analyst_instance()
        self.token_counter = TokenCounter()
        self.analysis_history = AnalysisHistory()

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Odoo GAP Analysis Assistant",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def initialize_session_state(self):
        """Initialize session state variables."""
        # Initialize session state variables if they don't exist
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None

        if 'analyst_initialized' not in st.session_state:
            st.session_state.analyst_initialized = False

        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False

        # Initialize model selection state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'gpt-4o-mini'

        if 'model_changed' not in st.session_state:
            st.session_state.model_changed = False

    @staticmethod
    @st.cache_resource
    def _get_analyst_instance():
        """Get or create the OdooGAPAnalyst instance with caching."""
        return OdooGAPAnalyst()

    def _update_analyst_model(self, model_name: str):
        """Update the analyst model and handle state changes."""
        if self.analyst.get_current_model() != model_name:
            self.analyst.set_model(model_name)
            st.session_state.model_changed = True
            # Reset data loading state when model changes
            st.session_state.data_loaded = False
            st.session_state.analyst_initialized = False

    def get_current_user_email(self) -> Optional[str]:
        """Get the current user's email."""
        if hasattr(self, 'login_manager') and self.login_manager:
            current_user = self.login_manager.get_current_user()
            if current_user:
                return current_user.get('email')
        return None

    def render_header(self):
        """Render the application header."""
        st.title("🔍 Odoo GAP Analysis Assistant")
        st.markdown("""
        This tool helps analyze business requirements and determine whether they require 
        standard Odoo configuration or custom development.

        If you provide an example, estimation will be more accurate.
        Also feel free to provide examples of your own estimation, for i.e.: 
        Examples for estimation: To add one simple field to a form, takes 10 minutes.
        """)
        st.divider()

    def render_model_selector(self):
        """Render the model selection interface."""
        st.subheader("🤖 AI Model Selection")

        # Get available models
        available_models = OdooGAPAnalyst.list_available_models()

        # Create model options with descriptions
        model_options = []
        model_labels = []

        for model_key, description in available_models.items():
            model_options.append(model_key)
            model_labels.append(f"{model_key} - {description}")

        # Current model info
        current_model = self.analyst.get_current_model()
        current_index = model_options.index(
            current_model) if current_model in model_options else 0

        # Model selector
        selected_model = st.selectbox(
            "Choose OpenAI Model:",
            options=model_options,
            index=current_index,
            format_func=lambda x: available_models.get(x, x),
            help="Select the OpenAI model to use for analysis. Different models have different capabilities and costs.",
            key="model_selector"
        )

        # Update model if changed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            self._update_analyst_model(selected_model)
            st.success(f"✅ Model changed to: {selected_model}")
            st.rerun()

        # Current model display
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Current Model:** {current_model}")

        with col2:
            # Show model change notification
            if st.session_state.model_changed:
                st.warning("⚠️ Model changed - please reload knowledge base")
                st.session_state.model_changed = False

        # Model comparison info
        with st.expander("📊 Model Comparison Guide", expanded=False):
            st.markdown("""
            **Model Recommendations:**

            🚀 **GPT-4 Turbo** (Use Only for very complex requirements)
            - Best overall performance
            - Most accurate analysis
            - Latest training data
            - Balanced cost/performance

            🎯 **GPT-4o** (Latest)
            - Newest model with multimodal capabilities
            - Excellent for complex analysis
            - Higher cost but best quality

            💰 **GPT-3.5 Turbo** (Budget-friendly)
            - Faster responses
            - Lower cost
            - Good for simple requirements
            - May be less accurate for complex analysis

            ⚡ **GPT-4o Mini** (Fast & Cheap)
            - Faster responses
            - Low cost
            - Good for quick analysis
            - May lack depth for complex requirements

            🔥 **GPT-4o Nano** (Most Cost-effective)
            - Fastest responses
            - Lowest cost
            - Good for basic analysis
            - Best value for routine tasks
            """)

    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("📋 Configuration")

            # Model selection
            self.render_model_selector()

            st.divider()

            # Data loading status
            st.subheader("📚 Knowledge Base Status")
            if st.session_state.data_loaded:
                st.success("✅ Knowledge base loaded")
            else:
                st.warning("⚠️ Knowledge base not loaded")
                if st.button("🔄 Load Knowledge Base", key="load_data"):
                    self.load_knowledge_base()

            st.divider()

            # User-specific analysis history
            self.render_user_history_sidebar()

            st.divider()

            # User statistics
            self.render_user_stats_sidebar()

    def render_user_history_sidebar(self):
        """Render user-specific analysis history in sidebar."""
        st.subheader("📈 Your Recent Analyses")

        current_user_email = self.get_current_user_email()
        if not current_user_email:
            st.info("Please log in to see your analysis history")
            return

        # Get user's analysis history
        user_history = self.analysis_history.get_user_history(current_user_email,
                                                              limit=5)

        if user_history:
            for i, analysis in enumerate(user_history):
                # Format the button text
                requirement_text = analysis['requirement'][:25] + "..." if len(
                    analysis['requirement']) > 25 else analysis['requirement']
                timestamp = datetime.fromisoformat(analysis['timestamp']).strftime(
                    "%m/%d %H:%M")
                model_used = analysis.get('model_used', 'Unknown')
                impl_type = analysis.get('summary', {}).get('implementation_type',
                                                            'unknown')

                # Color code by implementation type
                if 'development' in impl_type.lower():
                    icon = "🔧"
                elif 'configuration' in impl_type.lower():
                    icon = "⚙️"
                else:
                    icon = "❓"

                button_text = f"{icon} {requirement_text}"
                help_text = f"Date: {timestamp}\nModel: {model_used}\nType: {impl_type}\nRequirement: {analysis['requirement']}"

                if st.button(
                        button_text,
                        key=f"history_{analysis['id']}",
                        help=help_text
                ):
                    # Load this analysis
                    st.session_state.current_analysis = {
                        'requirement': analysis['requirement'],
                        'result': analysis['result'],
                        'timestamp': analysis['timestamp'],
                        'model_used': analysis['model_used']
                    }
                    st.rerun()
        else:
            st.info("No previous analyses")

        # Clear history button
        if user_history and st.button("🗑️ Clear My History", key="clear_user_history"):
            if self.analysis_history.clear_user_history(current_user_email):
                st.success("Your history has been cleared!")
                st.rerun()

    def render_user_stats_sidebar(self):
        """Render user statistics in sidebar."""
        current_user_email = self.get_current_user_email()
        if not current_user_email:
            return

        stats = self.analysis_history.get_user_stats(current_user_email)

        if stats['total_analyses'] > 0:
            st.subheader("📊 Your Statistics")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Analyses", stats['total_analyses'])
                st.metric("Configurations", stats['configuration_analyses'])

            with col2:
                st.metric("Developments", stats['development_analyses'])
                st.metric("Avg Confidence", f"{stats['avg_confidence_score']:.2f}")

            # Show most used model
            if stats['models_used']:
                most_used_model = max(stats['models_used'].items(), key=lambda x: x[1])
                st.info(
                    f"**Most Used Model:** {most_used_model[0]} ({most_used_model[1]} times)")

    def load_knowledge_base(self):
        """Load the knowledge base with progress indication."""
        with st.spinner("Loading knowledge base..."):
            try:
                # Load data to database
                vstore = self.analyst.load_data_to_database()
                if vstore:
                    # Create search tool
                    self.analyst.create_search_tool()
                    st.session_state.data_loaded = True
                    st.session_state.analyst_initialized = True
                    st.success("✅ Knowledge base loaded successfully!")
                else:
                    st.error("❌ Failed to load knowledge base")
            except Exception as e:
                st.error(f"❌ Error loading knowledge base: {str(e)}")

    def render_input_section(self):
        """Render the requirement input section."""
        st.header("📝 Business Requirement Input")

        # Show current model being used
        current_model = self.analyst.get_current_model()
        model_description = OdooGAPAnalyst.list_available_models().get(current_model,
                                                                       "Unknown")

        st.info(f"🤖 **Using Model:** {current_model} - {model_description}")

        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["✍️ Text Input", "📁 File Upload"],
            horizontal=True
        )

        requirement_text = ""

        if input_method == "✍️ Text Input":
            requirement_text = st.text_area(
                "Enter your business requirement:",
                height=150,
                placeholder="Example: Warning about raw material expiration before x days..."
            )

        elif input_method == "📁 File Upload":
            uploaded_file = st.file_uploader(
                "Upload requirement file:",
                type=['txt', 'md'],
                help="Upload a text file containing your business requirement"
            )

            if uploaded_file is not None:
                # Store in session state to persist across reruns
                st.session_state.uploaded_file = uploaded_file
                requirement_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=requirement_text, height=150,
                             disabled=True)

        return requirement_text

    def render_analysis_button(self, requirement_text: str) -> bool:
        """Render the analysis button and return if analysis should be performed."""
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if not st.session_state.data_loaded:
                st.warning("⚠️ Please load the knowledge base first")
                return False

            if not requirement_text.strip():
                st.warning("⚠️ Please enter a business requirement")
                return False

            return st.button(
                "🚀 Analyze Requirement",
                type="primary",
                use_container_width=True
            )

    async def perform_analysis(self, requirement_text: str) -> Optional[Dict[str, Any]]:
        """Perform the GAP analysis asynchronously and track token usage."""
        try:
            # Get current user email
            current_user = None
            if hasattr(self, 'login_manager') and self.login_manager:
                current_user = self.login_manager.get_current_user()

            # Perform analysis
            result = await self.analyst.analyze_requirement(requirement_text)

            # Track token usage if we have a result with token usage data
            if result and 'token_usage' in result and current_user:
                user_email = current_user.get('email')
                model_name = self.analyst.get_current_model()
                input_tokens = result['token_usage'].get('input_tokens', 0)
                output_tokens = result['token_usage'].get('output_tokens', 0)

                # Record token usage
                self.token_counter.add_tokens(
                    user_email=user_email,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

                print(
                    f"Recorded token usage for {user_email}: {input_tokens} input, {output_tokens} output tokens with model {model_name}")

            return result
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            return None

    def run_analysis(self, requirement_text: str):
        """Run the analysis with progress indication."""
        current_model = self.analyst.get_current_model()
        progress_bar = st.progress(0, text=f"Starting analysis with {current_model}...")

        try:
            # Update progress
            progress_bar.progress(20, text="Initializing analyst...")
            time.sleep(0.5)

            progress_bar.progress(40, text="Searching knowledge base...")
            time.sleep(0.5)

            progress_bar.progress(60, text="Generating GAP analysis...")
            time.sleep(0.5)

            progress_bar.progress(80, text="Creating time estimation...")
            time.sleep(0.5)

            # Run the actual analysis
            result = asyncio.run(self.perform_analysis(requirement_text))

            progress_bar.progress(100, text="Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()

            if result:
                # Store in session state and user-specific history
                analysis_data = {
                    'requirement': requirement_text,
                    'result': result,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'model_used': current_model  # Store model used for this analysis
                }

                st.session_state.current_analysis = analysis_data

                # Save to user-specific history
                current_user_email = self.get_current_user_email()
                if current_user_email:
                    analysis_id = self.analysis_history.add_analysis(current_user_email,
                                                                     analysis_data)
                    print(f"Saved analysis {analysis_id} for user {current_user_email}")

                return True

        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Analysis failed: {str(e)}")
            return False

        return False

    def render_analysis_results(self, analysis_data: Dict[str, Any]):
        """Render the analysis results in a structured format."""
        st.header("📊 Analysis Results")

        result = analysis_data['result']
        response = result.get('response', {})
        analysis_result = response.get('analysis_result', {})

        # Show analysis metadata
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📅 Analysis Date", analysis_data.get('timestamp', 'Unknown'))

        with col2:
            model_used = analysis_data.get('model_used') or response.get('model_used',
                                                                         'Unknown')
            st.metric("🤖 Model Used", model_used)

        with col3:
            current_user_email = self.get_current_user_email()
            if current_user_email:
                user_stats = self.analysis_history.get_user_stats(current_user_email)
                st.metric("📊 Your Total Analyses", user_stats['total_analyses'])

        st.divider()

        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 GAP Analysis", "⏱️ Time Estimation", "🔧 Design Solution", "📚 Sources"])

        with tab1:
            self.render_gap_analysis_tab(analysis_result, response)

        with tab2:
            self.render_estimation_tab(response.get('time_estimation', {}))

        with tab3:
            self.render_design_solution_tab(response)

        with tab4:
            self.render_sources_tab(result.get('sources', []))

    def render_gap_analysis_tab(self, analysis_result: Dict, response: Dict):
        """Render the GAP analysis tab content."""
        if not analysis_result:
            st.warning("No GAP analysis results available.")
            return

        # Implementation Type Badge
        implementation_type = response.get('implementation_type', 'Unknown')
        if 'development' in implementation_type.lower():
            st.error(f"🔧 **Implementation Type:** {implementation_type}")
        else:
            st.success(f"⚙️ **Implementation Type:** {implementation_type}")

        # Confidence Score
        confidence_score = analysis_result.get('confidence_score', 0)
        if confidence_score >= 0.8:
            st.success(f"✅ **Confidence Score:** {confidence_score:.2f} (High)")
        elif confidence_score >= 0.6:
            st.warning(f"⚠️ **Confidence Score:** {confidence_score:.2f} (Medium)")
        else:
            st.error(f"❌ **Confidence Score:** {confidence_score:.2f} (Low)")

        st.divider()

        # Analysis sections
        sections = [
            ("📋 Executive Summary",
             analysis_result.get('executive_summary', 'Not available')),
            ("💡 Recommendations",
             analysis_result.get('recommendations', 'Not available')),
            ("📝 Implementation Steps",
             analysis_result.get('implementation_steps', 'Not available'))
        ]

        for title, content in sections:
            st.subheader(title)
            st.write(content)
            st.markdown("---")

    def render_estimation_tab(self, estimation_data: Dict):
        """Render the time estimation tab content."""
        if not estimation_data or 'error' in estimation_data:
            st.warning("No time estimation available or estimation failed.")
            if 'error' in estimation_data:
                st.error(f"Error: {estimation_data['error']}")
            return

        # Main estimation metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            total_hours = estimation_data.get('total_estimation_hours', 'Not specified')
            st.metric("⏱️ Total Estimation", total_hours)

        with col2:
            complexity = estimation_data.get('complexity_rating', 'Unknown')
            if complexity.lower() == 'complex':
                st.error(f"🔴 Complexity: {complexity}")
            elif complexity.lower() == 'medium':
                st.warning(f"🟡 Complexity: {complexity}")
            else:
                st.success(f"🟢 Complexity: {complexity}")

        with col3:
            confidence = estimation_data.get('confidence_level', 'Unknown')
            confidence_score = estimation_data.get('confidence_score', 0)
            if confidence_score >= 0.8:
                st.success(f"✅ Confidence: {confidence}")
            elif confidence_score >= 0.6:
                st.warning(f"⚠️ Confidence: {confidence}")
            else:
                st.error(f"❌ Confidence: {confidence}")

        st.divider()

        # Breakdown section
        breakdown = estimation_data.get('breakdown', {})
        if breakdown:
            st.subheader("📊 Time Breakdown")

            phases = [
                ("🔍 Analysis Phase", breakdown.get('analysis_phase', 'Not specified')),
                ("⚙️ Development Phase",
                 breakdown.get('development_phase', 'Not specified')),
                ("🧪 Testing Phase", breakdown.get('testing_phase', 'Not specified')),
                ("🚀 Deployment Phase",
                 breakdown.get('deployment_phase', 'Not specified'))
            ]

            for phase_title, phase_content in phases:
                with st.expander(phase_title, expanded=False):
                    st.write(phase_content)

        # Additional information
        col1, col2 = st.columns(2)

        with col1:
            # Risk factors
            risk_factors = estimation_data.get('risk_factors', [])
            if risk_factors:
                st.subheader("⚠️ Risk Factors")
                for risk in risk_factors:
                    st.write(f"• {risk}")

        with col2:
            # Assumptions
            assumptions = estimation_data.get('assumptions', [])
            if assumptions:
                st.subheader("📋 Assumptions")
                for assumption in assumptions:
                    st.write(f"• {assumption}")

            # Recommended approach
            recommended_approach = estimation_data.get('recommended_approach', '')
            if recommended_approach:
                st.subheader("🎯 Recommended Approach")
                st.write(recommended_approach)

    def render_design_solution_tab(self, response: Dict):
        """Render the design solution tab content."""
        design_solution = response.get('design_solution')

        if not design_solution:
            st.info(
                "No design solution available. Design solutions are generated only for development requirements.")
            return

        st.subheader("🏗️ Technical Design Solution")
        st.markdown(design_solution)

    def render_sources_tab(self, sources: list):
        """Render the sources tab content."""
        if not sources:
            st.warning("No sources available.")
            return

        st.subheader("📚 Knowledge Base Sources")
        st.write("The following sources were used to generate this analysis:")

        for i, source in enumerate(sources, 1):
            metadata = source.get('metadata', {})
            title = metadata.get('title', f'Source {i}')
            content = source.get('content', 'No content available')

            with st.expander(f"📄 {title}", expanded=False):
                st.write("**Content:**")
                st.write(content)

                # Show source metadata if available
                source_path = metadata.get('source', 'Unknown')
                page = metadata.get('page', 'N/A')
                if source_path != 'Unknown':
                    st.write(f"**Source:** {source_path}")
                if page != 'N/A':
                    st.write(f"**Page:** {page}")

    def render_export_options(self, analysis_data: Dict[str, Any]):
        """Render export options for the analysis results."""
        st.header("📤 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            # JSON export
            if st.button("📄 Export as JSON", type="secondary"):
                json_data = json.dumps(analysis_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Download JSON",
                    data=json_data,
                    file_name=f"gap_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col2:
            # Text summary export
            if st.button("📝 Export as Text", type="secondary"):
                text_summary = self.generate_text_summary(analysis_data)
                st.download_button(
                    label="💾 Download Text",
                    data=text_summary,
                    file_name=f"gap_analysis_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        with col3:
            # Copy to clipboard (show the text)
            if st.button("📋 Show Text Summary", type="secondary"):
                st.text_area(
                    "Copy this text:",
                    value=self.generate_text_summary(analysis_data),
                    height=200
                )

    def generate_text_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Generate a text summary of the analysis results."""
        result = analysis_data['result']
        response = result.get('response', {})
        analysis_result = response.get('analysis_result', {})

        summary_parts = [
            "=" * 60,
            "ODOO GAP ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Analysis Date: {analysis_data.get('timestamp', 'Unknown')}",
            f"Model Used: {analysis_data.get('model_used', 'Unknown')}",
            "",
            "BUSINESS REQUIREMENT:",
            "-" * 20,
            analysis_data.get('requirement', 'Not available'),
            "",
            "ANALYSIS RESULTS:",
            "-" * 17,
            f"Implementation Type: {response.get('implementation_type', 'Unknown')}",
            f"Confidence Score: {analysis_result.get('confidence_score', 0):.2f}",
            "",
            "EXECUTIVE SUMMARY:",
            "-" * 18,
            analysis_result.get('executive_summary', 'Not available'),
            "",
            "RECOMMENDATIONS:",
            "-" * 16,
            analysis_result.get('recommendations', 'Not available'),
            "",
            "IMPLEMENTATION STEPS:",
            "-" * 21,
            analysis_result.get('implementation_steps', 'Not available'),
            ""
        ]

        # Add time estimation if available
        estimation_data = response.get('time_estimation', {})
        if estimation_data and 'error' not in estimation_data:
            summary_parts.extend([
                "TIME ESTIMATION:",
                "-" * 16,
                f"Total Hours: {estimation_data.get('total_estimation_hours', 'Not specified')}",
                f"Complexity: {estimation_data.get('complexity_rating', 'Unknown')}",
                f"Confidence: {estimation_data.get('confidence_level', 'Unknown')}",
            ])

            # Add breakdown if available
            breakdown = estimation_data.get('breakdown', {})
            if breakdown:
                summary_parts.extend([
                    "BREAKDOWN:",
                    "-" * 10,
                    f"Analysis: {breakdown.get('analysis_phase', 'Not specified')}",
                    f"Development: {breakdown.get('development_phase', 'Not specified')}",
                    f"Testing: {breakdown.get('testing_phase', 'Not specified')}",
                    f"Deployment: {breakdown.get('deployment_phase', 'Not specified')}",
                    ""
                ])

        # Add design solution if available
        design_solution = response.get('design_solution')
        if design_solution:
            summary_parts.extend([
                "DESIGN SOLUTION:",
                "-" * 16,
                design_solution,
                ""
            ])

        summary_parts.extend([
            "=" * 60,
            "End of Report",
            "=" * 60
        ])

        return "\n".join(summary_parts)

    def run(self):
        """Main application runner."""
        # Check if user is logged in
        authenticated, login_manager = render_login_ui()

        # Store login_manager for use in other methods
        self.login_manager = login_manager

        if not authenticated:
            # User is not logged in or is in admin panel
            return

        # User is authenticated, proceed with the main application
        self.render_header()
        self.render_sidebar()

        # Main content area
        requirement_text = self.render_input_section()

        # Analysis button
        if self.render_analysis_button(requirement_text):
            if self.run_analysis(requirement_text):
                st.rerun()

        # Display current analysis results
        if st.session_state.current_analysis:
            st.divider()
            self.render_analysis_results(st.session_state.current_analysis)

            st.divider()
            self.render_export_options(st.session_state.current_analysis)


# Entry point for the Streamlit app
if __name__ == "__main__":
    app = StreamlitOdooGAPApp()
    app.run()
