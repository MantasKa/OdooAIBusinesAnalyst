import streamlit as st
import asyncio
import json
import time
from typing import Dict, Any, Optional
from analyst_agent import OdooGAPAnalyst


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
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []

        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None

        if 'analyst_initialized' not in st.session_state:
            st.session_state.analyst_initialized = False

        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False

    @staticmethod
    @st.cache_resource
    def _get_analyst_instance():
        """Get or create the OdooGAPAnalyst instance with caching."""
        return OdooGAPAnalyst()

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

    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("📋 Configuration")

            # Data loading status
            st.subheader("📚 Knowledge Base Status")
            if st.session_state.data_loaded:
                st.success("✅ Knowledge base loaded")
            else:
                st.warning("⚠️ Knowledge base not loaded")
                if st.button("🔄 Load Knowledge Base", key="load_data"):
                    self.load_knowledge_base()

            st.divider()

            # Analysis history
            st.subheader("📈 Recent Analyses")
            if st.session_state.analysis_history:
                for i, analysis in enumerate(
                        reversed(st.session_state.analysis_history[-5:])):
                    if st.button(
                            f"📝 {analysis['requirement'][:30]}...",
                            key=f"history_{i}",
                            help=analysis['requirement']
                    ):
                        st.session_state.current_analysis = analysis
                        st.rerun()
            else:
                st.info("No previous analyses")

            st.divider()

            # Clear history button
            if st.button("🗑️ Clear History", key="clear_history"):
                st.session_state.analysis_history = []
                st.session_state.current_analysis = None
                st.success("History cleared!")
                st.rerun()

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
        """Perform the GAP analysis asynchronously."""
        try:
            result = await self.analyst.analyze_requirement(requirement_text)
            return result
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            return None

    def run_analysis(self, requirement_text: str):
        """Run the analysis with progress indication."""
        progress_bar = st.progress(0, text="Starting analysis...")

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
                # Store in session state and history
                analysis_data = {
                    'requirement': requirement_text,
                    'result': result,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }

                st.session_state.current_analysis = analysis_data
                st.session_state.analysis_history.append(analysis_data)

                # Keep only last 10 analyses
                if len(st.session_state.analysis_history) > 10:
                    st.session_state.analysis_history = st.session_state.analysis_history[
                                                        -10:]

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

    def render_gap_analysis_tab(self, analysis_result: Dict[str, Any], response: Dict[str, Any]):
        """Render the GAP analysis tab."""
        if not analysis_result:
            st.warning("No analysis result available")
            return
        
        # Top row with implementation type and confidence
        col1, col2 = st.columns(2)
        
        with col1:
            # Implementation type with colored badge
            impl_type = response.get('implementation_type', 'unknown').lower()
            if 'development' in impl_type:
                st.error(f"🔧 **Implementation Type:** Custom Development")
            elif 'configuration' in impl_type:
                st.success(f"⚙️ **Implementation Type:** Odoo Configuration")
            else:
                st.info(f"❓ **Implementation Type:** {impl_type.title()}")
        
        with col2:
            # GAP Analysis confidence score
            confidence_score = analysis_result.get('confidence_score', 0)
            if confidence_score >= 0.8:
                confidence_color = "🟢"
                confidence_text = "High"
            elif confidence_score >= 0.6:
                confidence_color = "🟡" 
                confidence_text = "Medium"
            else:
                confidence_color = "🔴"
                confidence_text = "Low"
            
            st.metric(
                "🎯 Analysis Confidence", 
                f"{confidence_color} {confidence_text}",
                delta=f"{confidence_score:.2f}"
            )
        
        st.divider()
        
        # Executive Summary
        if 'executive_summary' in analysis_result:
            st.subheader("📝 Executive Summary")
            st.write(analysis_result['executive_summary'])
        
        # Recommendations
        if 'recommendations' in analysis_result:
            st.subheader("💡 Recommendations")
            st.write(analysis_result['recommendations'])
        
        # Implementation Steps
        if 'implementation_steps' in analysis_result:
            st.subheader("📋 Implementation Steps")
            st.write(analysis_result['implementation_steps'])
    
    def render_estimation_tab(self, estimation: Dict[str, Any]):
        """Render the time estimation tab."""
        if 'error' in estimation:
            st.error(f"❌ Estimation Error: {estimation['error']}")
            return
        
        if not estimation:
            st.info("💡 No time estimation available.")
            return
        
        # Main estimation summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "⏱️ Total Time", 
                estimation.get('total_estimation_hours', 'N/A')
            )
        
        with col2:
            # Estimation confidence (separate from GAP analysis confidence)
            confidence = estimation.get('confidence_level', 'Unknown')
            confidence_score = estimation.get('confidence_score', 0)
            confidence_color = {
                'High': '🟢', 
                'Medium': '🟡', 
                'Low': '🔴'
            }.get(confidence, '⚪')
            st.metric(
                "🎯 Estimation Confidence", 
                f"{confidence_color} {confidence}",
                delta=f"{confidence_score:.2f}" if confidence_score else None
            )
        
        with col3:
            complexity = estimation.get('complexity_rating', 'Unknown')
            complexity_color = {
                'Simple': '🟢',
                'Medium': '🟡', 
                'Complex': '🔴'
            }.get(complexity, '⚪')
            st.metric(
                "⚙️ Complexity", 
                f"{complexity_color} {complexity}"
            )
        
        st.divider()
        
        # Detailed breakdown
        if 'breakdown' in estimation:
            st.subheader("📋 Time Breakdown")
            breakdown = estimation['breakdown']
            
            for phase, details in breakdown.items():
                with st.expander(f"🔹 {phase.replace('_', ' ').title()}", expanded=True):
                    st.write(details)
        
        # Additional information
        col1, col2 = st.columns(2)
        
        with col1:
            if 'risk_factors' in estimation and estimation['risk_factors']:
                st.subheader("⚠️ Risk Factors")
                for risk in estimation['risk_factors']:
                    st.write(f"• {risk}")
        
        with col2:
            if 'assumptions' in estimation and estimation['assumptions']:
                st.subheader("📝 Assumptions")
                for assumption in estimation['assumptions']:
                    st.write(f"• {assumption}")
        
        # Recommended approach
        if 'recommended_approach' in estimation:
            st.subheader("🎯 Recommended Approach")
            st.info(estimation['recommended_approach'])
        
        # Skill level required
        if 'skill_level_required' in estimation:
            st.subheader("👨‍💻 Required Skill Level")
            skill_level = estimation['skill_level_required']
            skill_color = {
                'Junior': '🟢',
                'Mid': '🟡',
                'Senior': '🔴'
            }.get(skill_level, '⚪')
            st.write(f"{skill_color} **{skill_level} Developer**")

    def render_design_solution_tab(self, response: Dict[str, Any]):
        """Render the design solution tab."""
        if 'design_solution' in response:
            st.subheader("🏗️ Technical Design Solution")
            st.markdown(response['design_solution'])
        else:
            st.info(
                "💡 Design solution is generated only for custom development requirements.")

    def render_sources_tab(self, sources: list):
        """Render the sources tab."""
        if not sources:
            st.info("No sources were referenced during this analysis.")
            return

        st.subheader("📚 Referenced Sources")

        for i, source in enumerate(sources, 1):
            with st.expander(
                    f"📄 Source {i}: {source.get('metadata', {}).get('title', 'Unknown')}",
                    expanded=False):
                content = source.get('content', 'No content available')
                metadata = source.get('metadata', {})

                st.write("**Content:**")
                st.write(content)

                if metadata:
                    st.write("**Metadata:**")
                    col1, col2 = st.columns(2)

                    with col1:
                        if 'source' in metadata:
                            st.write(f"**File:** {metadata['source']}")

                    with col2:
                        if 'page' in metadata and metadata['page']:
                            st.write(f"**Page:** {metadata['page']}")

    def render_export_options(self, analysis_data: Dict[str, Any]):
        """Render export options for the analysis results."""
        st.header("📤 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # Export as JSON
            json_data = json.dumps(analysis_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 Download as JSON",
                data=json_data,
                file_name=f"gap_analysis_{int(time.time())}.json",
                mime="application/json"
            )

        with col2:
            # Export as text summary
            text_summary = self.generate_text_summary(analysis_data)
            st.download_button(
                label="📝 Download as Text",
                data=text_summary,
                file_name=f"gap_analysis_{int(time.time())}.txt",
                mime="text/plain"
            )

    def generate_text_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Generate a text summary of the analysis."""
        result = analysis_data['result']
        response = result.get('response', {})
        analysis_result = response.get('analysis_result', {})
        estimation = response.get('time_estimation', {})

        summary = f"""
ODOO GAP ANALYSIS REPORT
========================

Timestamp: {analysis_data.get('timestamp', 'Unknown')}
Business Requirement: {analysis_data.get('requirement', 'Unknown')}

ANALYSIS RESULTS
================

Implementation Type: {response.get('implementation_type', 'Unknown')}

Executive Summary:
{analysis_result.get('executive_summary', 'N/A')}

Recommendations:
{analysis_result.get('recommendations', 'N/A')}

Implementation Steps:
{analysis_result.get('implementation_steps', 'N/A')}

TIME ESTIMATION
===============

Total Time: {estimation.get('total_estimation_hours', 'N/A')}
Confidence Level: {estimation.get('confidence_level', 'N/A')}
Complexity Rating: {estimation.get('complexity_rating', 'N/A')}
Required Skill Level: {estimation.get('skill_level_required', 'N/A')}

Breakdown:
{json.dumps(estimation.get('breakdown', {}), indent=2)}

Risk Factors:
{chr(10).join(f"- {risk}" for risk in estimation.get('risk_factors', []))}

Assumptions:
{chr(10).join(f"- {assumption}" for assumption in estimation.get('assumptions', []))}

Recommended Approach:
{estimation.get('recommended_approach', 'N/A')}
"""

        if 'design_solution' in response:
            summary += f"""

DESIGN SOLUTION
===============

{response['design_solution']}
"""

        return summary

    def run(self):
        """Main application runner."""
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