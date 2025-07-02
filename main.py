#!/usr/bin/env python3
"""
Main entry point for the Streamlit Odoo GAP Analysis Application.

This module serves as the entry point for running the Streamlit application.
Run with: streamlit run main.py

Usage:
    streamlit run main.py
    or
    python main.py (if using streamlit run internally)
"""

import sys
import os

# Add the current directory to the Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlit_app import StreamlitOdooGAPApp


def main():
    """
    Main function to initialize and run the Streamlit Odoo GAP Analysis application.

    This function creates an instance of the StreamlitOdooGAPApp class and runs it.
    It includes error handling to gracefully handle initialization failures.
    """
    try:
        # Create and run the Streamlit application
        app = StreamlitOdooGAPApp()
        app.run()
    except ImportError as e:
        import streamlit as st
        st.error(f"❌ Import Error: {str(e)}")
        st.error("Please ensure all required dependencies are installed.")
        st.stop()
    except Exception as e:
        import streamlit as st
        st.error(f"❌ Failed to initialize application: {str(e)}")
        st.exception(e)
        st.stop()


if __name__ == "__main__":
    main()
