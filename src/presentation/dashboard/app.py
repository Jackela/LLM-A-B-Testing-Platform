"""Main Streamlit dashboard application."""

import logging
from typing import Any, Dict

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LLM A/B Testing Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-align: center;
        color: #1f77b4;
    }
    
    .metric-card {
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-completed {
        color: #17a2b8;
        font-weight: bold;
    }
    
    .status-failed {
        color: #dc3545;
        font-weight: bold;
    }
    
    .winner-model {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "selected_test" not in st.session_state:
    st.session_state.selected_test = None


def main():
    """Main dashboard application."""

    # Authentication check
    if not st.session_state.authenticated:
        show_login_page()
        return

    # Main dashboard
    show_dashboard()


def show_login_page():
    """Display login page."""
    st.markdown(
        '<div class="main-header">🧪 LLM A/B Testing Platform</div>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Login")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                # Simple authentication (in production, use proper authentication)
                if username in ["admin", "user", "viewer"] and password in [
                    "admin123",
                    "user123",
                    "viewer123",
                ]:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        # Demo credentials
        with st.expander("Demo Credentials"):
            st.write("**Admin:** admin / admin123")
            st.write("**User:** user / user123")
            st.write("**Viewer:** viewer / viewer123")


def show_dashboard():
    """Display main dashboard."""

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            '<div class="main-header">🧪 LLM A/B Testing Platform</div>', unsafe_allow_html=True
        )
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"**Welcome, {st.session_state.username}!**")
        st.markdown("---")

        page = st.selectbox(
            "Navigate to:",
            ["Overview", "Create Test", "Test Results", "Model Comparison", "Settings"],
            index=0,
        )

    # Display selected page
    if page == "Overview":
        show_overview_page()
    elif page == "Create Test":
        show_create_test_page()
    elif page == "Test Results":
        show_test_results_page()
    elif page == "Model Comparison":
        show_comparison_page()
    elif page == "Settings":
        show_settings_page()


def show_overview_page():
    """Display overview dashboard."""
    from .pages.overview import render_overview_page

    render_overview_page()


def show_create_test_page():
    """Display test creation page."""
    from .pages.create_test import render_create_test_page

    render_create_test_page()


def show_test_results_page():
    """Display test results page."""
    from .pages.test_results import render_test_results_page

    render_test_results_page()


def show_comparison_page():
    """Display model comparison page."""
    from .pages.comparison import render_comparison_page

    render_comparison_page()


def show_settings_page():
    """Display settings page."""
    from .pages.settings import render_settings_page

    render_settings_page()


if __name__ == "__main__":
    main()
