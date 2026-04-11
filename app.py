"""
app.py — Streamlit GUI for editing Job Agent configuration.

Run with: streamlit run app.py
"""

import streamlit as st
import yaml
from pathlib import Path
import os


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():
    """Load config from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def save_config(config):
    """Save config to YAML file."""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    st.title("🚀 Job Agent Configuration Editor")
    st.markdown("Easily configure your job search preferences and profile.")

    # Load current config
    config = load_config()

    # Initialize defaults if missing
    if 'profile' not in config:
        config['profile'] = {}
    if 'preferences' not in config:
        config['preferences'] = {}
    if 'scoring' not in config:
        config['scoring'] = {}

    with st.form("config_form"):
        st.header("👤 Profile")

        # Name
        config['profile']['name'] = st.text_input(
            "Your Name",
            value=config['profile'].get('name', ''),
            help="Your full name"
        )

        # Resume options
        st.subheader("Resume")
        resume_option = st.radio(
            "How would you like to provide your resume?",
            ["Upload PDF", "Paste Text"],
            index=0 if config['profile'].get('resume_file') else 1
        )

        if resume_option == "Upload PDF":
            uploaded_file = st.file_uploader("Upload your resume PDF", type=['pdf'])
            if uploaded_file is not None:
                # Save uploaded PDF
                pdf_path = Path(__file__).parent / "resume.pdf"
                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                config['profile']['resume_file'] = str(pdf_path)
                config['profile']['resume'] = None  # Clear text resume
                st.success(f"PDF saved to: {pdf_path}")
        else:
            config['profile']['resume'] = st.text_area(
                "Resume Text",
                value=config['profile'].get('resume', ''),
                height=200,
                help="Paste your resume as plain text"
            )
            config['profile']['resume_file'] = None

        # Bio
        config['profile']['bio'] = st.text_area(
            "Bio",
            value=config['profile'].get('bio', ''),
            height=100,
            help="2-3 sentences about your background"
        )

        st.header("🎯 Preferences")

        # Job titles
        st.subheader("Job Titles")
        titles = st.text_area(
            "Desired job titles (one per line)",
            value='\n'.join(config['preferences'].get('titles', [])),
            height=100,
            help="Job titles you're interested in"
        )
        config['preferences']['titles'] = [t.strip() for t in titles.split('\n') if t.strip()]

        # Desired skills
        st.subheader("Desired Skills")
        skills = st.text_area(
            "Skills you want to use (one per line)",
            value='\n'.join(config['preferences'].get('desired_skills', [])),
            height=100,
            help="Technical skills and technologies"
        )
        config['preferences']['desired_skills'] = [s.strip() for s in skills.split('\n') if s.strip()]

        # Hard no keywords
        st.subheader("Hard No Keywords")
        hard_no = st.text_area(
            "Keywords that automatically disqualify jobs (one per line)",
            value='\n'.join(config['preferences'].get('hard_no_keywords', [])),
            height=100,
            help="Jobs with these phrases will be skipped"
        )
        config['preferences']['hard_no_keywords'] = [k.strip() for k in hard_no.split('\n') if k.strip()]

        # Location
        st.subheader("Location Preferences")
        config['preferences']['location'] = {}
        config['preferences']['location']['remote_ok'] = st.checkbox(
            "Remote work OK",
            value=config['preferences'].get('location', {}).get('remote_ok', True)
        )

        locations = st.text_area(
            "Preferred locations (one per line)",
            value='\n'.join(config['preferences'].get('location', {}).get('preferred_locations', [])),
            height=100,
            help="Cities or regions you prefer"
        )
        config['preferences']['location']['preferred_locations'] = [l.strip() for l in locations.split('\n') if l.strip()]

        # Compensation
        st.subheader("Compensation")
        config['preferences']['compensation'] = {}
        config['preferences']['compensation']['min_salary'] = st.number_input(
            "Minimum Salary ($)",
            value=config['preferences'].get('compensation', {}).get('min_salary', 100000),
            min_value=0,
            step=5000,
            help="Minimum annual salary you're willing to accept"
        )

        # Years of experience
        config['preferences']['yoe'] = st.number_input(
            "Years of Experience",
            value=config['preferences'].get('yoe', 0),
            min_value=0,
            max_value=50,
            help="Your years of professional experience"
        )

        st.header("📊 Scoring")

        # Min display score
        config['scoring']['min_display_score'] = st.slider(
            "Minimum Display Score",
            min_value=0,
            max_value=100,
            value=config['scoring'].get('min_display_score', 60),
            help="Only show jobs that score above this threshold"
        )

        # Submit button
        if st.form_submit_button("💾 Save Configuration"):
            save_config(config)
            st.success("Configuration saved successfully! 🎉")
            st.balloons()

    # Display current config
    with st.expander("📄 View Current Configuration"):
        st.code(yaml.dump(config, default_flow_style=False), language='yaml')


if __name__ == "__main__":
    main()