# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

st.set_page_config(page_title="Driver Monitoring", layout="wide")

# --- CONFIG ---
RECORDINGS_DIR = Path("recordings")

def find_sessions():
    """Finds all session.csv files and their matching .avi files."""
    sessions = []
    if not RECORDINGS_DIR.exists():
        return sessions
    
    csv_files = list(RECORDINGS_DIR.glob("session_*.csv"))
    regex = re.compile(r"session_(\d{8})_(\d{6})")
    
    for csv_path in csv_files:
        video_path = csv_path.with_suffix(".avi")
        if video_path.exists():
            match = regex.search(csv_path.name)
            if match:
                dt_str = match.group(1) + match.group(2)
                session_dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                sessions.append({
                    "name": session_dt.strftime("%Y-%m-%d %I:%M:%S %p"),
                    "dt": session_dt,
                    "log_path": csv_path,
                    "video_path": video_path
                })
    
    # Sort by datetime, newest first
    sessions.sort(key=lambda x: x["dt"], reverse=True)
    return sessions

# --- Main Page UI ---
st.title("🚗 Driver Monitoring System")
st.header("Session Selector")

sessions = find_sessions()

if not sessions:
    st.error("No sessions found in the 'recordings' folder.")
    st.info("Run `detect.py` to record a new session. Ensure both .avi and .csv files are present.")
    st.stop()

# --- Initialize session state ---
if "selected_session" not in st.session_state:
    st.session_state.selected_session = sessions[0] # Default to newest

# Create a dictionary for the selectbox options
session_options = {s["name"]: s for s in sessions}

selected_name = st.selectbox(
    "Select a session to review:",
    options=session_options.keys(),
    index=0
)

# Update session state when selection changes
if selected_name:
    st.session_state.selected_session = session_options[selected_name]

# Display info about the selected session
sess = st.session_state.selected_session
st.subheader("Selected Session Details")
st.text(f"Log File:   {sess['log_path'].name}")
st.text(f"Video File: {sess['video_path'].name}")
st.text(f"Recorded:   {sess['dt'].strftime('%Y-%m-%d %I:%M %p')}")

st.success("Session loaded. Use the sidebar to navigate to the **Dashboard** or **Summary**.")

st.sidebar.info("Select a session on the **Home** page first.")