# Start Streamlit using the virtualenv Python to avoid a broken streamlit.exe launcher
# Usage: Open PowerShell, activate the venv or run directly from the repo root.
#   & .\start_streamlit.ps1

# If you want to activate the venv first:
# & .\.venv\Scripts\Activate.ps1
# Then run Streamlit:
# python -m streamlit run app.py

# Run Streamlit directly via the venv Python (no activation needed):
& .\.venv\Scripts\python.exe -m streamlit run app.py
