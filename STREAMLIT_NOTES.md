Issue: Running `streamlit run app.py` from PowerShell failed with:

"Fatal error in launcher: Unable to create process using '"D:\Projects\rag-system\.venv\Scripts\python.exe" "D:\Projects\rag-systems\migracion\.venv\Scripts\streamlit.exe" run app.py': The system cannot find the file specified."

Cause: The streamlit.exe launcher in the virtualenv contains an embedded path to a different Python executable (`D:\Projects\rag-system\.venv\Scripts\python.exe`) which doesn't exist on this machine. This can happen if the virtualenv was copied or renamed.

Workarounds:

- Run Streamlit using the venv Python directly: `.venv\Scripts\python.exe -m streamlit run app.py` (recommended)
- Use the included `start_streamlit.ps1` helper script which runs the command above.

Permanent fixes:

- Recreate the virtualenv in-place: `python -m venv .venv` then reinstall dependencies from `requirements.txt`.
- Reinstall Streamlit in the venv to regenerate correct launchers: `.venv\Scripts\pip.exe install --force-reinstall streamlit`

Notes:

- I verified Streamlit is installed (version 1.48.1) and started the app using the venv python.
- If you prefer, I can recreate the venv now and reinstall packages; say the word and I'll proceed.
