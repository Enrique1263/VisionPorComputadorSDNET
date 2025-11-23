# Web folder
This folder contains the web application code for the project. It includes all the necessary files and directories to run the web server, which will be held and coded in streamlit (python).

## Running the demo

 - **Install dependencies:** `pip install -r web/requirements.txt`
 - **Run locally:** `streamlit run web/app.py`
 - **Project memory:** See `web/memory.md` for a short blog-like timeline and notes about the project's path.

The app attempts to load a model from the `models/` folder if available. If no framework is installed it uses a simple edge-density heuristic so you can test the UI and flow without heavy ML dependencies.