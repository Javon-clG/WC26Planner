# WC26 Planner POC (Local Streamlit)

## Setup
```bash

Navigate to the Folder where the files live when you run it locally, and do the following underneath:

cd wc26_poc
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data
All CSVs live in `data/`.
- match_catalog.csv is the primary table used by the app.
- itineraries.csv controls the dropdown of people/trips.
- itinerary_matches.csv is optional local persistence (you can also just export CSVs).

## Share/Compare
Export an itinerary CSV and use the Compare tab to upload multiple exports for side-by-side totals + conflict checks.
