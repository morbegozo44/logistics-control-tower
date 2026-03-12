# Logistics Control Tower Prototype (Streamlit)

## What it is
A lightweight, independent web app prototype with 3 tabs:
- Map: terminals + problem lanes (late % / cost per mile)
- Recommendations: scenario simulator (Add Drivers / New Facility / Rebalance)
- Financial Impact: baseline vs scenario + ROI/payback

## Run locally
1) Install Python 3.10+
2) In this folder:
   pip install -r requirements.txt
3) Start:
   streamlit run app.py

## Use your data
Upload your Excel/CSV in the sidebar.

Expected columns (case-insensitive):
- Origin, Destination
- Truck Miles (or Total Miles)
- Toll Cost
- Arrival Status (contains "Late" for late trips)

Tip: Use consistent terminal names like "Boston, MA" so the map can place them.
