# app.py

import os
import requests
import streamlit as st
import pandas as pd

# ------------------------- Config -------------------------
API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8011")
TOTAL_ITEMS = int(os.getenv("TOTAL_ITEMS", "1682"))  # MovieLens 100K default

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="RecSys Demo", layout="wide")
st.title("🎯 Two-Tower Recommendation Demo")

st.sidebar.header("User Controls")
user_id = st.sidebar.number_input("User ID (0-based index)", min_value=0, step=1, value=0)
top_k   = st.sidebar.slider("Top K Recommendations", 1, 50, 5)

# ----------------------- API Calls ------------------------
@st.cache_data(ttl=300)
def fetch_recs(user_id: int, top_k: int, total_items: int):
    """Call the LitServe endpoint and return the parsed results list."""
    payload = {
        "user_ids": [user_id] * total_items,
        "item_ids": list(range(total_items)),
        "top_k": top_k
    }
    resp = requests.post(f"{API_URL}/predict", json=payload)
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code}: {resp.text}")
    data = resp.json()

    # Handle both old ("scores") and new ("results") formats just in case
    if "results" in data:
        print("it's results")
        print(data['results'][:5])
        return data["results"]
    elif "scores" in data:
        print("it's scores")
        # Fallback (client-side sort); shouldn't happen with new server
        import torch
        scores = data["scores"]
        idxs = torch.tensor(scores).argsort(descending=True)[:top_k].tolist()
        return [
            {"item_idx": int(i), "score": float(scores[i])}
            for i in idxs
        ]
    else:
        raise KeyError(f"Unexpected response keys: {list(data.keys())}")

# ----------------------- Main Action ----------------------
if st.sidebar.button("Get Recommendations"):
    try:
        results = fetch_recs(user_id, top_k, TOTAL_ITEMS)
    except Exception as e:
        st.error(f"Error fetching recommendations: {e}")
    else:
        st.subheader(f"Top {top_k} Recommendations for User {user_id}")

        # Convert to DataFrame for nicer display / sorting
        df = pd.DataFrame(results)

        # Reorder columns if present
        ordered_cols = [c for c in ["movie_id", "title", "item_idx", "score"] if c in df.columns]
        df = df[ordered_cols]

        st.dataframe(df, use_container_width=True)

# ------------------- Health Check Sidebar -----------------
st.sidebar.markdown("---")
try:
    r = requests.get(f"{API_URL}/health", timeout=2)
    if r.ok:
        st.sidebar.success("API status: OK")
    else:
        st.sidebar.error(f"API status code: {r.status_code}")
except Exception as e:
    st.sidebar.warning(f"API unreachable: {e}")