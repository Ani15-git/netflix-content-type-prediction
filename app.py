"""
Netflix Content Type Prediction - Streamlit App
===============================================

A compact, Netflix-styled (dark + red) dashboard that predicts whether a title
is a "Movie" or a "TV Show". The whole page is designed to fit on one laptop
screen (1366x768) without scrolling.

Run it with:   streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# `clean_genres` is imported from train_model so that training and serving apply
# exactly the same leakage-removal transform to the genres.
from train_model import RAW_INPUT_COLUMNS, clean_genres

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("model", "best_model.pkl")
OUT_DIR = "outputs"
METRICS_PATH = os.path.join(OUT_DIR, "model_metrics.csv")
RUN_INFO_PATH = os.path.join(OUT_DIR, "run_info.json")

# Ratings that appear in the Netflix dataset (used to fill the dropdown).
RATING_OPTIONS = [
    "TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-G",
    "TV-Y", "PG", "G", "NC-17", "NR", "TV-Y7-FV", "UR",
]

st.set_page_config(
    page_title="Netflix Content Type Prediction",
    page_icon="🎬",
    layout="wide",                       # full width -> fits one screen
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Compact Netflix-style theme
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* ---------- hide Streamlit chrome (menu / footer / deploy) ---------- */
        #MainMenu, header, footer, [data-testid="stHeader"],
        [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"], .stDeployButton {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        /* ---------- background + tight page frame ---------- */
        .stApp {
            background: radial-gradient(circle at 15% 0%, #2b0b0d 0%, #141414 42%, #0b0b0b 100%);
            color: #ffffff;
        }
        .block-container {
            padding: 0.5rem 1.1rem 0.4rem 1.1rem !important;
            max-width: 100% !important;
        }
        /* tighten vertical rhythm everywhere */
        div[data-testid="stVerticalBlock"] { gap: 0.45rem; }
        div[data-testid="stHorizontalBlock"] { gap: 0.7rem; }
        html, body, [class*="css"] { font-size: 13px; }
        h1, h2, h3, h4 { color: #ffffff !important; }

        /* ---------- single-line header ---------- */
        .hero-title {
            font-size: 1.12rem; font-weight: 900; color: #E50914;
            letter-spacing: 0.4px; text-shadow: 0 2px 14px rgba(229,9,20,0.45);
        }
        .hero-sub { color: #a3a3a3; font-size: 0.72rem; }

        /* ---------- form card ---------- */
        div[data-testid="stForm"] {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 10px 12px 4px 12px;
        }
        .stTextArea textarea {
            height: 54px !important; min-height: 54px !important;
            font-size: 0.78rem !important; background-color: #1f1f1f !important;
            color: #fff !important; border-radius: 9px !important; border: 1px solid #333 !important;
        }
        .stTextInput input, .stNumberInput input {
            font-size: 0.78rem !important; padding: 0.3rem 0.5rem !important;
            background-color: #1f1f1f !important; color: #fff !important;
            border-radius: 9px !important; border: 1px solid #333 !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            font-size: 0.78rem !important; min-height: 34px;
            background-color: #1f1f1f !important; color: #fff !important;
            border-radius: 9px !important; border: 1px solid #333 !important;
        }
        div[data-testid="stWidgetLabel"] p {
            font-size: 0.7rem !important; margin-bottom: 0.05rem !important; color: #cfcfcf !important;
        }

        /* ---------- Netflix red buttons ---------- */
        div[data-testid="stFormSubmitButton"] button, .stButton > button {
            background: linear-gradient(135deg, #E50914 0%, #b1060f 100%);
            color: #ffffff; border: none; border-radius: 10px;
            font-weight: 700; font-size: 0.85rem; width: 100%;
            padding: 0.42rem 0.8rem; margin-top: 2px;
            box-shadow: 0 5px 16px rgba(229,9,20,0.35);
        }
        div[data-testid="stFormSubmitButton"] button:hover, .stButton > button:hover {
            color: #ffffff; box-shadow: 0 8px 22px rgba(229,9,20,0.55);
        }

        /* ---------- compact metric cards ---------- */
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px; padding: 6px 10px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35);
        }
        [data-testid="stMetricLabel"] { font-size: 0.64rem !important; color: #b3b3b3 !important; }
        [data-testid="stMetricValue"] { font-size: 0.98rem !important; color: #ffffff !important; }

        /* ---------- tabs ---------- */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.73rem; padding: 5px 10px; color: #b3b3b3;
            background: rgba(255,255,255,0.05); border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] { background: rgba(229,9,20,0.22); color: #ffffff !important; }
        .stTabs [data-baseweb="tab-panel"] { padding-top: 6px; }

        /* ---------- chart sizing (keeps charts small on screen) ---------- */
        [data-testid="stImage"] img {
            max-height: 262px; width: 100%; object-fit: contain;
            border-radius: 10px; background: #141414;
        }

        /* ---------- result card ---------- */
        .result-box {
            margin-top: 10px; padding: 14px 16px; border-radius: 14px;
            background: linear-gradient(135deg, rgba(229,9,20,0.18), rgba(229,9,20,0.05));
            border: 1px solid rgba(229,9,20,0.5);
            box-shadow: 0 8px 26px rgba(229,9,20,0.22);
        }
        .result-label { font-size: 0.62rem; letter-spacing: 2px; color: #b3b3b3; }
        .result-type { font-size: 1.75rem; font-weight: 900; color: #ffffff; line-height: 1.15; }
        .result-conf { font-size: 0.8rem; font-weight: 700; color: #E50914; }
        .bar-wrap {
            background: rgba(255,255,255,0.12); border-radius: 8px;
            height: 9px; overflow: hidden; margin-top: 8px;
        }
        .bar-fill {
            height: 100%; border-radius: 8px;
            background: linear-gradient(90deg, #E50914, #ff4b4b);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained pipeline saved by train_model.py."""
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Model comparison table (one row per model)."""
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return None


def load_run_info():
    """Small metadata file: best model name, accuracy, F1, test size."""
    if os.path.exists(RUN_INFO_PATH):
        with open(RUN_INFO_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def build_input_row(genres, description, country, rating, release_year):
    """Create the single-row DataFrame the pipeline expects.

    The dataset has no `description` column, so free text entered in the app is
    appended to the (cleaned) genres and fed through the same TF-IDF vocabulary.
    Words that are not part of the learned genre vocabulary are simply ignored.
    """
    text = f"{clean_genres(genres)} {(description or '').lower()}".strip()
    return pd.DataFrame([{
        "listed_in": text,
        "country": country,
        "rating": rating,
        "release_year": release_year,
        "added_year": np.nan,   # not collected in the form -> imputed by pipeline
    }])[RAW_INPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
inject_css()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:12px; margin-bottom:6px;">'
    '  <span class="hero-title">🎬 NETFLIX CONTENT TYPE PREDICTION</span>'
    '  <span class="hero-sub">Movie or TV Show? &nbsp;·&nbsp; TF-IDF + scikit-learn '
    '&nbsp;·&nbsp; leakage-free features</span>'
    '</div>',
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error("**Model not found.** Train it first:\n\n```bash\npython train_model.py\n```")
    st.stop()

model = load_model()
metrics = load_metrics()
run_info = load_run_info()

left, right = st.columns([0.4, 0.6], gap="medium")

# ===========================================================================
# LEFT - compact prediction form + result
# ===========================================================================
with left:
    with st.form("predict_form"):
        description = st.text_area(
            "Description (optional)",
            placeholder="Short plot / keywords, e.g. a family finds a hidden world",
        )

        row1 = st.columns(2)
        genres = row1[0].text_input("Genre(s)", placeholder="Dramas, Comedies")
        country = row1[1].text_input("Country", placeholder="United States")

        row2 = st.columns(2)
        rating = row2[0].selectbox("Rating", RATING_OPTIONS, index=0)
        release_year = row2[1].number_input(
            "Release year", min_value=1925, max_value=2025, value=2019, step=1
        )

        submitted = st.form_submit_button("🎬  Predict")

    prediction = None
    confidence = None
    proba = None

    if submitted:
        if not (genres.strip() or description.strip()):
            st.warning("Enter at least a genre or a description.")
        else:
            row_df = build_input_row(genres, description, country, rating, release_year)
            prediction = model.predict(row_df)[0]

            # Confidence = probability of the predicted class.
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(row_df)[0]
                classes = list(model.classes_)
                confidence = float(proba[classes.index(prediction)] * 100)

        if prediction is not None:
            bar_width = confidence if confidence is not None else 0
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">PREDICTION</div>
                    <div class="result-type">{prediction}</div>
                    <div class="result-conf">Confidence: {bar_width:.1f}%</div>
                    <div class="bar-wrap"><div class="bar-fill" style="width:{bar_width:.1f}%"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if proba is not None:
                st.caption(
                    "  ·  ".join(f"{c}: {p * 100:.1f}%" for c, p in zip(model.classes_, proba))
                )

    st.caption(
        "Leak-free by design: `duration`, `title`, `show_id` and `director` are never used, "
        "and genres are stripped of the words *movie / tv / show / series*."
    )

# ===========================================================================
# RIGHT - metrics + charts in tabs
# ===========================================================================
with right:
    # ---- 4 small metric cards ------------------------------------------------
    if run_info:
        best_model = run_info.get("best_model", "-")
        accuracy = run_info.get("accuracy", 0.0)
        f1 = run_info.get("f1", 0.0)
        test_samples = run_info.get("test_samples", 0)
    elif metrics is not None:
        best = metrics.iloc[0]
        best_model, accuracy, f1, test_samples = best["Model"], best["Accuracy"], best["F1"], "-"
    else:
        best_model, accuracy, f1, test_samples = "-", 0.0, 0.0, "-"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", str(best_model))
    m2.metric("Accuracy", f"{accuracy * 100:.2f}%")
    m3.metric("F1 Score", f"{f1 * 100:.2f}%")
    m4.metric("Test Samples", str(test_samples))

    # ---- charts inside tabs --------------------------------------------------
    tab_cmp, tab_cm, tab_dist, tab_feat = st.tabs(
        ["Model Comparison", "Confusion Matrix", "Class Distribution", "Top Features"]
    )

    chart_tabs = [
        (tab_cmp, "model_comparison.png", "Accuracy / Precision / Recall / F1 per model"),
        (tab_cm, "confusion_matrix.png", "Best model on the 20% hold-out set"),
        (tab_dist, "class_distribution.png", "Movie vs TV Show in the dataset"),
        (tab_feat, "top_features.png", "Most influential features of the best model"),
    ]
    for tab, filename, caption in chart_tabs:
        with tab:
            path = os.path.join(OUT_DIR, filename)
            if os.path.exists(path):
                st.image(path, width="stretch")
                st.caption(caption)
            else:
                st.info(f"`{filename}` not found - run `python train_model.py`.")

    if metrics is not None:
        st.caption(
            "Leakage fix: genre words containing *movie / tv / show / series / stand-up / special* "
            "are removed, and `duration` + director names are excluded."
        )
