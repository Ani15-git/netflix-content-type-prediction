"""
Netflix Content Type Prediction - Model Training
=================================================

Goal
----
Predict whether a Netflix title is a "Movie" or a "TV Show" (the `type` column).

Data leakage notes (IMPORTANT - two separate leaks are handled here)
--------------------------------------------------------------------
1. The `duration` column is deliberately NOT used as a feature.
   Its value literally gives the answer away:
       "90 min"    -> Movie
       "2 Seasons" -> TV Show
   `show_id` and `title` are also excluded (unique identifiers / free text that
   carry no generalisable signal).

2. The `listed_in` (genres) column is *cleaned* before vectorising. Netflix
   names TV genres "TV Dramas", "Crime TV Shows", ... and movie genres
   "Dramas", "International Movies", ... so the raw genre words leaked the
   label almost perfectly. We strip any token containing movie/tv/show/series/
   stand-up/special and keep the real genre words. See `clean_genres()`.

3. The `director` column is NOT used either. A TF-IDF over director *names* alone
   scored ~94.7% accuracy, i.e. the model was simply memorising which directors
   work on movies vs series (same problem as `title`: a high-cardinality
   identifier that does not generalise to unseen titles).

This script:
    1. Loads and cleans Dataset.csv
    2. Builds features with a scikit-learn ColumnTransformer
    3. Splits the data 80/20 (stratified)
    4. Trains and compares 3 models
    5. Saves the best model to model/best_model.pkl
    6. Saves charts to outputs/
"""

import json
import os
import re

import joblib
import numpy as np
import pandas as pd

# Use a non-interactive backend so the script can save charts on any machine.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_PATH = "dataset.csv"
MODEL_DIR = "model"
OUT_DIR = "outputs"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# Columns we intentionally never feed to the model (see leakage notes above).
LEAKY_COLUMNS = ["duration", "show_id", "title", "director"]

# Tokens inside `listed_in` that leak the label. The Netflix catalogue names TV
# genres "TV Dramas" / "Crime TV Shows" and movie genres "... Movies", so these
# words alone predicted the target almost perfectly (99.9%).
LEAKY_GENRE_SUBSTRINGS = ("movie", "tv", "show", "series", "stand-up", "standup", "special")

# Raw columns that the saved pipeline expects as input.
RAW_INPUT_COLUMNS = ["listed_in", "country", "rating", "release_year", "added_year"]

# Netflix brand colours for a consistent look in the saved charts.
NETFLIX_RED = "#E50914"
DARK = "#141414"
GREY = "#B3B3B3"


# ---------------------------------------------------------------------------
# 0. Leakage removal for the genres column
# ---------------------------------------------------------------------------
def clean_genres(text) -> str:
    """Strip label-leaking words out of a `listed_in` genre string.

    Examples
    --------
        "TV Dramas, TV Mysteries"                  -> "dramas mysteries"
        "Dramas, International Movies, Comedies"   -> "dramas international comedies"
        "Crime TV Shows, TV Action & Adventure"    -> "crime action adventure"
        "Stand-Up Comedy"                          -> "comedy"
        "Docuseries, British"                      -> "british"

    We remove any token containing a leaky substring (movie/tv/show/series/
    stand-up/special) but keep the genuine genre words (drama, comedy, action,
    thriller, documentary, romance, ...) so the model still has real signal.

    NOTE: this same function is imported by app.py, so training and serving
    always apply an identical transformation.
    """
    if not isinstance(text, str):
        return ""
    tokens = re.findall(r"[a-z0-9][a-z0-9&'\-]*", text.lower())
    kept = [t for t in tokens if not any(bad in t for bad in LEAKY_GENRE_SUBSTRINGS)]
    return " ".join(kept)


# ---------------------------------------------------------------------------
# 1. Loading & cleaning
# ---------------------------------------------------------------------------
def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply the minimal cleaning the model needs."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Put Dataset.csv in the project folder "
            f"and run this script from the same folder."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # ---- Fix missing values -------------------------------------------------
    # The bundled dataset is quite clean, but we handle gaps defensively so the
    # script also works on the original (messier) Netflix dataset.
    for col in ["director", "country", "listed_in", "rating"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # release_year -> numeric, missing years filled with the median year.
    df["release_year"] = pd.to_numeric(df.get("release_year"), errors="coerce")
    df["release_year"] = df["release_year"].fillna(df["release_year"].median())

    # Remove rows where the target is missing.
    df = df[df["type"].notna()].copy()

    # ---- Extract the year from date_added (when the column is present) ------
    if "date_added" in df.columns:
        added = pd.to_datetime(df["date_added"], errors="coerce", format="mixed")
        # We only keep the year; the exact day/month adds noise for this task.
        df["added_year"] = added.dt.year
    else:
        df["added_year"] = np.nan
    df["added_year"] = df["added_year"].fillna(df["added_year"].median())

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the feature frame that the pipeline / Streamlit app both use."""
    features = pd.DataFrame(index=df.index)

    # Country is used as-is (word tokens).
    features["country"] = df["country"].fillna("").astype(str)

    # Genres: strip the label-leaking words ("TV", "Movies", "Shows", ...)
    # BEFORE they ever reach the vectoriser.
    features["listed_in"] = df["listed_in"].fillna("").astype(str).map(clean_genres)

    # Categorical text.
    features["rating"] = df["rating"].fillna("Unknown").astype(str)

    # Numeric fields.
    features["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    features["added_year"] = pd.to_numeric(
        df.get("added_year", pd.Series(np.nan, index=df.index)), errors="coerce"
    )

    # NOTE: `director` is intentionally missing here - director names memorise
    # the label (see leakage note 3 in the module docstring).

    return features


# ---------------------------------------------------------------------------
# 2. Feature engineering pipeline
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """A ColumnTransformer that turns raw columns into a numeric feature matrix.

    - `listed_in` -> TF-IDF over the *cleaned* genre words (see clean_genres).
      This is the main text signal, since the dataset has no `description`
      column.
    - `country`           -> TF-IDF over the country words.
    - `rating`            -> one-hot encoding.
    - `release_year`      -> scaled to [0, 1] (also keeps values non-negative
                             so Multinomial Naive Bayes works).
    - `added_year`        -> scaled; missing values fall back to a constant so
                             the Streamlit app can predict without a date.

    NOTE: `director` is deliberately absent (it leaks the label), and we use the
    built-in word tokenizer instead of a custom Python function, because custom
    callables get pickled by reference and would break `joblib.load` in app.py.
    """
    tfidf_listed = TfidfVectorizer(
        lowercase=True, max_features=4000, min_df=2,
    )
    tfidf_country = TfidfVectorizer(
        lowercase=True, max_features=600, min_df=2,
    )

    numeric_transformer = Pipeline(steps=[
        # Constant fallback (≈ median "date_added" year) keeps app input working.
        ("imputer", SimpleImputer(strategy="constant", fill_value=2016)),
        ("scaler", MinMaxScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("genres", tfidf_listed, "listed_in"),
            ("country", tfidf_country, "country"),
            ("rating", OneHotEncoder(handle_unknown="ignore"), ["rating"]),
            ("year", numeric_transformer, ["release_year", "added_year"]),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------
def _style_axes(ax):
    ax.set_facecolor(DARK)
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors=GREY)


def plot_class_distribution(df: pd.DataFrame, out_path: str):
    """Bar chart of how many Movies vs TV Shows are in the dataset."""
    plt.figure(figsize=(5, 3))
    ax = sns.countplot(x="type", data=df, hue="type",
                       palette={"Movie": NETFLIX_RED, "TV Show": GREY},
                       order=["Movie", "TV Show"], legend=False)
    for container in ax.containers:
        ax.bar_label(container, color="white", fontsize=8, fontweight="bold")

    plt.title("Class Distribution (type)", fontsize=11, fontweight="bold")
    plt.xlabel("Type", fontsize=9)
    plt.ylabel("Count", fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gcf().patch.set_facecolor(DARK)
    _style_axes(ax)
    plt.tight_layout(pad=0.6)
    plt.savefig(out_path, dpi=150, facecolor=DARK)
    plt.close()


def plot_confusion_matrix(cm, labels, out_path: str):
    """Heatmap of the best model's confusion matrix."""
    plt.figure(figsize=(5, 3))
    ax = sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds", cbar=False,
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 11, "weight": "bold"},
    )
    plt.title("Confusion Matrix - Best Model", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted", fontsize=9)
    plt.ylabel("Actual", fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8, rotation=0)
    plt.gcf().patch.set_facecolor(DARK)
    _style_axes(ax)
    plt.tight_layout(pad=0.6)
    plt.savefig(out_path, dpi=150, facecolor=DARK)
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, out_path: str):
    """Grouped bar chart comparing all metrics for every model."""
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    plot_df = results_df.set_index("Model")[metrics]

    ax = plot_df.plot(kind="bar", figsize=(5.5, 3), colormap="Reds", width=0.8)
    plt.title("Model Comparison", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=9)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc="lower right", fontsize=7, ncols=2)
    plt.gcf().patch.set_facecolor(DARK)
    _style_axes(ax)
    plt.tight_layout(pad=0.6)
    plt.savefig(out_path, dpi=150, facecolor=DARK)
    plt.close()


def plot_top_features(pipeline, feature_names, model_name: str, out_path: str, top_n=6):
    """Show the most influential features for the winning model.

    Handles the three model families used in this project:
      * Linear models (Logistic Regression) -> coefficients
      * Naive Bayes                          -> difference in log-probabilities
      * Tree models (Random Forest)          -> feature importances
    """
    clf = pipeline.named_steps["classifier"]
    classes = list(clf.classes_)

    if hasattr(clf, "coef_"):  # Logistic Regression
        scores = clf.coef_[0]
        positive_label = classes[1]  # positive coefficient -> this class
        pos_idx = np.argsort(scores)[-top_n:][::-1]
        neg_idx = np.argsort(scores)[:top_n]
        selected = np.concatenate([neg_idx, pos_idx])
        values = scores[selected]
        colors = [GREY] * len(neg_idx) + [NETFLIX_RED] * len(pos_idx)
        title = f"Top Features - {model_name}\n(red -> {positive_label}, grey -> {classes[0]})"
    elif hasattr(clf, "feature_log_prob_"):  # Naive Bayes
        scores = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        positive_label = classes[1]
        pos_idx = np.argsort(scores)[-top_n:][::-1]
        neg_idx = np.argsort(scores)[:top_n]
        selected = np.concatenate([neg_idx, pos_idx])
        values = scores[selected]
        colors = [GREY] * len(neg_idx) + [NETFLIX_RED] * len(pos_idx)
        title = f"Top Features - {model_name}\n(red -> {positive_label}, grey -> {classes[0]})"
    else:  # Random Forest / other tree models
        scores = clf.feature_importances_
        selected = np.argsort(scores)[-top_n:][::-1]
        values = scores[selected]
        colors = [NETFLIX_RED] * len(selected)
        title = f"Top Features - {model_name}"

    labels = [feature_names[i] for i in selected]

    plt.figure(figsize=(5.5, 3.2))
    plt.barh(range(len(selected)), values, color=colors)
    plt.yticks(range(len(selected)), labels, fontsize=8)
    plt.xticks(fontsize=8)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=9, fontweight="bold")
    plt.xlabel("Importance / Coefficient", fontsize=9)
    plt.gcf().patch.set_facecolor(DARK)
    _style_axes(plt.gca())
    plt.tight_layout(pad=0.6)
    plt.savefig(out_path, dpi=150, facecolor=DARK)
    plt.close()


# ---------------------------------------------------------------------------
# 4. Main training routine
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    # ---- Load & clean -------------------------------------------------------
    print("=" * 70)
    print("NETFLIX CONTENT TYPE PREDICTION - TRAINING")
    print("=" * 70)

    df = load_and_clean(DATA_PATH)
    print(f"\nDataset shape       : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target classes      : {df['type'].value_counts().to_dict()}")
    print(f"Leaky cols excluded : {LEAKY_COLUMNS}")
    print("Genre cleaning demo :")
    for raw in ["TV Dramas, TV Mysteries", "Dramas, International Movies",
                "Crime TV Shows, TV Action & Adventure", "Stand-Up Comedy"]:
        print(f"    {raw!r:45s} -> {clean_genres(raw)!r}")

    # ---- Features & target --------------------------------------------------
    X = build_features(df)
    y = df["type"].astype(str)

    # Stratified split keeps the Movie / TV Show ratio identical in train & test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train / Test split  : {len(X_train)} / {len(X_test)} (stratified 80/20)")

    # ---- Chart 1: class distribution ---------------------------------------
    plot_class_distribution(df, os.path.join(OUT_DIR, "class_distribution.png"))
    print(f"\nSaved chart -> {os.path.join(OUT_DIR, 'class_distribution.png')}")

    # ---- Candidate models ---------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=5.0, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Naive Bayes": MultinomialNB(alpha=0.3),
    }

    results = []
    pipelines = {}

    for name, clf in models.items():
        print(f"\n--- Training {name} ...")
        pipe = Pipeline(steps=[("preprocessor", build_preprocessor()),
                               ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        results.append(metrics)
        pipelines[name] = pipe

        print(f"    Accuracy : {metrics['Accuracy']:.4f}")
        print(f"    Precision: {metrics['Precision']:.4f}")
        print(f"    Recall   : {metrics['Recall']:.4f}")
        print(f"    F1 score : {metrics['F1']:.4f}")

    # ---- Comparison table ---------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by F1)")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    results_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)

    # ---- Pick the best model (highest weighted F1 -> balances both classes) --
    best_name = results_df.iloc[0]["Model"]
    best_pipeline = pipelines[best_name]
    print(f"\nBEST MODEL -> {best_name} (F1 = {results_df.iloc[0]['F1']:.4f})")

    # ---- Charts for the best model -----------------------------------------
    y_pred_best = best_pipeline.predict(X_test)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred_best, labels=labels)
    plot_confusion_matrix(cm, labels, os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_model_comparison(results_df, os.path.join(OUT_DIR, "model_comparison.png"))

    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    plot_top_features(
        best_pipeline, feature_names, best_name,
        os.path.join(OUT_DIR, "top_features.png"),
    )
    print(f"Saved charts -> {OUT_DIR}/ (class_distribution, confusion_matrix, "
          f"model_comparison, top_features)")

    # ---- Persist the best pipeline -----------------------------------------
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"Saved best model -> {BEST_MODEL_PATH}")

    # ---- Small metadata file used by the Streamlit dashboard ---------------
    run_info = {
        "best_model": str(best_name),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(X_train)),
        "accuracy": float(results_df.iloc[0]["Accuracy"]),
        "f1": float(results_df.iloc[0]["F1"]),
    }
    with open(os.path.join(OUT_DIR, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, indent=2)
    print(f"Saved run info   -> {os.path.join(OUT_DIR, 'run_info.json')}")

    print("\nDone. Next step:  streamlit run app.py")


if __name__ == "__main__":
    main()
