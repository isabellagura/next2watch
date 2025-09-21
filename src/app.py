# Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter

# App config
st.set_page_config(page_title="Next2Watch", page_icon="🎬", layout="wide")

# Constants
TOP_N = 10
DATA_PATH = "data/processed/movies_clean.parquet"

# Load data (cached)
@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

# Feature builders (cached)
@st.cache_resource(show_spinner=False)
def build_features(movies: pd.DataFrame):
    # Text features (combination of title + overview)
    text_data = (movies['title_norm'].astype('string').str.lower() + ' ' + movies['overview_norm'].astype('string').str.lower())

    # Text features: TF-IDF
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
    text_matrix = tfidf_vectorizer.fit_transform(text_data).astype(np.float32)

    # Keyword features: binary tokens
    keyword_vectorizer = CountVectorizer(binary=True, min_df=3)
    keyword_matrix = keyword_vectorizer.fit_transform(movies['keywords_norm'].astype('string').str.lower()).astype(np.float32)

    # Meta features (combination of decade + length form): binary tokens
    meta_vectorizer = CountVectorizer(binary=True)
    meta_tokens = (movies['decade_token'].astype('string') + ' ' + movies['form_token'].astype('string')).str.lower()
    meta_matrix = meta_vectorizer.fit_transform(meta_tokens).astype(np.float32)

    # Genres: binary tokens
    genre_vectorizer = CountVectorizer(binary=True)
    genre_matrix = genre_vectorizer.fit_transform(movies['genres_norm'].fillna('').astype('string').str.lower()).astype(np.float32)

    # Countries: binary tokens
    country_vectorizer = CountVectorizer(binary=True)
    country_matrix = country_vectorizer.fit_transform(movies['countries_norm'].fillna('').astype('string').str.lower()).astype(np.float32)

    # Companies: binary tokens
    company_vectorizer = CountVectorizer(binary=True)
    company_matrix = company_vectorizer.fit_transform(movies['companies_norm'].fillna('').astype('string').str.lower()).astype(np.float32)

    # Language: one-hot encode
    try:
        lang_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        lang_encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)

    lang_matrix = lang_encoder.fit_transform(movies[['original_language']]).astype(np.float32)

    matrices = {
        "text": text_matrix,
        "keyword": keyword_matrix,
        "meta": meta_matrix,
        "genre": genre_matrix,
        "country": country_matrix,
        "company": company_matrix,
        "lang": lang_matrix,

    }
    return matrices

# Column labels for display
DISPLAY_LABELS = {
    'id': 'ID',
    'title': 'Title',
    'release_year': 'Year',
    'genres_norm': 'Genres',
    'keywords_norm':'Keywords',
    'original_language': 'Language',
    'countries_norm': 'Countries',
    'companies_norm': 'Companies',
    'similarity': 'Similarity Score',
    'vote_count': 'Vote Count',
    'popularity': 'Popularity'
}

def as_pretty(df: pd.DataFrame, labels=None):
    labels = labels or DISPLAY_LABELS
    return df.rename(columns=labels)

# Search function - maybe for dropdown results to display

# SELECT MOVIE
# Pick base movie by normalized title (exact match first, then substring)
def get_base_index(movies: pd.DataFrame, title: str):
    query = str(title).strip().casefold()
    titles_norm = movies['title_norm'].astype('string').str.casefold().str.strip()

    # pick base movie row
    matches = movies.index[titles_norm.eq(query)]
    if len(matches) == 0:
        matches = movies.index[titles_norm.str.contains(query, regex=False)]
        if len(matches) == 0:
            return None, None  # signal "no match"

    # Break ties by vote_count then popularity.
    base_index = movies.loc[matches, ['vote_count','popularity']].fillna(0).sort_values(['vote_count','popularity'], ascending=False).index[0]
    base_pos = movies.index.get_loc(base_index)
    return base_index, base_pos

# COSINE SIMILARITIES
# Compute per-field cosine similarities from the base position
def compute_similarities(base_pos: int, matrices: dict) -> dict:
    return {
        feature: cosine_similarity(matrices[feature][base_pos], matrices[feature]).ravel()
        for feature in ("text", "keyword", "genre", "meta", "lang", "country", "company")
    }

# RECOMMENDER
# Movie recommender (cosine + small popularity bump)
def recommend(movies: pd.DataFrame, matrices: dict, title: str, top_n: int = TOP_N):

    display_cols = ['title','similarity','release_year','genres_norm','keywords_norm','original_language','countries_norm','companies_norm','popularity','vote_count']

    base_index, base_pos = get_base_index(movies, title)

    # handle case when there is no match for movie
    if base_index is None or base_pos is None:
        empty = pd.DataFrame(columns=display_cols)
        return empty, None, f"No recommendations to show for '{title}'."

    sims = compute_similarities(base_pos, matrices)

    # weighted blend
    score = (0.25 * sims['text'] +
             0.25 * sims['keyword'] +
             0.20 * sims['genre'] +
             0.12 * sims['meta'] +
             0.08 * sims['lang'] +
             0.06 * sims['country'] +
             0.04 * sims['company'])

    # don't recommend the same movie
    score[base_pos] = -1.0

    # small popularity bump
    pop = movies['popularity'].to_numpy(dtype='float64')
    pop_min, pop_max = np.nanmin(pop), np.nanmax(pop)
    pop_scaled = (pop - pop_min) / (pop_max - pop_min + 1e-9)
    score = score + 0.05 * pop_scaled
    score[base_pos] = -1.0

    # top-N by score
    top_idx = np.argpartition(score, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(score[top_idx])[::-1]]

    results = movies.iloc[top_idx][
        ['title','release_year','genres_norm','keywords_norm','original_language','countries_norm','companies_norm','popularity','vote_count']].copy()
    results['similarity'] = score[top_idx]

    order = ["title", "similarity", "release_year", "genres_norm", "keywords_norm", "original_language", "countries_norm", "vote_count", "popularity", "companies_norm"]
    results = results.reindex(columns=[c for c in order if c in results.columns]).reset_index(drop=True)
    pretty_results = as_pretty(results)

    base_row = movies.loc[base_index, ['title','release_year']]
    year = int(base_row['release_year']) if pd.notna(base_row['release_year']) else None
    header = f"Top {top_n} Recommendations for: {base_row['title']} ({year})" if year is not None else f"Top {top_n} Recommendations for: {base_row['title']}"

    return pretty_results, header, None

# -- VISUALIZATIONS --
# Return movie title with year
def _base_title(title: str):
    base_index, base_pos = get_base_index(movies, title)
    if base_index is None:
        return title, base_index, base_pos
    row = movies.loc[base_index]
    yr = int(row["release_year"]) if pd.notna(row["release_year"]) else None
    base_title = f"{row['title']} ({yr})" if yr is not None else f"{row['title']}"
    return base_title, base_index, base_pos


# Visualization 1: Top-N recommendation scores (horizontal bar chart)
def plot_topn_scores_fig(title: str, top_n: int = TOP_N, figsize=(8, 4), dpi=110):
    base_title, _, _ = _base_title(title)
    recs, _, msg = recommend(movies, matrices, title, top_n=top_n)
    if msg or recs.empty:
        return None

    titles = recs["Title"].tolist()
    scores = recs["Similarity Score"].to_numpy(float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    y = np.arange(len(titles))
    ax.barh(y, scores, height=0.9)
    ax.set_yticks(y, titles)
    ax.invert_yaxis()
    ax.set_xlabel("Overall Recommendation Score")
    ax.set_ylabel("Title")
    ax.set_title(f"Top {top_n} Recommendations — {base_title}")
    ax.set_xlim(0, 1.0)
    plt.close(fig)
    return fig


# Visualization 2: Similarity score distribution (histogram)
def plot_score_distribution_fig(title: str, top_n: int = 200, bins: int = 20, figsize=(8, 4), dpi=110):
    base_title, _, _ = _base_title(title)
    recs, _, msg = recommend(movies, matrices, title, top_n=top_n)
    if msg or recs.empty:
        return None

    scores = recs["Similarity Score"].to_numpy(float)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.hist(scores, bins=bins)
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Similarity Score Distribution — {base_title}")
    plt.close(fig)
    return fig


# Visualization 3: Score breakdown by signal (stacked horizontal bar chart)
def plot_score_breakdown_fig(title: str, top_n: int = TOP_N, figsize=(10, 5), dpi=110):
    weights = {"text":0.25, "keyword":0.25, "genre":0.20, "meta":0.12, "lang":0.08, "country":0.06, "company":0.04}
    signals = list(weights)

    base_title, _, base_pos = _base_title(title)
    if base_pos is None:
        return None

    recs, _, msg = recommend(movies, matrices, title, top_n=top_n)
    if msg or recs.empty:
        return None

    # Align Top-N rows back to the master DataFrame by (Title, Year)
    labels = (movies.reset_index()
              .merge(recs[["Title", "Year"]], left_on=["title", "release_year"], right_on=["Title", "Year"],
                     how="right")["index"].to_numpy())
    idx = movies.index.get_indexer(labels)

    sims = compute_similarities(int(base_pos), matrices)
    parts = np.vstack([weights[s] * sims[s][idx] for s in signals])

    # Popularity bump part (same normalization as recommend)
    pop = movies["popularity"].to_numpy(float)
    pop = (pop - np.nanmin(pop)) / (np.nanmax(pop) - np.nanmin(pop) + 1e-9)
    parts = np.vstack([parts, 0.05 * pop[idx]])
    legend_labels = signals + ["popularity"]

    y = np.arange(len(recs))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    left = np.zeros(len(recs))
    for row, lab in zip(parts, legend_labels):
        ax.barh(y, row, left=left, label=lab)
        left += row

    ax.set_yticks(y, recs["Title"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Score Contribution (Weighted)")
    ax.set_ylabel("Title")
    ax.set_title(f"Score Breakdown by Signal — {base_title}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    plt.close(fig)
    return fig


# Visualization 4: Token frequency bar chart (keywords or genres)
def plot_dist_fig(title: str, top_n: int = TOP_N, by: str = "keywords",
                  top_tokens: int = 15, figsize=(8, 4), dpi=110):
    base_title, _, _ = _base_title(title)
    recs, _, msg = recommend(movies, matrices, title, top_n=top_n)
    if msg or recs.empty:
        return None

    col = {"keywords":"Keywords", "genres":"Genres"}.get(str(by).strip().lower())
    if col is None or col not in recs.columns:
        return None

    # explode comma-separated tokens from the pretty table
    items = []
    for v in recs[col].fillna("").astype(str):
        items.extend([t.strip() for t in v.split(",") if t.strip()])

    counts = Counter(items)
    if not counts:
        return None

    labels, nums = zip(*counts.most_common(top_tokens))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.bar(range(len(nums)), nums)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"{col} Distribution in Top {top_n} — {base_title}")
    plt.close(fig)
    return fig

# --- UI --- 
# Header
st.title("Next2Watch 🎬 - Movie Recommender")

movies = load_df(DATA_PATH)
matrices = build_features(movies)

# Search box
movie_title = st.text_input("Type a movie title to get recommendations:", placeholder="e.g., The Matrix")

as_percent = st.checkbox("Show similarity as %", value=False)

# Top-N slider

# Call recommender
if movie_title.strip():
    recs, header, msg = recommend(movies, matrices, movie_title, top_n=TOP_N)
    if msg:
        st.warning(msg)
    elif recs.empty:
        st.warning(f"No recommendations to show for {movie_title}.")
    else:
        st.subheader(header)
        st.caption("Scored by: text (0.25), keywords (0.25), genres (0.20), meta (0.10), language (0.10), countries (0.06), companies (0.04) (+ small popularity bump)")

        label = "Similarity Score"
        format = None
        display = recs

        if as_percent:
            display = recs.copy()
            display[label] = (display[label] * 100).round(2)
            label = "Similarity Score (%)"
            format = "%.2f%%"

        st.dataframe(display, width='stretch', column_config={"Similarity Score": st.column_config.NumberColumn(label=label, format=format)})

        st.markdown("**Visualizations:** Top-N scores, distribution, signal contributions, and token mix for the Top-N set")

# Display visualizations (UI)
st.divider()

col1, col2 = st.columns(2, gap="small")
fig = plot_topn_scores_fig(movie_title, top_n=TOP_N)
if fig: col1.pyplot(fig)

fig = plot_score_distribution_fig(movie_title, top_n=200, bins=20)
if fig: col2.pyplot(fig)

col3, col4 = st.columns(2, gap="small")
fig = plot_score_breakdown_fig(movie_title, top_n=TOP_N)
if fig: col3.pyplot(fig)

fig = plot_dist_fig(movie_title, top_n=TOP_N, by="keywords", top_tokens=15)
if fig: col4.pyplot(fig)

# Footer / Notes