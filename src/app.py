# Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from collections import Counter

# App config
st.set_page_config(page_title='Next2Watch', page_icon='🎬', layout='wide')

from recommender import (
    WEIGHTS,
    DISPLAY_LABELS,
    as_pretty,
    prettify_tokens_col,
    scaled_popularity,
    build_reasons_string,
    get_base_index,
    compute_similarities,
    recommend,
)

# Constants
TOP_N = 10
DATA_PATH = 'data/processed/movies_clean.parquet'

# Cache loaders
# Load data (cached)
@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

# Feature builders (cached)
@st.cache_resource(show_spinner=False)
def build_features(movies: pd.DataFrame):
    # Combine text data (title + overview)
    text_data = (
        movies['title_norm'].astype('string').str.lower() + ' ' +
        movies['overview_norm'].astype('string').str.lower()
        )

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
        'text': text_matrix,
        'keyword': keyword_matrix,
        'meta': meta_matrix,
        'genre': genre_matrix,
        'country': country_matrix,
        'company': company_matrix,
        'lang': lang_matrix,
    }
    return matrices

# Search function for dropdown results to display
def search_title(movies: pd.DataFrame, query: str, top_k: int = 20) -> pd.DataFrame:
    q = str(query).strip().casefold()
    if not q:
        return movies.iloc[0:0][['title', 'release_year']]

    titles_norm = movies['title_norm'].astype('string').str.casefold().str.strip()

    # exact match first, else substring
    idx = titles_norm.eq(q)
    if not idx.any():
        idx = titles_norm.str.contains(q, regex=False)

    if not idx.any():
        return movies.iloc[0:0][['title', 'release_year']]

    # sort by vote_count/popularity (ties -> most “known” first), keep original index
    cols = ['title', 'release_year', 'vote_count', 'popularity']
    out = movies.loc[idx, cols].copy()
    out = out.sort_values(['vote_count', 'popularity'], ascending=False).head(top_k)
    return out[['title', 'release_year']]

# -- Visualizations --
# Return chosen movie title with year
def _base_title(title: str):
    base_index, base_pos = get_base_index(movies, title)
    if base_index is None:
        return title, base_index, base_pos
    row = movies.loc[base_index]
    year = int(row['release_year']) if pd.notna(row['release_year']) else None
    base_title = f"{row['title']} ({year})" if year is not None else f"{row['title']}"
    return base_title, base_index, base_pos


# Visualization 1: Top-N recommendation scores (horizontal bar chart)
def plot_topn_scores_fig(title: str, top_n: int = TOP_N, figsize=(8, 4), dpi=110):
    base_title, _, _ = _base_title(title)
    
    recs, _, msg = recommend(movies, matrices, title, top_n=top_n)
    if msg or recs.empty:
        return None

    titles = recs['Title'].tolist()
    scores = recs['Similarity Score'].to_numpy(float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    y = np.arange(len(titles))
    ax.barh(y, scores, height=0.9)
    ax.set_yticks(y, titles)
    ax.invert_yaxis()
    ax.set_xlabel('Overall Recommendation Score')
    ax.set_ylabel('Title')
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

    scores = recs['Similarity Score'].to_numpy(float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.hist(scores, bins=bins)
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Count')
    ax.set_title(f"Similarity Score Distribution (Top {top_n}) — {base_title}")
    
    plt.close(fig)
    return fig


# Visualization 3: Score breakdown by signal (stacked horizontal bar chart)
def plot_score_breakdown_fig(title: str, top_n: int = TOP_N, figsize=(8, 4), dpi=110):
    signals = list(WEIGHTS)

    base_title, _, base_pos = _base_title(title)
    if base_pos is None:
        return None

    recs, _, msg = recommend(movies, matrices, title, top_n=top_n)
    if msg or recs.empty:
        return None

    # Align Top-N rows back to the master DataFrame by (Title, Year)
    recs_key = recs[['Title', 'Year']].reset_index(drop=True)
    labels = (movies.reset_index()
              .merge(recs_key[['Title', 'Year']], left_on=['title', 'release_year'], right_on=['Title', 'Year'],
                     how='right')['index'].to_numpy())
    idx = movies.index.get_indexer(labels)
    mask = idx != -1
    idx = idx[mask]
    if idx.size == 0:
        return None
    
    recs_aligned = recs.loc[mask].reset_index(drop=True)

    sims = compute_similarities(int(base_pos), matrices)
    parts = np.vstack([WEIGHTS[s] * sims[s][idx] for s in signals])

    # Popularity bump part (same as recommend)
    pop = movies['popularity'].to_numpy(float)
    pop_scaled = scaled_popularity(pop)
    parts = np.vstack([parts, 0.05 * pop_scaled[idx]])
    legend_labels = signals + ['popularity']

    y = np.arange(len(recs_aligned))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    left = np.zeros(len(recs_aligned))
    for row, lab in zip(parts, legend_labels):
        ax.barh(y, row, left=left, label=lab)
        left += row

    ax.set_yticks(y, recs_aligned['Title'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Score Contribution (Weighted)')
    ax.set_ylabel('Title')
    ax.set_title(f"Score Breakdown by Signal — {base_title}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

    plt.close(fig)
    return fig

# Visualization 4: Token frequency bar chart (keywords or genres)
def plot_dist_fig(title: str, top_n: int = 50, by: str = 'keywords',
                  top_tokens: int = 15, figsize=(8, 4), dpi=110):
    base_title, _, _ = _base_title(title)

    recs, _, msg = recommend(movies, matrices, title, top_n=top_n)
    if msg or recs.empty:
        return None

    col = {'keywords':'Keywords', 'genres':'Genres'}.get(str(by).strip().lower())
    if col is None or col not in recs.columns:
        return None

    # split comma-separated tokens
    items = []
    for v in recs[col].fillna('').astype(str).str.replace('_', ' ', regex=False):
        items.extend([t.strip() for t in v.split(',') if t.strip()])

    counts = Counter(items)
    if not counts:
        return None

    labels, nums = zip(*counts.most_common(top_tokens))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    ax.bar(range(len(nums)), nums)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.set_title(f"{col} Distribution in Top {top_n} — {base_title}")

    plt.close(fig)
    return fig

# --- UI --- 
# Header
st.title('Next2Watch 🎬 - Movie Recommender')

# Load data for UI
movies = load_df(DATA_PATH)
matrices = build_features(movies)

# Search box
movie_title = st.text_input("Type a movie title to get recommendations:", placeholder="e.g., The Matrix")

# Select exact movie if there are same title matches
selected_index = None
q = movie_title.strip()
if len(q) >= 2:
    suggestions = search_title(movies, q, top_k=20)
    if not suggestions.empty:
        # Build options as (row_index, pretty_label) tuples; keep row_index to get the exact movie
        options = [(int(idx),
                    f"{row['title']} ({int(row['release_year'])})"
                    if pd.notna(row['release_year']) else
                    f"{row['title']}")
                   for idx, row in suggestions.iterrows()]

        selected = st.selectbox(
            "Pick the exact movie (optional):",
            options,
            index=0, # default to first ranked suggestion
            format_func=lambda x: x[1], # show the pretty label
        )
        selected_index = selected[0]

if selected_index is not None:
    sel_title = movies.loc[selected_index, 'title']
    sel_year = movies.loc[selected_index, 'release_year']
    query_title = f"{sel_title} ({int(sel_year)})" if pd.notna(sel_year) else str(sel_title)
else:
    query_title = movie_title


# Top-N slider - Let user decide how many recommendations to show
TABLE_N = st.slider("How many recommendations?", min_value=1, max_value=100, value=10)

# Toggle to show similarity score as percentage
as_percent = st.toggle("Show similarity as %", value=False)

# Display recommendations table
if movie_title.strip():

    recs, header, msg = recommend(movies, matrices, query_title, top_n=TABLE_N)

    if msg:
        st.warning(msg)
    elif recs.empty:
        st.warning(f"No recommendations to show for {query_title}.")
    else:
        st.subheader(header)
        st.caption("Scored by: text (0.25), keywords (0.25), genres (0.20), meta (0.10), language (0.10), countries (0.06), companies (0.04) (+ small popularity bump)")

        score_col_key = 'Similarity Score'
        score_label = 'Similarity Score'
        score_format = None

        display = recs.copy()
        display.index = pd.RangeIndex(start=1, stop=len(display)+1, name='Rank')

        if as_percent:
            display[score_col_key] = (display[score_col_key] * 100).round(2)
            score_label = 'Similarity Score (%)'
            score_format = '%.2f%%'

        for col in ('Genres', 'Keywords', 'Countries', 'Production Companies'):
            if col in display.columns:
                display[col] = prettify_tokens_col(display[col])

        # Build Reasons column using the raw base row
        base_index, _ = get_base_index(movies, query_title)
        base_row_dict = movies.loc[base_index].to_dict() if base_index is not None else {}

        display['Reasons'] = display.apply(lambda r: build_reasons_string(base_row_dict, r), axis=1)

        preferred_order = ['Title', score_col_key, 'Reasons', 'Year', 'Language', 'Countries', 'Production Companies', 'Genres', 'Keywords', 'Popularity', 'Vote Count']

        display = display.reindex(columns=[c for c in preferred_order if c in display.columns])


        st.dataframe(display, width='stretch', column_config={score_col_key: st.column_config.NumberColumn(label=score_label, format=score_format)})
        st.divider()
        st.markdown("**Visualizations:** Top-N scores, distribution, signal contributions, and token mix for the Top-N set")
        st.caption("Note: Charts below use static Top-N for consistency.")

    # Display visualizations
    col1, col2 = st.columns(2, gap='small')

    fig = plot_topn_scores_fig(query_title, top_n=TOP_N)
    if fig: col1.pyplot(fig)

    fig = plot_score_distribution_fig(query_title, top_n=200, bins=20)
    if fig: col2.pyplot(fig)

    col3, col4 = st.columns(2, gap='small')
    
    fig = plot_score_breakdown_fig(query_title, top_n=TOP_N)
    if fig: col3.pyplot(fig)

    fig = plot_dist_fig(query_title, top_n=50, by='keywords', top_tokens=15)
    if fig: col4.pyplot(fig)