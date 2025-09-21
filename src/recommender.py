import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Constants
WEIGHTS = {'text': 0.25,
           'keyword': 0.25,
           'genre': 0.20,
           'meta': 0.12,
           'lang': 0.08,
           'country': 0.06,
           'company': 0.04
}

# Column labels for display
DISPLAY_LABELS = {
    'id': 'ID',
    'title': 'Title',
    'release_year': 'Year',
    'genres_norm': 'Genres',
    'keywords_norm':'Keywords',
    'original_language': 'Language',
    'countries_norm': 'Countries',
    'companies_norm': 'Production Companies',
    'similarity': 'Similarity Score',
    'vote_count': 'Vote Count',
    'popularity': 'Popularity'
}

# Change column names to display labels
def as_pretty(df: pd.DataFrame, labels=None):
    labels = labels or DISPLAY_LABELS
    return df.rename(columns=labels)

# Replace underscore with space in multi-word tokens (genres, keywords, etc.)
def prettify_tokens_col(s: pd.Series) -> pd.Series:
    return (s.fillna('').astype(str).str.replace('_', ' ', regex=False))

# Popularity scale
def scaled_popularity(arr: np.ndarray) -> np.ndarray:
    # Min-max scale to [0, 1] with NaN safety
    pop_min, pop_max = np.nanmin(arr), np.nanmax(arr)
    return (arr - pop_min) / (pop_max - pop_min + 1e-9)

# Readability for recommendation rationale tokens
def _token_set(text: str) -> set[str]:
    # Comma-separated -> set of normalized tokens (underscores→spaces, lowercase).
    if not isinstance(text, str):
        return set()
    return {t.strip().lower().replace('_', ' ') for t in text.split(',') if t.strip()}

# Create a readable list of reasons for each recommendation
def build_reasons_string(base_row: dict, rec_pretty_row: pd.Series) -> str:
    # Return a short, readable rationale for a single recommendation.
    parts: list[str] = []

    # Keywords: show up to 3 shared examples
    bk = _token_set(base_row.get('keywords_norm', ''))
    rk = _token_set(rec_pretty_row.get('Keywords', ''))
    kw = sorted(bk & rk)
    if kw:
        shown = ', '.join(w.title() for w in kw[:3])
        if len(kw) > 3:
            shown += '…'
        parts.append(f"Keywords: {shown}")

    # Genres: only call out overlap
    bg = _token_set(base_row.get('genres_norm', ''))
    rg = _token_set(rec_pretty_row.get('Genres', ''))
    if bg & rg:
        parts.append('Genre overlap')

    # Same decade
    try:
        by = base_row.get('release_year')
        ry = rec_pretty_row.get('Year')
        by = int(by) if by is not None else None
        ry = int(ry) if pd.notna(ry) else None
        if by is not None and ry is not None and (by // 10) == (ry // 10):
            parts.append(f"Same decade: {ry // 10}0s")
    except Exception:
        pass

    # Country matches
    bc = _token_set(base_row.get('countries_norm', ''))
    rc = _token_set(rec_pretty_row.get('Countries', ''))
    if bc & rc:
        parts.append('Country match')

    # Company matches
    bp = _token_set(base_row.get('companies_norm', ''))
    rp = _token_set(rec_pretty_row.get('Production Companies', ''))
    if bp & rp:
        parts.append('Studio match')

    # Make it compact
    return ' · '.join(parts)

# Select base movie by normalized title (exact match first, then substring)
def get_base_index(movies: pd.DataFrame, title: str):
    raw = str(title).strip()
    m = re.match(r'^(.*?)[\s]*\((\d{4})\)$', raw)
    if m:
        query_title = m.group(1).strip().casefold()
        query_year = int(m.group(2))
    else:
        query_title = raw.casefold()
        query_year = None

    titles_norm = movies['title_norm'].astype('string').str.casefold().str.strip()

    # exact title; prefer exact year if given
    matches = movies.index[titles_norm.eq(query_title)]
    if query_year is not None and len(matches) > 0:
        year_mask = (movies.loc[matches, 'release_year'].astype('Int64') == query_year)
        matches = matches[year_mask.values]

    # fallback: substring (optionally year-filtered)
    if len(matches) == 0:
        sub = titles_norm.str.contains(query_title, regex=False)
        if query_year is not None:
            sub &= (movies['release_year'].astype('Int64') == query_year)
        matches = movies.index[sub]
        if len(matches) == 0:
            return None, None
        
    # Break ties by vote_count then popularity.
    base_index = (movies.loc[matches, ['vote_count','popularity']]
                  .fillna(0)
                  .sort_values(['vote_count','popularity'], ascending=False)
                  .index[0])
    
    base_pos = movies.index.get_loc(base_index)
    return base_index, base_pos

# Compute per-field cosine similarities from the base position
def compute_similarities(base_pos: int, matrices: dict) -> dict:
    return {
        feature: cosine_similarity(matrices[feature][base_pos], matrices[feature]).ravel()
        for feature in WEIGHTS.keys()
    }

# Movie recommender (cosine + small popularity bump)
def recommend(movies: pd.DataFrame, matrices: dict, title: str, top_n: int = 10):

    display_cols = ['title','similarity','release_year','genres_norm','keywords_norm','original_language','countries_norm','companies_norm','popularity','vote_count']

    base_index, base_pos = get_base_index(movies, title)

    # handle case when there is no match for movie
    if base_index is None or base_pos is None:
        empty = pd.DataFrame(columns=display_cols)
        return empty, None, f"No recommendations to show for '{title}'."

    sims = compute_similarities(base_pos, matrices)

    # weighted blend
    score = sum(WEIGHTS[k] * sims[k] for k in WEIGHTS)

    # small popularity bump
    pop = movies['popularity'].to_numpy(float)
    pop_scaled = scaled_popularity(pop)
    score = score + 0.05 * pop_scaled

    # exclude the same movie in rec results
    score[base_pos] = -1.0

    # top-N by score
    top_idx = np.argpartition(score, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(score[top_idx])[::-1]]

    results = movies.iloc[top_idx][
        ['title','release_year','genres_norm','keywords_norm','original_language','countries_norm','companies_norm','popularity','vote_count']].copy()
    results['similarity'] = score[top_idx]

    order = ['title', 'similarity', 'release_year', 'genres_norm', 'keywords_norm', 'original_language', 'countries_norm', 'vote_count', 'popularity', 'companies_norm']
    results = results.reindex(columns=[c for c in order if c in results.columns]).reset_index(drop=True)
    pretty_results = as_pretty(results)

    base_row = movies.loc[base_index, ['title','release_year']]
    year = int(base_row['release_year']) if pd.notna(base_row['release_year']) else None
    header = f"Top {top_n} Recommendations for: {base_row['title']} ({year})" if year is not None else f"Top {top_n} Recommendations for: {base_row['title']}"

    return pretty_results, header, None