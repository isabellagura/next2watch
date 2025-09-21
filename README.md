# Next2Watch

Movie recommendation web application using content-based filtering, built with Python and Streamlit.

## Features

- TF-IDF + tokenized features (keywords, genres, metadata, language, countries, companies)
- Cosine similarity
- Title search box with dropdown for selecting correct film
- Top-N recommendations table
  - Similarity score (toggle % view)
  - Reasons for recommendation pick (overlapping keywords/genres, same decade, country/studio)
- Four Visualizations: Top-N scores, score distribution, signal breakdown, token distribution

## Instructions

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
streamlit run app.py
```
