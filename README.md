# CINEIQ

Explainable hybrid movie recommendation platform built on **MovieLens 25M**, **TMDB metadata**, and **IMDb 50K reviews**.

## Overview

CINEIQ is a demo-ready recommendation system designed to make movie discovery more transparent and personalized. Instead of relying on a single recommendation method, it combines multiple signals:

- **Content-based filtering** using TF-IDF + cosine similarity
- **Collaborative filtering** using item-item similarity
- **SVD-based matrix factorization** using `Surprise`
- **Sentiment-aware reranking** using IMDb 50K reviews + VADER
- **Rule-based explainability** for human-readable recommendation reasons

The project includes:

- A **FastAPI** backend
- A **Streamlit + Plotly** dashboard
- **MLflow** tracking for training metadata and artifacts

## Quick Start

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
python recommender.py
uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

In a second terminal:

```bash
source .venv312/bin/activate
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

If a trained artifact already exists in `models/`, you can skip `python recommender.py` and start the API directly.

## Key Features

- Hybrid recommendation engine with weighted ensemble scoring
- Personalized recommendations using historical user ratings
- Similar-title discovery
- User taste dashboard with:
  - genre radar
  - decade preferences
  - director affinities
  - actor affinities
  - favorite movies
- Explainability layer for every recommendation
- Experiment tracking through MLflow

## Datasets

This project uses the same dataset direction as the original concept:

- **MovieLens 25M** for user-movie ratings
- **TMDB Metadata** for genres, cast, directors, keywords, and overviews
- **IMDb 50K Reviews** for sentiment-aware ranking


## Architecture

### Recommendation pipeline

1. Load and join MovieLens, TMDB, and IMDb-aligned data
2. Build content features from movie metadata
3. Train collaborative filtering model
4. Train SVD latent-factor model
5. Compute sentiment-aware movie scores
6. Combine all signals into a weighted final rank
7. Return recommendations with explanation text

### Final scoring signals

The final recommendation score blends:

- content similarity
- collaborative filtering score
- SVD score
- popularity score
- sentiment score

## Project Structure

```text
CINEIQ/
├── backend/              # FastAPI app
├── cineiq/               # Core training and inference engine
├── dashboard/            # Streamlit UI
├── data/                 # Raw datasets + legacy sample files
├── mlruns/               # MLflow artifacts and runs
├── models/               # Saved trained model artifact
├── recommender.py        # Training entrypoint
├── src/                  # Legacy baseline scripts from the initial repo
├── requirements.txt      # Python dependencies
├── CINEIQ_Project_Report.docx  # Detailed project report
└── README.md
```

## Setup

From the repo root:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

## Train the Model

### Standard training

```bash
source .venv312/bin/activate
python recommender.py
```

By default, this uses the built-in demo-oriented training profile from `cineiq/engine.py`, which is tuned for faster local iteration.

### Training controls

The training pipeline supports environment-variable tuning:

```bash
export CINEIQ_MAX_TRAIN_USERS=2000
export CINEIQ_MIN_USER_RATINGS=150
export CINEIQ_MIN_MOVIE_RATINGS=100
export CINEIQ_CONTENT_FEATURES=12000
export CINEIQ_REVIEW_FEATURES=8000
export CINEIQ_SVD_FACTORS=40
export CINEIQ_SVD_EPOCHS=8
export CINEIQ_KNN_NEIGHBORS=60
python recommender.py
```

### Larger full-scale training

For a much larger run closer to the full usable MovieLens set:

```bash
export CINEIQ_MAX_TRAIN_USERS=999999
export CINEIQ_MIN_USER_RATINGS=1
export CINEIQ_MIN_MOVIE_RATINGS=100
python recommender.py
```

This produces a much larger artifact and slower startup time, but better matches the large-scale project setup.

## Run the API

```bash
source .venv312/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

Notes:

- Startup can take noticeably longer when loading a large full-scale artifact
- If the saved artifact is missing or outdated, the engine can retrain on startup
- A full-scale artifact may be several GB in size, so first load is expected to be slower

API base URL:

- `http://127.0.0.1:8001`

## Run the Dashboard

```bash
source .venv312/bin/activate
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

Dashboard URL:

- `http://127.0.0.1:8501`

If the API runs on a different host or port:

```bash
export CINEIQ_API_URL=http://127.0.0.1:8001
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

## API Endpoints

### `GET /health`

Returns service health and model metadata.

### `GET /movies`

Returns available movie titles.

Example:

```text
/movies?limit=200
```

### `GET /users`

Returns available demo user IDs.

### `GET /recommend`

Returns hybrid recommendations.

Example:

```text
/recommend?movie=Forrest%20Gump%20(1994)&user_id=187&top_n=10
```

Optional weights:

- `content_weight`
- `collaborative_weight`
- `svd_weight`
- `popularity_weight`
- `sentiment_weight`

### `GET /similar`

Returns similar titles for a seed movie.

### `GET /profile/{user_id}`

Returns taste-profile analytics for a user.

### `GET /explain`

Returns an explanation for a specific candidate recommendation.

## Current Deliverables Covered

- Hybrid Recommendation Engine
- Sentiment-Aware Re-Ranker
- User Taste Dashboard
- Explainability Layer
- FastAPI serving
- Streamlit + Plotly dashboard
- MLflow experiment tracking

## Notes

- The system currently personalizes using **historical rating data**
- It does **not** yet implement a live production feedback loop
- Explainability is implemented using **rule-based templates**
- Sentiment reranking is implemented using **VADER**, which is valid for the project scope
- The repo supports both a **demo-sized training mode** and a **full-scale training mode**

## Report

See the full write-up in:

- `CINEIQ_Project_Report.docx`




