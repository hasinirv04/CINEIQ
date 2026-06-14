from __future__ import annotations

import ast
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from nltk import download as nltk_download
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset as SurpriseDataset
from surprise import KNNBasic, Reader, SVD as SurpriseSVD

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACT_PATH = MODELS_DIR / "cineiq_artifacts.joblib"
MLRUNS_DIR = ROOT_DIR / "mlruns"
ARTIFACT_VERSION = "slide-aligned-v1"

ML25M_DIR = RAW_DATA_DIR / "ml-25m"
TMDB_LINK_DIR = RAW_DATA_DIR / "the-movies-dataset"
IMDB_REVIEW_DIR = RAW_DATA_DIR / "imdb-50k-reviews"

MAX_TRAIN_USERS = int(os.getenv("CINEIQ_MAX_TRAIN_USERS", "2000"))
MIN_USER_RATINGS = int(os.getenv("CINEIQ_MIN_USER_RATINGS", "150"))
MIN_MOVIE_RATINGS = int(os.getenv("CINEIQ_MIN_MOVIE_RATINGS", "100"))
CONTENT_FEATURES = int(os.getenv("CINEIQ_CONTENT_FEATURES", "12000"))
REVIEW_FEATURES = int(os.getenv("CINEIQ_REVIEW_FEATURES", "8000"))
SVD_FACTORS = int(os.getenv("CINEIQ_SVD_FACTORS", "40"))
SVD_EPOCHS = int(os.getenv("CINEIQ_SVD_EPOCHS", "8"))
KNN_NEIGHBORS = int(os.getenv("CINEIQ_KNN_NEIGHBORS", "60"))


def _normalize_title(title: str) -> str:
    title = re.sub(r"\s*\(\d{4}\)$", "", str(title))
    title = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return " ".join(title.split())


def _safe_parse_names(raw_text: Any, limit: int | None = None, field: str = "name") -> list[str]:
    if pd.isna(raw_text):
        return []
    try:
        items = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError):
        return []
    values: list[str] = []
    for item in items:
        value = item.get(field)
        if value:
            values.append(str(value))
        if limit and len(values) >= limit:
            break
    return values


def _extract_director(raw_text: Any) -> str:
    if pd.isna(raw_text):
        return ""
    try:
        items = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError):
        return ""
    for item in items:
        if item.get("job") == "Director":
            return str(item.get("name", ""))
    return ""


def _tokenize_terms(values: list[str]) -> list[str]:
    return [value.replace(" ", "") for value in values if value]


def _normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    minimum = np.nanmin(values)
    maximum = np.nanmax(values)
    if math.isclose(maximum, minimum):
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _ensure_vader() -> SentimentIntensityAnalyzer:
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk_download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def _ensure_ml25m() -> Path:
    if ML25M_DIR.exists():
        return ML25M_DIR
    archive_path = ROOT_DIR / "ml-25m.zip"
    if archive_path.exists():
        import zipfile

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(RAW_DATA_DIR)
        return ML25M_DIR
    raise FileNotFoundError("MovieLens 25M is missing. Add `ml-25m.zip` to the project root.")


def _resolve_kaggle_dataset(local_dir: Path, dataset_name: str) -> Path:
    if local_dir.exists():
        return local_dir
    try:
        import kagglehub
    except ImportError as error:
        raise FileNotFoundError(f"{dataset_name} not found and kagglehub is unavailable.") from error
    downloaded = Path(kagglehub.dataset_download(dataset_name))
    return downloaded


def _tmdb_dir() -> Path:
    return _resolve_kaggle_dataset(TMDB_LINK_DIR, "rounakbanik/the-movies-dataset")


def _imdb_dir() -> Path:
    return _resolve_kaggle_dataset(IMDB_REVIEW_DIR, "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")


def _load_sentiment_review_frame() -> tuple[pd.DataFrame, str]:
    path = _imdb_dir() / "IMDB Dataset.csv"
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
    return frame, "IMDb 50K Reviews + VADER"


def _build_sentiment_scores(movie_frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    review_frame, source = _load_sentiment_review_frame()
    analyzer = _ensure_vader()

    sampled_reviews = review_frame.sample(
        n=min(len(review_frame), 30000),
        random_state=42,
        replace=False,
    ).copy()
    sampled_reviews["review"] = sampled_reviews["review"].astype(str)
    positive_mask = sampled_reviews["sentiment"].astype(str).str.lower().eq("positive").to_numpy()

    review_vectorizer = TfidfVectorizer(
        max_features=REVIEW_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
    )
    review_matrix = review_vectorizer.fit_transform(sampled_reviews["review"])
    positive_centroid = np.asarray(review_matrix[positive_mask].mean(axis=0))
    negative_centroid = np.asarray(review_matrix[~positive_mask].mean(axis=0))

    movie_overviews = movie_frame["overview"].fillna("").astype(str)
    movie_review_matrix = review_vectorizer.transform(movie_overviews)
    positive_alignment = cosine_similarity(movie_review_matrix, positive_centroid).ravel()
    negative_alignment = cosine_similarity(movie_review_matrix, negative_centroid).ravel()
    review_alignment = _normalize_array(positive_alignment - negative_alignment)

    vader_overview = movie_overviews.map(lambda text: analyzer.polarity_scores(text)["compound"]).to_numpy(dtype=float)
    vader_overview = (vader_overview + 1.0) / 2.0

    audience_signal = _normalize_array(
        movie_frame["vote_average"].fillna(movie_frame["avg_rating"]).to_numpy(dtype=float)
    )

    sentiment_scores = np.clip(
        (review_alignment * 0.55) + (vader_overview * 0.25) + (audience_signal * 0.20),
        0,
        1,
    )
    return sentiment_scores, source


def _weighted_rating(row: pd.Series, global_mean: float, vote_threshold: float) -> float:
    votes = row["rating_count"]
    rating = row["avg_rating"]
    if votes <= 0:
        return global_mean
    return (votes / (votes + vote_threshold) * rating) + (
        vote_threshold / (votes + vote_threshold) * global_mean
    )


def _load_tmdb_metadata() -> pd.DataFrame:
    tmdb_path = _tmdb_dir()

    movies = pd.read_csv(tmdb_path / "movies_metadata.csv", low_memory=False)
    credits = pd.read_csv(tmdb_path / "credits.csv")
    keywords = pd.read_csv(tmdb_path / "keywords.csv")

    movies["tmdb_id"] = pd.to_numeric(movies["id"], errors="coerce")
    credits["tmdb_id"] = pd.to_numeric(credits["id"], errors="coerce")
    keywords["tmdb_id"] = pd.to_numeric(keywords["id"], errors="coerce")

    movies["genres_list"] = movies["genres"].map(_safe_parse_names)
    movies["release_year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    movies["normalized_title"] = movies["title"].map(_normalize_title)

    keywords["keywords_list"] = keywords["keywords"].map(_safe_parse_names)
    credits["cast_list"] = credits["cast"].map(lambda raw: _safe_parse_names(raw, limit=6))
    credits["director"] = credits["crew"].map(_extract_director)

    metadata = (
        movies[
            [
                "tmdb_id",
                "title",
                "original_title",
                "overview",
                "popularity",
                "vote_average",
                "vote_count",
                "genres_list",
                "release_year",
                "normalized_title",
            ]
        ]
        .rename(columns={"title": "tmdb_title", "original_title": "tmdb_original_title"})
        .merge(keywords[["tmdb_id", "keywords_list"]], on="tmdb_id", how="left")
        .merge(credits[["tmdb_id", "cast_list", "director"]], on="tmdb_id", how="left")
    )

    metadata["keywords_list"] = metadata["keywords_list"].apply(lambda value: value if isinstance(value, list) else [])
    metadata["cast_list"] = metadata["cast_list"].apply(lambda value: value if isinstance(value, list) else [])
    metadata["director"] = metadata["director"].fillna("")
    metadata["overview"] = metadata["overview"].fillna("")
    return metadata


def _prepare_movies() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ml25m_dir = _ensure_ml25m()
    movielens_movies = pd.read_csv(
        ml25m_dir / "movies.csv",
        dtype={"movieId": "int32", "title": "string", "genres": "string"},
    )
    movielens_links = pd.read_csv(
        ml25m_dir / "links.csv",
        dtype={"movieId": "int32", "imdbId": "float64", "tmdbId": "float64"},
    )
    ratings = pd.read_csv(
        ml25m_dir / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )

    movielens_movies["normalized_title"] = movielens_movies["title"].map(_normalize_title)
    movielens_movies["release_year_movielens"] = (
        movielens_movies["title"].str.extract(r"\((\d{4})\)").fillna("0").astype(int)
    )
    movielens_movies["movielens_genres"] = movielens_movies["genres"].fillna("").str.split("|")

    tmdb_metadata = _load_tmdb_metadata()

    merged = (
        movielens_movies.merge(movielens_links, on="movieId", how="left")
        .merge(tmdb_metadata, left_on="tmdbId", right_on="tmdb_id", how="inner")
        .drop_duplicates(subset=["movieId"])
    )

    ratings_summary = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    )
    merged = merged.merge(ratings_summary, on="movieId", how="left")
    merged = merged[merged["rating_count"].fillna(0) >= MIN_MOVIE_RATINGS].copy()

    eligible_movie_ids = set(merged["movieId"].tolist())
    ratings = ratings[ratings["movieId"].isin(eligible_movie_ids)].copy()

    user_activity = ratings.groupby("userId").size().sort_values(ascending=False)
    selected_users = user_activity[user_activity >= MIN_USER_RATINGS].head(MAX_TRAIN_USERS).index.tolist()
    ratings = ratings[ratings["userId"].isin(selected_users)].copy()

    slice_movie_counts = ratings.groupby("movieId").size()
    slice_movie_ids = slice_movie_counts[slice_movie_counts >= MIN_MOVIE_RATINGS].index.tolist()
    ratings = ratings[ratings["movieId"].isin(slice_movie_ids)].copy()
    merged = merged[merged["movieId"].isin(slice_movie_ids)].copy()

    merged["release_year"] = merged["release_year"].replace(0, np.nan).fillna(merged["release_year_movielens"])
    merged["release_year"] = merged["release_year"].fillna(0).astype(int)
    merged["genres_list"] = merged.apply(
        lambda row: sorted(set((row["genres_list"] or []) + (row["movielens_genres"] or []))),
        axis=1,
    )
    merged["keywords_list"] = merged["keywords_list"].apply(lambda value: value if isinstance(value, list) else [])
    merged["cast_list"] = merged["cast_list"].apply(lambda value: value if isinstance(value, list) else [])
    merged["director"] = merged["director"].fillna("")
    merged["overview"] = merged["overview"].fillna("")

    merged["decade"] = (merged["release_year"] // 10 * 10).fillna(0).astype(int)
    merged["tags"] = merged.apply(
        lambda row: " ".join(
            row["overview"].split()
            + _tokenize_terms(row["genres_list"])
            + _tokenize_terms(row["keywords_list"])
            + _tokenize_terms(row["cast_list"])
            + _tokenize_terms([row["director"]])
        ).lower(),
        axis=1,
    )

    global_mean = float(merged.loc[merged["avg_rating"] > 0, "avg_rating"].mean())
    vote_threshold = float(merged["rating_count"].quantile(0.70))
    merged["weighted_rating"] = merged.apply(
        lambda row: _weighted_rating(row, global_mean, vote_threshold),
        axis=1,
    )
    merged["popularity_score"] = _normalize_array(
        (merged["weighted_rating"] * np.log1p(merged["rating_count"] + 1)).to_numpy(dtype=float)
    )
    merged["sentiment_score"], sentiment_source = _build_sentiment_scores(merged)
    merged["sentiment_source"] = sentiment_source
    merged = merged.sort_values(["rating_count", "avg_rating"], ascending=False).reset_index(drop=True)

    dataset_summary = {
        "movielens": "MovieLens 25M",
        "tmdb": "TMDB Metadata (45K movies)",
        "imdb": "IMDb 50K Reviews",
        "selected_users": int(ratings["userId"].nunique()),
        "eligible_movies": int(len(merged)),
        "ratings_used": int(len(ratings)),
    }
    return merged, ratings.reset_index(drop=True), dataset_summary


def train_and_save() -> dict[str, Any]:
    MODELS_DIR.mkdir(exist_ok=True)
    MLRUNS_DIR.mkdir(exist_ok=True)

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("cineiq")

    movies, ratings, dataset_summary = _prepare_movies()

    content_vectorizer = TfidfVectorizer(max_features=CONTENT_FEATURES, stop_words="english")
    content_matrix = content_vectorizer.fit_transform(movies["tags"])
    content_similarity = cosine_similarity(content_matrix).astype(np.float32)

    ratings_for_model = ratings[["userId", "movieId", "rating"]].copy()
    ratings_for_model["userId"] = ratings_for_model["userId"].astype(str)
    ratings_for_model["movieId"] = ratings_for_model["movieId"].astype(str)

    reader = Reader(rating_scale=(0.5, 5.0))
    trainset = SurpriseDataset.load_from_df(ratings_for_model[["userId", "movieId", "rating"]], reader).build_full_trainset()

    collaborative_model = KNNBasic(
        k=KNN_NEIGHBORS,
        min_k=1,
        sim_options={"name": "cosine", "user_based": False},
        verbose=False,
    )
    collaborative_model.fit(trainset)

    svd_model = SurpriseSVD(
        n_factors=SVD_FACTORS,
        n_epochs=SVD_EPOCHS,
        random_state=42,
        biased=True,
        verbose=False,
    )
    svd_model.fit(trainset)

    movie_ids = movies["movieId"].astype(int).tolist()
    movie_inner_ids = [trainset.to_inner_iid(str(movie_id)) for movie_id in movie_ids]
    collaborative_similarity = collaborative_model.sim[np.ix_(movie_inner_ids, movie_inner_ids)].astype(np.float32)
    item_factors = svd_model.qi[movie_inner_ids].astype(np.float32)
    latent_similarity = cosine_similarity(item_factors).astype(np.float32)

    movie_index_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    ratings_with_titles = ratings.merge(
        movies[
            [
                "movieId",
                "title",
                "genres_list",
                "cast_list",
                "director",
                "decade",
                "release_year",
            ]
        ],
        on="movieId",
        how="left",
    )

    user_histories: dict[int, list[dict[str, Any]]] = {}
    for user_id, frame in ratings_with_titles.groupby("userId"):
        enriched = frame.sort_values("rating", ascending=False)[["movieId", "title", "rating"]]
        user_histories[int(user_id)] = enriched.to_dict("records")

    default_weights = {
        "content": 0.30,
        "collaborative": 0.25,
        "svd": 0.20,
        "popularity": 0.10,
        "sentiment": 0.15,
    }

    artifacts = {
        "artifact_version": ARTIFACT_VERSION,
        "movies": movies,
        "ratings": ratings,
        "ratings_with_titles": ratings_with_titles,
        "content_similarity": content_similarity,
        "collaborative_similarity": collaborative_similarity,
        "latent_similarity": latent_similarity,
        "item_factors": item_factors,
        "movie_index_map": movie_index_map,
        "movie_ids": movie_ids,
        "user_ids": sorted(user_histories.keys()),
        "user_histories": user_histories,
        "vector_features": int(content_matrix.shape[1]),
        "sentiment_source": str(movies["sentiment_source"].iloc[0]),
        "dataset_summary": dataset_summary,
        "default_weights": default_weights,
        "collaborative_model": collaborative_model,
        "svd_model": svd_model,
        "tech_stack": {
            "ml": ["Python", "scikit-learn", "Surprise (SVD)", "Pandas", "NumPy"],
            "nlp": ["VADER"],
            "serving": ["FastAPI"],
            "dashboard": ["Streamlit", "Plotly"],
            "tracking": ["MLflow"],
        },
    }

    joblib.dump(artifacts, ARTIFACT_PATH)

    with mlflow.start_run(run_name="train-slide-aligned-engine"):
        mlflow.log_param("artifact_version", ARTIFACT_VERSION)
        mlflow.log_param("movielens_dataset", dataset_summary["movielens"])
        mlflow.log_param("tmdb_dataset", dataset_summary["tmdb"])
        mlflow.log_param("imdb_dataset", dataset_summary["imdb"])
        mlflow.log_param("content_features", artifacts["vector_features"])
        mlflow.log_param("svd_backend", "Surprise SVD")
        mlflow.log_param("svd_factors", SVD_FACTORS)
        mlflow.log_param("svd_epochs", SVD_EPOCHS)
        mlflow.log_param("collaborative_backend", "Surprise KNNBasic (item-item cosine)")
        mlflow.log_param("sentiment_source", artifacts["sentiment_source"])
        mlflow.log_param("train_users", dataset_summary["selected_users"])
        mlflow.log_param("min_user_ratings", MIN_USER_RATINGS)
        mlflow.log_param("min_movie_ratings", MIN_MOVIE_RATINGS)
        for key, value in default_weights.items():
            mlflow.log_param(f"weight_{key}", value)
        mlflow.log_metric("movies_ready", float(len(movies)))
        mlflow.log_metric("users_ready", float(len(artifacts["user_ids"])))
        mlflow.log_metric("ratings_used", float(len(ratings)))
        mlflow.log_metric("avg_rating", float(movies["avg_rating"].mean()))
        mlflow.log_metric("max_rating_count", float(movies["rating_count"].max()))
        mlflow.log_artifact(str(ARTIFACT_PATH))

    return artifacts


class HybridRecommender:
    def __init__(self, artifact_path: Path | str = ARTIFACT_PATH):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            train_and_save()

        artifacts = joblib.load(self.artifact_path)
        if artifacts.get("artifact_version") != ARTIFACT_VERSION:
            artifacts = train_and_save()

        self.movies: pd.DataFrame = artifacts["movies"].copy()
        self.ratings: pd.DataFrame = artifacts["ratings"].copy()
        self.ratings_with_titles: pd.DataFrame = artifacts["ratings_with_titles"].copy()
        self.content_similarity: np.ndarray = artifacts["content_similarity"]
        self.collaborative_similarity: np.ndarray = artifacts["collaborative_similarity"]
        self.latent_similarity: np.ndarray = artifacts["latent_similarity"]
        self.item_factors: np.ndarray = artifacts["item_factors"]
        self.movie_index_map: dict[int, int] = artifacts["movie_index_map"]
        self.user_ids: list[int] = artifacts["user_ids"]
        self.user_histories: dict[int, list[dict[str, Any]]] = artifacts["user_histories"]
        self.default_weights: dict[str, float] = artifacts["default_weights"]
        self.vector_features: int = int(artifacts["vector_features"])
        self.sentiment_source: str = str(artifacts["sentiment_source"])
        self.dataset_summary: dict[str, Any] = artifacts["dataset_summary"]
        self.tech_stack: dict[str, list[str]] = artifacts["tech_stack"]
        self.collaborative_model: KNNBasic = artifacts["collaborative_model"]
        self.svd_model: SurpriseSVD = artifacts["svd_model"]
        self.title_lookup = {
            _normalize_title(title): idx for idx, title in enumerate(self.movies["title"].tolist())
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "movies": int(len(self.movies)),
            "users": int(len(self.user_ids)),
            "ratings": int(len(self.ratings)),
            "vector_features": self.vector_features,
            "sentiment_source": self.sentiment_source,
            "tracking_uri": f"file:{MLRUNS_DIR}",
            "datasets": self.dataset_summary,
            "tech_stack": self.tech_stack,
            "deliverables": [
                "Hybrid recommendation engine",
                "Sentiment-aware reranker",
                "User taste dashboard",
                "Explainability layer",
            ],
            "model_backends": {
                "content": "TF-IDF + cosine similarity",
                "collaborative": "Surprise KNNBasic (item-item cosine)",
                "svd": "Surprise SVD matrix factorization",
                "sentiment": self.sentiment_source,
            },
        }

    def list_movies(self, limit: int = 200) -> list[str]:
        return (
            self.movies.sort_values(["rating_count", "avg_rating"], ascending=False)["title"]
            .head(limit)
            .tolist()
        )

    def list_users(self) -> list[int]:
        return self.user_ids

    def resolve_title(self, movie_title: str) -> tuple[int, str]:
        normalized = _normalize_title(movie_title)
        if normalized in self.title_lookup:
            idx = self.title_lookup[normalized]
            return idx, str(self.movies.iloc[idx]["title"])
        exact = self.movies[self.movies["title"].str.lower() == movie_title.lower()]
        if not exact.empty:
            idx = int(exact.index[0])
            return idx, str(exact.iloc[0]["title"])
        matches = self.movies[self.movies["title"].str.lower().str.contains(movie_title.lower(), regex=False)]
        if not matches.empty:
            idx = int(matches.index[0])
            return idx, str(matches.iloc[0]["title"])
        raise ValueError(f"Movie '{movie_title}' not found")

    def _user_history_indices(self, user_id: int | None) -> tuple[list[int], np.ndarray]:
        if not user_id or user_id not in self.user_histories:
            return [], np.array([], dtype=float)
        history = pd.DataFrame(self.user_histories[user_id])
        liked = history[history["rating"] >= 4.0]
        if liked.empty:
            liked = history.sort_values("rating", ascending=False).head(12)
        indices = [
            self.movie_index_map[movie_id]
            for movie_id in liked["movieId"].tolist()
            if movie_id in self.movie_index_map
        ]
        ratings = liked["rating"].to_numpy(dtype=float)
        return indices, ratings

    def _score_user_affinity(
        self,
        candidate_idx: int,
        history_indices: list[int],
        ratings: np.ndarray,
    ) -> tuple[float, float, float]:
        if not history_indices or ratings.size == 0:
            return 0.0, 0.0, 0.0
        weights = np.clip(ratings - 2.5, 0.1, None)
        content_values = self.content_similarity[candidate_idx, history_indices]
        collaborative_values = self.collaborative_similarity[candidate_idx, history_indices]
        latent_values = self.latent_similarity[candidate_idx, history_indices]
        content_score = float(np.average(content_values, weights=weights))
        collaborative_score = float(np.average(collaborative_values, weights=weights))
        latent_score = float(np.average(latent_values, weights=weights))
        return content_score, collaborative_score, latent_score

    def _svd_prediction_score(self, user_id: int | None, movie_id: int) -> float:
        if not user_id or user_id not in self.user_histories:
            return 0.0
        estimate = self.svd_model.predict(str(user_id), str(movie_id)).est
        return float(np.clip((estimate - 0.5) / 4.5, 0, 1))

    def explain_recommendation(
        self,
        seed_title: str | None,
        candidate_idx: int,
        user_id: int | None = None,
    ) -> str:
        candidate = self.movies.iloc[candidate_idx]
        reasons: list[str] = []
        if seed_title:
            try:
                seed_idx, _ = self.resolve_title(seed_title)
                seed = self.movies.iloc[seed_idx]
                shared_genres = sorted(set(seed["genres_list"]) & set(candidate["genres_list"]))
                shared_keywords = sorted(set(seed["keywords_list"]) & set(candidate["keywords_list"]))
                shared_cast = sorted(set(seed["cast_list"]) & set(candidate["cast_list"]))
                if shared_genres:
                    reasons.append(f"shares {', '.join(shared_genres[:2])} with {seed['title']}")
                if shared_keywords:
                    reasons.append(f"matches topics like {', '.join(shared_keywords[:2])}")
                if seed["director"] and seed["director"] == candidate["director"]:
                    reasons.append(f"has the same director as {seed['title']}")
                elif shared_cast:
                    reasons.append(f"features cast overlap such as {', '.join(shared_cast[:2])}")
            except ValueError:
                pass

        if user_id and user_id in self.user_histories:
            history = pd.DataFrame(self.user_histories[user_id]).sort_values("rating", ascending=False).head(5)
            liked_titles = history["title"].tolist()
            if liked_titles:
                reasons.append(f"aligns with titles you rated highly like {liked_titles[0]}")

        if float(candidate["sentiment_score"]) >= 0.65:
            reasons.append("inherits strong positive sentiment patterns from the IMDb review model")
        if float(candidate["rating_count"]) >= 1000:
            reasons.append("is reinforced by strong collaborative audience behavior")
        return ". ".join(reasons[:3]).strip(". ") + "."

    def recommend(
        self,
        movie_title: str | None = None,
        user_id: int | None = None,
        top_n: int = 10,
        weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        weights = {**self.default_weights, **(weights or {})}
        history_indices, history_ratings = self._user_history_indices(user_id)
        watched_movie_ids = {entry["movieId"] for entry in self.user_histories.get(user_id or -1, [])}

        seed_idx: int | None = None
        canonical_title: str | None = None
        if movie_title:
            seed_idx, canonical_title = self.resolve_title(movie_title)

        candidates: set[int] = set()
        if seed_idx is not None:
            content_neighbors = np.argsort(self.content_similarity[seed_idx])[::-1][1:180]
            collaborative_neighbors = np.argsort(self.collaborative_similarity[seed_idx])[::-1][1:180]
            latent_neighbors = np.argsort(self.latent_similarity[seed_idx])[::-1][1:180]
            candidates.update(map(int, content_neighbors))
            candidates.update(map(int, collaborative_neighbors))
            candidates.update(map(int, latent_neighbors))
        else:
            candidates.update(
                self.movies.sort_values(["popularity_score", "avg_rating"], ascending=False).head(300).index.tolist()
            )

        for history_idx in history_indices[:8]:
            content_neighbors = np.argsort(self.content_similarity[history_idx])[::-1][1:40]
            collaborative_neighbors = np.argsort(self.collaborative_similarity[history_idx])[::-1][1:40]
            latent_neighbors = np.argsort(self.latent_similarity[history_idx])[::-1][1:40]
            candidates.update(map(int, content_neighbors))
            candidates.update(map(int, collaborative_neighbors))
            candidates.update(map(int, latent_neighbors))

        scored: list[dict[str, Any]] = []
        for candidate_idx in candidates:
            movie = self.movies.iloc[candidate_idx]
            movie_id = int(movie["movieId"])
            if seed_idx is not None and candidate_idx == seed_idx:
                continue
            if user_id and movie_id in watched_movie_ids:
                continue

            content_seed_score = (
                float(self.content_similarity[seed_idx, candidate_idx]) if seed_idx is not None else 0.0
            )
            collaborative_seed_score = (
                float(self.collaborative_similarity[seed_idx, candidate_idx]) if seed_idx is not None else 0.0
            )
            latent_seed_score = (
                float(self.latent_similarity[seed_idx, candidate_idx]) if seed_idx is not None else 0.0
            )

            content_user_score, collaborative_user_score, latent_user_score = self._score_user_affinity(
                candidate_idx,
                history_indices,
                history_ratings,
            )
            content_score = max(content_seed_score, content_user_score)
            collaborative_score = max(collaborative_seed_score, collaborative_user_score)
            svd_similarity_score = max(latent_seed_score, latent_user_score)
            svd_prediction_score = self._svd_prediction_score(user_id, movie_id)
            svd_score = max(svd_similarity_score, svd_prediction_score)

            popularity_score = float(movie["popularity_score"])
            sentiment_score = float(movie["sentiment_score"])

            final_score = (
                weights["content"] * content_score
                + weights["collaborative"] * collaborative_score
                + weights["svd"] * svd_score
                + weights["popularity"] * popularity_score
                + weights["sentiment"] * sentiment_score
            )

            scored.append(
                {
                    "title": str(movie["title"]),
                    "movieId": movie_id,
                    "year": int(movie["release_year"]),
                    "genres": movie["genres_list"],
                    "director": str(movie["director"]),
                    "cast": movie["cast_list"],
                    "avg_rating": round(float(movie["avg_rating"]), 2),
                    "rating_count": int(movie["rating_count"]),
                    "final_score": round(float(final_score), 4),
                    "scores": {
                        "content": round(content_score, 4),
                        "collaborative": round(collaborative_score, 4),
                        "svd": round(svd_score, 4),
                        "popularity": round(popularity_score, 4),
                        "sentiment": round(sentiment_score, 4),
                    },
                    "reason": self.explain_recommendation(canonical_title, candidate_idx, user_id),
                }
            )

        scored.sort(key=lambda item: item["final_score"], reverse=True)
        return scored[:top_n]

    def similar_movies(self, movie_title: str, top_n: int = 10) -> list[dict[str, Any]]:
        seed_idx, canonical_title = self.resolve_title(movie_title)
        neighbors = self.recommend(movie_title=canonical_title, top_n=top_n + 1)
        return [item for item in neighbors if item["movieId"] != int(self.movies.iloc[seed_idx]["movieId"])][:top_n]

    def user_profile(self, user_id: int) -> dict[str, Any]:
        if user_id not in self.user_histories:
            raise ValueError(f"User {user_id} not found")
        profile = self.ratings_with_titles[self.ratings_with_titles["userId"] == user_id].copy()
        liked = profile[profile["rating"] >= 4.0].copy()
        if liked.empty:
            liked = profile.sort_values("rating", ascending=False).head(20).copy()

        genre_counter: Counter[str] = Counter()
        actor_counter: Counter[str] = Counter()
        director_counter: Counter[str] = Counter()
        decade_counter: Counter[str] = Counter()

        for _, row in liked.iterrows():
            weight = float(row["rating"])
            for genre in row["genres_list"]:
                genre_counter[genre] += weight
            for actor in row["cast_list"][:6]:
                actor_counter[actor] += weight
            if row["director"]:
                director_counter[str(row["director"])] += weight
            decade_counter[f"{int(row['decade'])}s"] += weight

        return {
            "user_id": user_id,
            "ratings_count": int(len(profile)),
            "avg_rating": round(float(profile["rating"].mean()), 2),
            "favorite_movies": profile.sort_values("rating", ascending=False)[["title", "rating", "release_year"]]
            .head(10)
            .to_dict("records"),
            "top_genres": [{"label": key, "value": round(value, 2)} for key, value in genre_counter.most_common(8)],
            "top_actors": [{"label": key, "value": round(value, 2)} for key, value in actor_counter.most_common(8)],
            "top_directors": [
                {"label": key, "value": round(value, 2)} for key, value in director_counter.most_common(8)
            ],
            "decade_preferences": [
                {"label": key, "value": round(value, 2)} for key, value in decade_counter.most_common()
            ],
            "taste_summary": self._taste_summary(genre_counter, decade_counter, director_counter),
        }

    def _taste_summary(
        self,
        genres: Counter[str],
        decades: Counter[str],
        directors: Counter[str],
    ) -> str:
        summary: list[str] = []
        if genres:
            summary.append(f"leans toward {genres.most_common(1)[0][0].lower()} stories")
        if decades:
            summary.append(f"prefers {decades.most_common(1)[0][0]} releases")
        if directors:
            summary.append(f"often rates {directors.most_common(1)[0][0]} highly")
        return ", ".join(summary).capitalize() + "."
