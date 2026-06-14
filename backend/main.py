from fastapi import FastAPI, HTTPException, Query

from cineiq import HybridRecommender

app = FastAPI(
    title="CINEIQ API",
    description="Explainable hybrid movie recommendation service built on MovieLens 25M, TMDB 45K metadata, IMDb 50K reviews, Surprise SVD, and Streamlit analytics.",
    version="3.0.0",
)

engine = HybridRecommender()


@app.get("/")
def home() -> dict:
    return {
        "message": "Welcome to CINEIQ",
        "capabilities": [
            "hybrid recommendations",
            "similar movies",
            "user taste profiles",
            "explainable suggestions",
            "sentiment-aware reranking",
            "mlflow-backed training metadata",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "metadata": engine.metadata()}


@app.get("/movies")
def get_movies(limit: int = Query(default=200, ge=10, le=1000)) -> dict:
    return {"movies": engine.list_movies(limit=limit)}


@app.get("/users")
def get_users() -> dict:
    return {"users": engine.list_users()}


@app.get("/recommend")
def get_recommendations(
    movie: str | None = None,
    user_id: int | None = None,
    top_n: int = Query(default=10, ge=3, le=25),
    content_weight: float | None = Query(default=None, ge=0.0, le=1.0),
    collaborative_weight: float | None = Query(default=None, ge=0.0, le=1.0),
    svd_weight: float | None = Query(default=None, ge=0.0, le=1.0),
    popularity_weight: float | None = Query(default=None, ge=0.0, le=1.0),
    sentiment_weight: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict:
    weights = {
        "content": content_weight,
        "collaborative": collaborative_weight,
        "svd": svd_weight,
        "popularity": popularity_weight,
        "sentiment": sentiment_weight,
    }
    weights = {key: value for key, value in weights.items() if value is not None}
    try:
        recommendations = engine.recommend(
            movie_title=movie,
            user_id=user_id,
            top_n=top_n,
            weights=weights or None,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "seed_movie": movie,
        "user_id": user_id,
        "top_n": top_n,
        "weights": weights or engine.default_weights,
        "recommendations": recommendations,
    }


@app.get("/similar")
def get_similar_movies(movie: str, top_n: int = Query(default=10, ge=3, le=25)) -> dict:
    try:
        recommendations = engine.similar_movies(movie, top_n=top_n)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"movie": movie, "similar": recommendations}


@app.get("/profile/{user_id}")
def get_profile(user_id: int) -> dict:
    try:
        return engine.user_profile(user_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/explain")
def explain_candidate(
    candidate: str,
    movie: str | None = None,
    user_id: int | None = None,
) -> dict:
    try:
        candidate_idx, canonical = engine.resolve_title(candidate)
        reason = engine.explain_recommendation(movie, candidate_idx, user_id=user_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"candidate": canonical, "seed_movie": movie, "user_id": user_id, "reason": reason}
