import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("CINEIQ_API_URL", "http://127.0.0.1:8001")
PLOTLY_TEMPLATE = "plotly_dark"

ACCENT = "#7c5cff"
ACCENT_2 = "#22d3ee"
INK = "#eef1fb"
MUTED = "#98a2c0"

# Stable color per genre so chips stay consistent across cards.
GENRE_COLORS = [
    "#7c5cff", "#22d3ee", "#f472b6", "#fbbf24", "#34d399",
    "#60a5fa", "#fb7185", "#a78bfa", "#2dd4bf", "#f59e0b",
]


def api_get(path: str, params: dict | None = None) -> dict:
    query = f"?{urlencode(params)}" if params else ""
    with urlopen(f"{API_URL}{path}{query}") as response:
        return json.loads(response.read().decode("utf-8"))


def genre_chip(name: str) -> str:
    color = GENRE_COLORS[sum(ord(c) for c in name) % len(GENRE_COLORS)]
    return (
        f'<span class="chip" style="--c:{color};">{name}</span>'
    )


def star_row(avg_rating: float) -> str:
    """Render a 5-star row from a 0-5 average rating."""
    rating = max(0.0, min(5.0, float(avg_rating)))
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half
    stars = "★" * full + ("⯨" if half else "") + "☆" * empty
    return f'<span class="stars">{stars}</span><span class="stars-num">{rating:.1f}</span>'


def score_bar(score: float) -> str:
    pct = max(0, min(100, round(float(score) * 100)))
    return (
        f'<div class="score-track"><div class="score-fill" style="width:{pct}%;"></div></div>'
    )


def section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Inter, system-ui, sans-serif"),
        margin=dict(l=12, r=12, t=52, b=12),
        legend_title="",
        title_font=dict(size=15, color=INK),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(154,163,184,0.08)", zerolinecolor="rgba(154,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(154,163,184,0.08)", zerolinecolor="rgba(154,163,184,0.12)")
    return fig


@st.cache_data(ttl=120)
def load_movies() -> list[str]:
    return api_get("/movies", {"limit": 500})["movies"]


@st.cache_data(ttl=120)
def load_users() -> list[int]:
    return api_get("/users")["users"]


@st.cache_data(ttl=120)
def load_health() -> dict:
    return api_get("/health")


st.set_page_config(page_title="CINEIQ", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
        --bg: #090b14;
        --surface: #141828;
        --surface-2: #1b2034;
        --border: rgba(150, 160, 200, 0.14);
        --ink: #eef1fb;
        --muted: #98a2c0;
        --accent: #7c5cff;
        --accent-2: #22d3ee;
    }
    .stApp {
        background:
            radial-gradient(1100px 520px at 78% -8%, rgba(124, 92, 255, 0.18), transparent 60%),
            radial-gradient(900px 480px at 8% -6%, rgba(34, 211, 238, 0.10), transparent 55%),
            var(--bg);
        color: var(--ink);
        font-family: 'Inter', system-ui, sans-serif;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }
    #MainMenu, header[data-testid="stHeader"] { background: transparent; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        border: 1px solid var(--border);
        background:
            radial-gradient(700px 300px at 90% 0%, rgba(124,92,255,0.28), transparent 60%),
            linear-gradient(135deg, #1a1640 0%, #131a33 48%, #0e1326 100%);
        padding: 2.1rem 2.2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 24px 60px rgba(0,0,0,0.45);
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -60px; top: -60px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(34,211,238,0.22), transparent 70%);
        filter: blur(8px);
    }
    .brand-row { display: flex; align-items: center; gap: 0.85rem; }
    .brand-badge {
        width: 46px; height: 46px;
        display: grid; place-items: center;
        font-size: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 8px 22px rgba(124,92,255,0.45);
    }
    .brand-name {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        padding-bottom: 2px;
        background: linear-gradient(90deg, #ffffff 10%, #c9b8ff 55%, #7be8fb 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-tag {
        color: #c7cdec;
        font-size: 1.02rem;
        font-weight: 500;
        margin: 0.9rem 0 1.3rem 0;
        max-width: 680px;
        line-height: 1.6;
    }
    .stat-row { display: flex; flex-wrap: wrap; gap: 0.7rem; }
    .stat {
        position: relative; z-index: 1;
        padding: 0.75rem 1.15rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(6px);
        min-width: 130px;
    }
    .stat-num {
        font-size: 1.4rem; font-weight: 800; color: #fff;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em;
        text-transform: uppercase; color: var(--muted); margin-top: 0.25rem;
    }

    /* ---------- Section headers ---------- */
    .section { margin: 0.6rem 0 1.2rem 0; }
    .section-title {
        color: var(--ink); font-size: 1.3rem; font-weight: 750;
        letter-spacing: -0.01em;
    }
    .section-subtitle {
        color: var(--muted); font-size: 0.92rem; line-height: 1.55;
        margin-top: 0.3rem; max-width: 780px;
    }

    /* ---------- Movie cards ---------- */
    .movie-card {
        position: relative; overflow: hidden;
        background: linear-gradient(180deg, var(--surface) 0%, #11152340 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem 1.25rem 1.15rem 1.45rem;
        margin-bottom: 1.05rem;
        transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease;
    }
    .movie-card::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
        background: linear-gradient(180deg, var(--accent), var(--accent-2));
    }
    .movie-card:hover {
        border-color: rgba(124,92,255,0.55);
        transform: translateY(-3px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.4);
    }
    .card-top {
        display: flex; justify-content: space-between; align-items: flex-start;
        gap: 0.8rem; margin-bottom: 0.55rem;
    }
    .card-title-wrap { display: flex; align-items: baseline; gap: 0.6rem; }
    .rank {
        font-size: 0.85rem; font-weight: 800; color: var(--accent);
        background: rgba(124,92,255,0.14); border: 1px solid rgba(124,92,255,0.3);
        border-radius: 8px; padding: 0.12rem 0.5rem; line-height: 1.5;
    }
    .card-title { color: #fff; font-size: 1.1rem; font-weight: 720; line-height: 1.3; }
    .card-year { color: var(--muted); font-size: 0.9rem; font-weight: 500; }
    .stars { color: #fbbf24; font-size: 0.98rem; letter-spacing: 1px; }
    .stars-num { color: var(--muted); font-size: 0.82rem; margin-left: 0.4rem; font-weight: 600; }

    .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.75rem 0 0.85rem 0; }
    .chip {
        font-size: 0.74rem; font-weight: 600; padding: 0.22rem 0.6rem;
        border-radius: 999px; color: var(--c);
        background: color-mix(in srgb, var(--c) 14%, transparent);
        border: 1px solid color-mix(in srgb, var(--c) 35%, transparent);
    }
    .crew { color: var(--muted); font-size: 0.85rem; line-height: 1.5; margin-bottom: 0.7rem; }
    .crew strong { color: var(--ink); font-weight: 600; }

    .score-line { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.85rem; }
    .score-label { font-size: 0.72rem; color: var(--muted); font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase; white-space: nowrap; }
    .score-track {
        flex: 1; height: 7px; border-radius: 999px; background: rgba(255,255,255,0.07);
        overflow: hidden;
    }
    .score-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }
    .score-val { font-size: 0.82rem; font-weight: 700; color: #c9b8ff; white-space: nowrap; }

    .reason {
        padding: 0.72rem 0.85rem; border-radius: 12px;
        background: var(--surface-2);
        color: #cdd4ee; font-size: 0.86rem; line-height: 1.55;
    }
    .reason strong { color: var(--accent-2); }

    .banner {
        padding: 0.85rem 1.05rem; border-radius: 13px;
        background: linear-gradient(135deg, rgba(124,92,255,0.16), rgba(34,211,238,0.10));
        border: 1px solid rgba(124,92,255,0.28);
        color: #e6e8fb; font-size: 0.92rem; margin-bottom: 1.2rem;
    }

    /* metrics in taste profile */
    .metric {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 14px; padding: 1rem 1.1rem;
    }
    .metric-label { color: var(--muted); font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase; }
    .metric-value { color: #fff; font-size: 1.55rem; font-weight: 780;
        line-height: 1.2; margin-top: 0.3rem; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0f1d 0%, #090b14 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    }
    [data-testid="stSidebar"] label {
        color: var(--muted) !important; font-weight: 600 !important;
        font-size: 0.78rem !important; letter-spacing: 0.03em;
    }
    .side-brand {
        display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.2rem;
    }
    .side-badge {
        width: 38px; height: 38px; display: grid; place-items: center; font-size: 1.2rem;
        border-radius: 11px; background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 6px 16px rgba(124,92,255,0.4);
    }
    .side-name {
        font-size: 1.25rem; font-weight: 800; letter-spacing: -0.01em;
        background: linear-gradient(90deg,#fff,#c9b8ff); -webkit-background-clip: text;
        background-clip: text; -webkit-text-fill-color: transparent;
    }
    .side-ver { color: var(--muted); font-size: 0.74rem; font-weight: 500; margin: 0.1rem 0 0 0.1rem; }
    .side-group {
        color: #c9b8ff; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; margin: 0.4rem 0 0.1rem 0;
    }
    .status {
        display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem;
        color: var(--ink); padding: 0.5rem 0.7rem; border-radius: 10px;
        background: var(--surface); border: 1px solid var(--border);
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; box-shadow: 0 0 8px currentColor; }
    .dot.ok { background: #34d399; color: #34d399; }
    .dot.down { background: #fb7185; color: #fb7185; }
    [data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }

    /* ---------- Tabs ---------- */
    div[data-testid="stTabs"] button { height: 42px; color: var(--muted); font-weight: 650; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #fff; }
    div[data-testid="stTabs"] [data-baseweb="tab-border"] { background: rgba(150,160,200,0.12) !important; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important; height: 3px !important;
    }
    .stMarkdown p, .stMarkdown li { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)

health = load_health()
metadata = health["metadata"]
movie_options = load_movies()
user_options = load_users()

st.markdown(
    f"""
    <div class="hero">
        <div class="brand-row">
            <div class="brand-badge">🎬</div>
            <div class="brand-name">CINEIQ</div>
        </div>
        <div class="hero-tag">
            Explainable movie recommendations — a hybrid engine that blends viewing behavior,
            content similarity, and review sentiment, and tells you <em>why</em> every title made the cut.
        </div>
        <div class="stat-row">
            <div class="stat"><div class="stat-num">{metadata['movies']:,}</div><div class="stat-label">Movies</div></div>
            <div class="stat"><div class="stat-num">{metadata['users']:,}</div><div class="stat-label">Users</div></div>
            <div class="stat"><div class="stat-num">{metadata['ratings']:,}</div><div class="stat-label">Ratings</div></div>
            <div class="stat"><div class="stat-num">IMDb 50K</div><div class="stat-label">Sentiment</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar controls ---
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="side-badge">🎬</div>
            <div>
                <div class="side-name">CINEIQ</div>
            </div>
        </div>
        <div class="side-ver">Recommendation Studio · v3.0</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="side-group">Discover</div>', unsafe_allow_html=True)
    selected_movie = st.selectbox("Seed movie", options=movie_options, index=0)
    selected_user = st.selectbox(
        "User profile (optional)",
        options=[None] + user_options,
        index=0,
        format_func=lambda value: "None — movie-only" if value is None else f"User {value}",
        help="A MovieLens demo user ID used to personalize results from rating history.",
    )
    top_n = st.slider("Number of results", min_value=5, max_value=15, value=8)

    st.markdown('<div class="side-group">Tune the engine</div>', unsafe_allow_html=True)
    weight_preset = st.segmented_control(
        "Preset",
        options=["Balanced", "Personalized", "Trending"],
        default="Balanced",
        help="Quick presets adjust the ensemble weights. Fine-tune them below.",
    )
    presets = {
        "Balanced": (0.35, 0.25, 0.20, 0.10, 0.10),
        "Personalized": (0.20, 0.40, 0.25, 0.05, 0.10),
        "Trending": (0.20, 0.15, 0.10, 0.40, 0.15),
    }
    base = presets.get(weight_preset, presets["Balanced"])
    with st.expander("Ensemble weights", expanded=False):
        weight_content = st.slider("Content similarity", 0.0, 1.0, base[0], 0.05)
        weight_collaborative = st.slider("Collaborative filtering", 0.0, 1.0, base[1], 0.05)
        weight_svd = st.slider("SVD latent factors", 0.0, 1.0, base[2], 0.05)
        weight_popularity = st.slider("Popularity", 0.0, 1.0, base[3], 0.05)
        weight_sentiment = st.slider("Sentiment", 0.0, 1.0, base[4], 0.05)

    st.divider()
    st.markdown(
        f"""
        <div class="status">
            <span class="dot ok"></span>
            <span>API online · <strong>{metadata['movies']:,}</strong> titles loaded</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Endpoint `{API_URL}` · Sentiment `{metadata['sentiment_source']}`")

tab_recommend, tab_similar, tab_dashboard = st.tabs(
    ["✨ Recommendations", "🎞️ Similar Titles", "📊 Taste Profile"]
)

# --- Recommendations ---
with tab_recommend:
    section_header(
        "Recommendations",
        "A weighted blend of content similarity, collaborative filtering, SVD matrix factorization, "
        "popularity, and review sentiment — each suggestion comes with the reason behind it.",
    )

    params = {"top_n": top_n}
    if selected_movie:
        params["movie"] = selected_movie
    if selected_user:
        params["user_id"] = selected_user
    params.update(
        {
            "content_weight": weight_content,
            "collaborative_weight": weight_collaborative,
            "svd_weight": weight_svd,
            "popularity_weight": weight_popularity,
            "sentiment_weight": weight_sentiment,
        }
    )
    rec_response = api_get("/recommend", params)
    recs = rec_response["recommendations"]

    if selected_user:
        st.markdown(
            f'<div class="banner">🎯 Personalized for <strong>user {selected_user}</strong> — '
            "blending seed-title similarity with their rating history.</div>",
            unsafe_allow_html=True,
        )

    export_frame = pd.DataFrame(
        [
            {
                "Rank": i + 1,
                "Title": item["title"],
                "Year": item["year"],
                "Avg Rating": item["avg_rating"],
                "Ratings": item["rating_count"],
                "Score": item["final_score"],
                "Genres": ", ".join(item["genres"]),
                "Director": item["director"],
                "Reason": item["reason"],
            }
            for i, item in enumerate(recs)
        ]
    )
    bar = st.columns([3, 1.1, 1.3])
    with bar[0]:
        st.caption(
            f"Showing **{len(recs)}** titles for seed **{selected_movie}** · "
            f"{'personalized' if selected_user else 'movie-only'} mode"
        )
    with bar[1]:
        show_breakdown = st.toggle("Score breakdown", value=True)
    with bar[2]:
        st.download_button(
            "⬇ Export CSV",
            data=export_frame.to_csv(index=False).encode("utf-8"),
            file_name="cineiq_recommendations.csv",
            mime="text/csv",
            width="stretch",
        )

    for start in range(0, len(recs), 2):
        row = st.columns(2)
        for offset, (column, item) in enumerate(zip(row, recs[start : start + 2])):
            rank = start + offset + 1
            with column:
                genres = item["genres"][:3] if item["genres"] else ["Mixed"]
                chips = "".join(genre_chip(g) for g in genres)
                cast = ", ".join(item["cast"][:3]) if item["cast"] else "Cast unavailable"
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="card-top">
                            <div>
                                <div class="card-title-wrap">
                                    <span class="rank">#{rank}</span>
                                    <span class="card-title">{item['title']}</span>
                                </div>
                                <div style="margin-top:0.45rem;">{star_row(item['avg_rating'])}
                                    <span class="card-year"> · {item['year']} · {item['rating_count']:,} ratings</span>
                                </div>
                            </div>
                        </div>
                        <div class="chips">{chips}</div>
                        <div class="crew"><strong>Director:</strong> {item['director'] or 'Unknown'} &nbsp;·&nbsp; <strong>Cast:</strong> {cast}</div>
                        <div class="score-line">
                            <span class="score-label">Match</span>
                            {score_bar(item['final_score'])}
                            <span class="score-val">{item['final_score']}</span>
                        </div>
                        <div class="reason"><strong>Why →</strong> {item['reason']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if show_breakdown:
        score_frame = pd.DataFrame(
            [
                {
                    "title": item["title"],
                    "Content": item["scores"]["content"],
                    "Collaborative": item["scores"]["collaborative"],
                    "SVD": item["scores"]["svd"],
                    "Popularity": item["scores"]["popularity"],
                    "Sentiment": item["scores"]["sentiment"],
                }
                for item in recs
            ]
        )
        score_melted = score_frame.melt(id_vars="title", var_name="Signal", value_name="Score")
        fig = px.bar(
            score_melted,
            x="title",
            y="Score",
            color="Signal",
            title="How each recommendation was scored",
            barmode="stack",
            color_discrete_sequence=["#7c5cff", "#22d3ee", "#f472b6", "#fbbf24", "#34d399"],
        )
        fig.update_layout(xaxis_title="", yaxis_title="Signal strength", xaxis_tickangle=-25)
        st.plotly_chart(style_figure(fig, height=420), width="stretch")

# --- Similar titles ---
with tab_similar:
    section_header(
        "Similar Titles",
        "Nearest neighbors to the seed movie using both content similarity and latent collaborative structure.",
    )
    similar = api_get("/similar", {"movie": selected_movie, "top_n": top_n})["similar"]
    similar_frame = pd.DataFrame(similar)[
        ["title", "year", "avg_rating", "rating_count", "final_score", "reason"]
    ].rename(
        columns={
            "title": "Title",
            "year": "Year",
            "avg_rating": "Avg Rating",
            "rating_count": "Ratings",
            "final_score": "Score",
            "reason": "Why",
        }
    )
    st.dataframe(similar_frame, width="stretch", hide_index=True)

# --- Taste profile ---
with tab_dashboard:
    section_header(
        "Taste Profile",
        "Genre, decade, director, and actor affinities derived from a user's rating history.",
    )
    if not selected_user:
        st.info("Select a user profile in the sidebar to view personalized taste analytics.")
    else:
        profile = api_get(f"/profile/{selected_user}")
        top_genre = profile["top_genres"][0]["label"] if profile["top_genres"] else "N/A"
        top_decade = profile["decade_preferences"][0]["label"] if profile["decade_preferences"] else "N/A"

        profile_cols = st.columns(3)
        profile_cols[0].metric("Ratings analyzed", f"{profile['ratings_count']:,}", border=True)
        profile_cols[1].metric("Top genre", top_genre, border=True)
        profile_cols[2].metric("Top decade", top_decade, border=True)

        st.markdown(
            f'<div class="banner">🍿 {profile["taste_summary"]}</div>',
            unsafe_allow_html=True,
        )

        top_genres = pd.DataFrame(profile["top_genres"])
        if not top_genres.empty:
            radar = go.Figure(
                data=go.Scatterpolar(
                    r=top_genres["value"],
                    theta=top_genres["label"],
                    fill="toself",
                    name="Genres",
                    line=dict(color=ACCENT, width=2.5),
                    fillcolor="rgba(124, 92, 255, 0.32)",
                )
            )
            radar.update_layout(title="Genre affinity", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(style_figure(radar, height=420), width="stretch")

        chart_columns = st.columns(2)
        with chart_columns[0]:
            decade_frame = pd.DataFrame(profile["decade_preferences"])
            if not decade_frame.empty:
                decade_fig = px.bar(
                    decade_frame, x="label", y="value", title="Decade preferences",
                    color_discrete_sequence=[ACCENT],
                )
                decade_fig.update_layout(xaxis_title="", yaxis_title="")
                st.plotly_chart(style_figure(decade_fig, height=340), width="stretch")
        with chart_columns[1]:
            director_frame = pd.DataFrame(profile["top_directors"])
            if not director_frame.empty:
                director_fig = px.bar(
                    director_frame, x="value", y="label", orientation="h",
                    title="Director affinities", color_discrete_sequence=[ACCENT_2],
                )
                director_fig.update_layout(xaxis_title="", yaxis_title="")
                st.plotly_chart(style_figure(director_fig, height=340), width="stretch")

        affinity_columns = st.columns(2)
        with affinity_columns[0]:
            actor_frame = pd.DataFrame(profile["top_actors"])
            if not actor_frame.empty:
                actor_fig = px.bar(
                    actor_frame, x="value", y="label", orientation="h",
                    title="Actor affinities", color_discrete_sequence=["#f472b6"],
                )
                actor_fig.update_layout(xaxis_title="", yaxis_title="")
                st.plotly_chart(style_figure(actor_fig, height=340), width="stretch")
        with affinity_columns[1]:
            st.markdown(
                '<div class="section-title" style="font-size:0.98rem;margin-bottom:0.6rem;">Favorite movies</div>',
                unsafe_allow_html=True,
            )
            favorite_movies = pd.DataFrame(profile["favorite_movies"]).rename(
                columns={"title": "Title", "rating": "Rating", "release_year": "Year"}
            )
            st.dataframe(favorite_movies, width="stretch", hide_index=True)
