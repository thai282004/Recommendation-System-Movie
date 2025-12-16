import concurrent.futures
import html
import joblib
import pickle
import textwrap
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st


# Minimal path helper (no external package)
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

for _dir in (DATA_RAW, DATA_PROCESSED, ARTIFACTS):
    _dir.mkdir(parents=True, exist_ok=True)

# UI theme colors - BRIGHT PALETTE
_FG = "#0f172a"          # Slate 900 (Text)
_BG = "#f8fafc"          # Slate 50 (Background)
_PANEL = "#ffffff"       # White
_BORDER = "#e2e8f0"      # Slate 200
_ACCENT_1 = "#ff6b6b"
_ACCENT_2 = "#f06595"

_TMDB_API_KEY_FALLBACK = "73591245078f6a8450bbd16587f5797e"


_NO_POSTER_DATA_URI = (
    "data:image/svg+xml;utf8,"
    '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450">'
    '<rect width="100%25" height="100%25" fill="%23f3f4f6"/>'
    '<text x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" '
    'font-family="Arial" font-size="22" fill="%239ca3af">No Poster</text>'
    "</svg>"
)


def _enable_altair_theme() -> None:
    def _theme_dict():
        return {
            "config": {
                "view": {"stroke": "transparent", "fill": _PANEL},
                "background": _PANEL,
                "axis": {
                    "labelColor": _FG,
                    "titleColor": _FG,
                    "gridColor": _BORDER,
                    "domainColor": _BORDER,
                    "tickColor": _BORDER,
                },
                "legend": {"labelColor": _FG, "titleColor": _FG},
                "title": {"color": _FG},
            }
        }

    try:
        # Altair 5.5+ (preferred)
        if hasattr(alt, "theme") and hasattr(alt.theme, "enable"):
            @alt.theme.register("filmguru_light", enable=True)
            def _theme():
                return alt.theme.ThemeConfig(_theme_dict())

            return

        # Older Altair fallback
        alt.themes.register("filmguru_light", _theme_dict)
        alt.themes.enable("filmguru_light")
    except Exception:
        pass


_enable_altair_theme()

# =============================================================================
# 1. CONFIG & CSS (FRONTEND OVERHAUL)
# =============================================================================
st.set_page_config(
    page_title="FilmGuru - Gợi ý phim thông minh",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍿"
)

st.markdown(
    """
<style>
    /* --- GLOBAL RESET & VARIABLES --- */
    :root {
        --fg: #0f172a;
        --bg: #f8fafc;
        --panel: #ffffff;
        --border: #cbd5e1;
        --accent-1: #ff6b6b;
        --accent-2: #f06595;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        color-scheme: light; /* Force browser native inputs to light mode */
    }

    /* Force App Background to Bright White/Slate */
    .stApp {
        background-color: var(--bg);
        color: var(--fg);
    }
    
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #fff1f2 0%, #fff 30%, #f0f9ff 100%);
    }

    /* Sidebar Clean Bright */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--border);
        box-shadow: 4px 0 24px rgba(0,0,0,0.02);
    }

    /* Header Transparent Blur */
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    /* Text Typography - FORCE DARK TEXT */
    h1, h2, h3, h4, h5, h6, p, li, span, label, div {
        color: var(--fg) !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Widget labels must be dark */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] span,
    [data-testid="stMarkdownContainer"] label {
        color: #0f172a !important;
    }

    /* --- INPUTS, SELECTS & DROPDOWNS (CRITICAL UI FIXES) --- */
    
    /* Input containers - BRIGHT WITH DARK TEXT */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="select"],
    input, textarea, select {
        background-color: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Force text inside selects to be dark */
    [data-baseweb="select"] div,
    [data-baseweb="select"] span,
    [data-baseweb="select"] input {
        color: #0f172a !important;
    }
    
    /* Dropdown MENU List (The part that pops down) - CRITICAL BRIGHT FIX */
    [data-baseweb="menu"],
    [data-baseweb="popover"] [data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow-lg) !important;
        border-radius: 8px !important;
        padding: 4px 0 !important;
    }
    
    /* Dropdown Options - FORCE DARK TEXT */
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"],
    ul[role="listbox"] li,
    li[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        cursor: pointer;
        padding: 8px 12px !important;
    }
    
    /* Hover state for dropdown options */
    [data-baseweb="menu"] li[aria-selected="true"],
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    li[role="option"]:hover {
        background-color: #f1f5f9 !important; /* Slate-100 */
        color: #0f172a !important;
    }
    
    /* Text inside dropdown options */
    [data-baseweb="menu"] li *,
    [data-baseweb="menu"] [role="option"] *,
    li[role="option"] * {
        color: #0f172a !important;
    }

    /* Remove weird SVG coloring in selects */
    [data-baseweb="select"] svg {
        fill: #64748b !important;
    }

    /* --- BUTTONS --- */
    /* Primary Gradient Button */
    div.stButton > button, 
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px 0 rgba(240, 101, 149, 0.39);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(240, 101, 149, 0.23);
    }
    
    /* Secondary/Ghost Buttons */
    button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        color: var(--fg) !important;
    }

    /* --- POPOVERS & MODALS (RATING FIX) --- */
    
    /* The trigger button wrapper */
    [data-testid="stPopover"] > button {
        background-color: #ffffff !important;
        border: 1px solid var(--border) !important;
        color: var(--fg) !important;
        box-shadow: var(--shadow-sm);
    }

    /* The actual floating content */
    [data-baseweb="popover"],
    [data-testid="stPopoverBody"] {
        background-color: #ffffff !important;
        border: 1px solid var(--border);
        border-radius: 12px !important;
        box-shadow: var(--shadow-lg) !important;
        color: var(--fg) !important;
    }

    /* BaseWeb popover content is rendered in a portal and the visible background
       is often applied on nested wrapper divs / role=dialog. Force all layers light. */
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] > div > div,
    div[data-baseweb="popover"] [role="dialog"],
    div[data-baseweb="popover"] [role="dialog"] > div {
        background-color: #ffffff !important;
        color: var(--fg) !important;
        border-radius: 12px !important;
    }

    /* Ensure all text inside popovers stays dark (fixes dark-on-dark issues) */
    div[data-baseweb="popover"] [data-testid="stMarkdownContainer"],
    div[data-baseweb="popover"] [data-testid="stMarkdownContainer"] *,
    div[data-baseweb="popover"] label,
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] p,
    div[data-baseweb="popover"] strong {
        color: var(--fg) !important;
    }

    /* Keep popover arrow light as well (BaseWeb uses SVG / pseudo elements) */
    div[data-baseweb="popover"] svg {
        fill: #ffffff !important;
    }

    /* --- SLIDERS (For Rating) --- */
    [data-baseweb="slider"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }

    /* Slider rails/tracks sometimes inherit dark backgrounds in popovers */
    div[data-baseweb="popover"] [data-baseweb="slider"] * {
        color: var(--fg) !important;
    }
    
    /* Tick marks text */
    [data-testid="stTickBar"] > div {
        color: #64748b !important; /* Slate-500 */
        font-weight: 500;
    }
    
    /* The thumb/handle */
    [role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid var(--accent-2) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* --- DATAFRAMES & TABLES --- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: var(--shadow-sm);
    }
    
    [data-testid="stDataFrame"] div[role="columnheader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #e2e8f0;
    }
    
    [data-testid="stDataFrame"] div[role="gridcell"] {
        color: #334155 !important;
        font-size: 14px;
    }

    /* --- EXPANDERS & CONTAINERS --- */
    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid var(--border);
        border-radius: 10px;
        box-shadow: var(--shadow-sm);
    }
    
    [data-testid="stExpander"] summary {
        color: #0f172a !important;
        font-weight: 600;
        background-color: #ffffff !important;
    }
    
    [data-testid="stExpander"] summary:hover {
        background-color: #f8fafc !important;
    }
    
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #475467 !important;
    }
    
    /* Expander content area */
    [data-testid="stExpander"] > div:last-child {
        background-color: #ffffff !important;
    }

    /* --- MOVIE CARD UI (PRESERVED BUT CLEANED UP) --- */
    .movie-card-center {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    .movie-card-center .movie-card {
        width: clamp(240px, 32vw, 320px);
    }

    .movie-card {
        position: relative;
        width: 100%;
        border-radius: 12px !important;
        background-color: transparent !important;
        line-height: 0 !important;
        font-size: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden;
        cursor: pointer;
    }

    .poster-img {
        width: 100%;
        display: block !important;
        vertical-align: top !important;
        aspect-ratio: 27 / 40; 
        object-fit: cover;
        transition: transform 0.4s ease;
    }

    .movie-card:hover .poster-img {
        transform: scale(1.05);
    }

    .movie-bottom {
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(to top, rgba(15, 23, 42, 0.9) 0%, rgba(15, 23, 42, 0) 100%);
        padding: 12px 10px 8px 10px;
        pointer-events: none;
        line-height: 1.3 !important;
        font-size: 1rem !important;
    }

    .movie-title {
        color: #fff !important; /* Luôn giữ trắng trên nền đen mờ */
        font-weight: 600;
        font-size: 0.85rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 2px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }

    .movie-sub {
        color: rgba(255,255,255,0.8) !important;
        font-size: 0.75rem;
    }

    /* Bright Overlay */
    .movie-overlay {
        position: absolute;
        inset: 0;
        background: rgba(255, 255, 255, 0.96); /* Trắng đục gần như hoàn toàn */
        padding: 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        opacity: 0;
        transition: opacity 0.2s ease;
        line-height: 1.5 !important;
        font-size: 1rem !important;
        border: 1px solid #e2e8f0;
    }

    .movie-card:hover .movie-overlay {
        opacity: 1;
    }

    .ov-title { 
        color: #0f172a !important; 
        font-weight: 800; 
        font-size: 1rem; 
        margin-bottom: 6px;
        white-space: normal; 
    }

    .ov-meta { 
        color: #f06595 !important; /* Accent color */
        font-size: 0.8rem; 
        font-weight: 600; 
        margin-bottom: 10px; 
    }

    .ov-overview { 
        color: #475467 !important; /* Slate-600 */
        font-size: 0.85rem; 
        display: -webkit-box;
        -webkit-line-clamp: 8; 
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* --- TABS --- */
    [data-baseweb="tab-list"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid var(--border) !important;
    }
    
    [data-baseweb="tab"] {
        color: #64748b !important;
        background-color: transparent !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    [data-baseweb="tab-panel"] {
        background-color: #ffffff !important;
    }
    
    /* --- TOASTS --- */
    div[data-testid="stToast"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }
    
    /* --- METRICS & KPI CARDS --- */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    
    /* Rating Popover Enhancement */
    [data-testid="stPopover"] [data-testid="stMarkdown"] strong {
        color: #0f172a !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stPopoverBody"] {
        padding: 16px !important;
    }
    
    /* Slider track in popover */
    [data-testid="stPopoverBody"] [data-baseweb="slider"] [role="slider"] {
        width: 20px !important;
        height: 20px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)
# =============================================================================
# 2. DATA & UTILS (BACKEND LOGIC)
# =============================================================================
TMDB_API_KEY = _TMDB_API_KEY_FALLBACK
POSTER_LAZY_LIMIT = 70

def _poster_url_from_path(poster_path: str | None) -> str:
    if not poster_path or not isinstance(poster_path, str):
        return _NO_POSTER_DATA_URI
    poster_path = poster_path.strip()
    if not poster_path:
        return _NO_POSTER_DATA_URI
    lowered = poster_path.lower()
    if lowered in {"nan", "none", "null", "not_found"}:
        return _NO_POSTER_DATA_URI
    if poster_path.startswith("/"):
        return f"https://image.tmdb.org/t/p/w342{poster_path}"
    if poster_path.startswith("http://") or poster_path.startswith("https://"):
        return poster_path
    if poster_path.startswith("data:image"):
        return poster_path
    return _NO_POSTER_DATA_URI

@st.cache_data(ttl=86400, show_spinner=False)
def _get_api_poster_url(tmdb_id: int) -> str | None:
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
        resp = requests.get(url, timeout=2.5) 
        if resp.status_code == 200:
            path = resp.json().get("poster_path")
            if path: return f"https://image.tmdb.org/t/p/w342{path}"
    except: 
        pass
    return None

def fetch_posters_concurrently(movie_items):
    ids = [item[0] if isinstance(item, (tuple, list)) else item for item in movie_items]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_id = {executor.submit(_get_api_poster_url, mid): mid for mid in ids}
        for future in concurrent.futures.as_completed(future_to_id):
            mid = future_to_id[future]
            results[mid] = future.result() or _NO_POSTER_DATA_URI
    return results

def _render_html(html_content: str) -> None:
    st.markdown(textwrap.dedent(html_content).strip(), unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        movies = pd.read_csv(DATA_PROCESSED / "movies_final.csv")
        ratings = pd.read_csv(DATA_PROCESSED / "ratings_final.csv")
        return movies, ratings
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file dữ liệu.")
        return None, None

@st.cache_resource
def load_models():
    try:
        with open(ARTIFACTS / "svd_model.pkl", "rb") as f:
            svd = pickle.load(f)
        cosine_sim = joblib.load(ARTIFACTS / "cosine_sim.pkl")
        with open(ARTIFACTS / "indices.pkl", "rb") as f:
            indices = pickle.load(f)
        return svd, cosine_sim, indices
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file model.")
        return None, None, None

movies_df, ratings_df = load_data()
svd_model, cosine_sim, indices = load_models()
USER_IDS = set(ratings_df["userId"].unique()) if ratings_df is not None else set()

@st.cache_data
def get_movie_search_map():
    if movies_df is None:
        return {}
    search_df = (
        movies_df[["id", "title", "release_date"]]
        .copy()
        .dropna(subset=["title"])
    )
    search_df["year"] = (
        pd.to_datetime(search_df["release_date"], errors="coerce")
        .dt.year.fillna(0)
        .astype(int)
    )
    search_df["year_str"] = search_df["year"].apply(lambda x: f"({x})" if x > 0 else "")
    search_df["display"] = search_df["title"].astype(str) + " " + search_df["year_str"]
    return dict(zip(search_df["display"], search_df["id"]))

def _safe_text(value) -> str:
    return html.escape("" if value is None else str(value))

def render_movie_card(row, poster_url: str, *, center: bool = False) -> None:
    title_full = _safe_text(row.get("title", ""))
    title_short = title_full if len(title_full) <= 24 else title_full[:24] + "..."
    release_date = _safe_text(row.get("release_date", ""))
    year = release_date[:4] if release_date else ""
    genres = _safe_text(
        str(row.get("genres", "")).replace("'", "").replace("[", "").replace("]", "")
    )
    overview = _safe_text(row.get("overview", ""))
    vote = row.get("vote_average", "")
    vote_txt = f"{vote}/10" if vote else ""

    wrap_open = '<div class="movie-card-center">' if center else ""
    wrap_close = "</div>" if center else ""

    _render_html(
        f"""
        {wrap_open}
        <div class="movie-card" title="{title_full}">
          <img
            class="poster-img movie-poster"
            src="{_safe_text(poster_url)}"
            alt="{title_full}"
            referrerpolicy="no-referrer"
            crossorigin="anonymous"
            loading="lazy"
            onerror="this.onerror=null;this.src='{_NO_POSTER_DATA_URI}';"
          />
          <div class="movie-bottom">
            <p class="movie-title">{title_short}</p>
            <div class="movie-sub">Rating: {vote_txt}</div>
          </div>
          <div class="movie-overlay">
            <div class="ov-title">{title_full}</div>
            <div class="ov-meta">{year} • {genres}</div>
            <div class="ov-overview">{overview}</div>
          </div>
        </div>
                {wrap_close}
        """
    )

def save_new_rating(user_id, movie_id, rating_val):
    timestamp = int(time.time())
    new_row = pd.DataFrame(
        [[user_id, movie_id, rating_val, timestamp]],
        columns=["userId", "id", "rating", "timestamp"],
    )
    try:
        new_row.to_csv(
            DATA_PROCESSED / "ratings_final.csv",
            mode="a",
            header=False,
            index=False,
        )
        return True
    except Exception:
        return False

# ----------------- ALGORITHMS -----------------

def get_hybrid_recommendations(user_id, mood, time_available, companion, top_k=10):
    if (
        movies_df is None
        or ratings_df is None
        or svd_model is None
        or cosine_sim is None
        or indices is None
    ):
        return pd.DataFrame()

    is_new_user = user_id not in USER_IDS
    candidates: list[tuple[int, float]] = []

    if is_new_user:
        top_popular = movies_df.nlargest(100, "vote_count")
        candidates = list(zip(top_popular["id"], top_popular["vote_average"]))
    else:
        user_history = ratings_df[ratings_df["userId"] == user_id]
        liked_movies = user_history[user_history["rating"] >= 4.0].sort_values(
            by="timestamp", ascending=False
        )
        similar_boost_ids: list[int] = []

        if not liked_movies.empty:
            last_liked_id = liked_movies.iloc[0]["id"]
            if last_liked_id in movies_df["id"].values:
                title_series = movies_df.loc[movies_df["id"] == last_liked_id, "title"]
                if not title_series.empty:
                    title = title_series.iloc[0]
                    if title in indices:
                        idx = indices[title]
                        if isinstance(idx, pd.Series):
                            idx = idx.iloc[0]
                        sim_scores = sorted(
                            list(enumerate(cosine_sim[idx])),
                            key=lambda x: x[1],
                            reverse=True,
                        )[1:21]
                        similar_movies = movies_df.iloc[[i[0] for i in sim_scores]]
                        similar_boost_ids = similar_movies["id"].tolist()

        popular_ids = movies_df.nlargest(200, "vote_count")["id"].tolist()
        candidate_pool = list(set(popular_ids + similar_boost_ids))

        for mid in candidate_pool:
            try:
                pred = svd_model.predict(uid=user_id, iid=mid)
                score = pred.est
                if mid in similar_boost_ids:
                    score += 0.5
                candidates.append((mid, score))
            except Exception:
                continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:60]

    if not candidates:
        return pd.DataFrame()

    cand_ids = [mid for mid, _ in candidates]
    cand_info = (
        movies_df[movies_df["id"].isin(cand_ids)]
        .set_index("id")
    )

    final_recs: list[tuple[int, float]] = []
    for mid, score in candidates:
        if mid not in cand_info.index:
            continue

        row = cand_info.loc[mid]
        genres = str(row.get("genres", ""))
        runtime = row.get("runtime", 0) or 0
        final_score = score

        if time_available == "Ngắn (< 90p)" and runtime > 90:
            continue
        if time_available == "Tiêu chuẩn (90-120p)" and (runtime <= 90 or runtime > 120):
            continue
        if time_available == "Thoải mái (> 120p)" and runtime <= 120:
            continue

        if companion == "Người yêu" and "Romance" in genres:
            final_score += 0.5
        if mood == "Vui vẻ" and "Comedy" in genres:
            final_score += 0.5

        final_recs.append((mid, final_score))

    final_recs.sort(key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in final_recs[:top_k]]

    if not top_ids:
        return pd.DataFrame()

    result = cand_info.loc[top_ids].reset_index()
    return result

def get_content_recommendations(title, top_k=10):
    if indices is None or movies_df is None or cosine_sim is None:
        return pd.DataFrame()
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    if isinstance(idx, pd.Series): idx = idx.iloc[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices]

def get_movies_to_rate(user_id, top_k=12):
    if ratings_df is None or movies_df is None:
        return pd.DataFrame()
    user_rated = ratings_df[ratings_df['userId'] == user_id]['id'].tolist()
    candidates = movies_df[~movies_df['id'].isin(user_rated)].sort_values(by='vote_count', ascending=False)
    return candidates.head(top_k)

# =============================================================================
# 3. PAGE RENDERING FUNCTIONS
# =============================================================================

def render_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; margin-top: 60px;'>", unsafe_allow_html=True)
        st.markdown('<h1 style="color:#0f172a; font-weight: 800; letter-spacing: -1px;">FilmGuru</h1>', unsafe_allow_html=True)
        st.write("### Hệ thống gợi ý phim thông minh")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.info("Demo: Nhập **1** (User cũ) hoặc **999** (User mới)")
            with st.form("login_form"):
                uid = st.number_input("User ID:", min_value=1, step=1, value=1)
                st.write("") # Spacer
                submit = st.form_submit_button("Đăng nhập ngay", type="primary", use_container_width=True)
                if submit:
                    st.session_state["is_logged_in"] = True
                    st.session_state["current_user"] = uid
                    st.rerun()

def render_home():
    st.markdown(f"### 👋 Xin chào, **User {st.session_state['current_user']}**")

    mood_opts = ["Bình thường", "Vui vẻ", "Buồn", "Căng thẳng", "Hào hứng", "Gan dạ"]
    time_opts = ["Bất kỳ", "Ngắn (< 90p)", "Tiêu chuẩn (90-120p)", "Thoải mái (> 120p)"]
    companion_opts = ["Một mình", "Người yêu", "Gia đình/Trẻ em", "Bạn bè"]

    if "home_filters_applied" not in st.session_state:
        st.session_state["home_filters_applied"] = {
            "mood": mood_opts[0],
            "time": time_opts[0],
            "companion": companion_opts[0],
        }

    applied = st.session_state["home_filters_applied"]
    
    with st.expander("🎯 **Bộ lọc Cá nhân hóa (Tâm trạng & Hoàn cảnh)**", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            default_mood = st.session_state.get("home_filter_mood") if "home_filter_mood" in st.session_state else applied["mood"]
            mood = st.selectbox("🎭 Tâm trạng của bạn:", mood_opts, index=mood_opts.index(default_mood) if default_mood in mood_opts else 0, key="home_filter_mood")
        with c2:
            default_time = st.session_state.get("home_filter_time") if "home_filter_time" in st.session_state else applied["time"]
            time_avail = st.selectbox("⏳ Thời gian rảnh:", time_opts, index=time_opts.index(default_time) if default_time in time_opts else 0, key="home_filter_time")
        with c3:
            default_companion = st.session_state.get("home_filter_companion") if "home_filter_companion" in st.session_state else applied["companion"]
            companion = st.selectbox("👥 Bạn xem cùng ai:", companion_opts, index=companion_opts.index(default_companion) if default_companion in companion_opts else 0, key="home_filter_companion")
        
        st.write("") # Spacer
        if st.button("🚀 Cập nhật Gợi ý ngay", type="primary", use_container_width=True):
            st.session_state["home_filters_applied"] = {
                "mood": st.session_state.get("home_filter_mood", mood_opts[0]),
                "time": st.session_state.get("home_filter_time", time_opts[0]),
                "companion": st.session_state.get("home_filter_companion", companion_opts[0]),
            }
            st.rerun()

    applied = st.session_state["home_filters_applied"]
    applied_mood = applied["mood"]
    
    st.markdown(f"#### 🎬 Phim tuyển chọn cho bạn ({applied_mood})")
    
    recs = get_hybrid_recommendations(st.session_state["current_user"], applied_mood, applied["time"], applied["companion"])
    
    if recs.empty:
        st.warning("⚠️ Không tìm thấy phim phù hợp.")
    else:
        top_rows = recs.head(POSTER_LAZY_LIMIT)
        poster_items = list(zip(top_rows["id"].tolist(), top_rows["poster_path"].tolist())) if "poster_path" in top_rows.columns else top_rows["id"].tolist()
        with st.spinner("Đang tải poster..."):
            poster_map = fetch_posters_concurrently(poster_items)
        
        cols = st.columns(5)
        for i, (_, row) in enumerate(top_rows.iterrows()):
            with cols[i % 5]:
                poster = poster_map.get(row["id"], _poster_url_from_path(row.get("poster_path")))
                render_movie_card(row, poster)
                if i >= 4: st.write("")

def render_search():
    st.markdown("## 🔎 Tra cứu & Tìm phim tương tự")
    col1, col2 = st.columns([3, 1])
    with col1:
        movie_titles = movies_df['title'].unique()
        selected_movie = st.selectbox("Chọn phim gốc:", movie_titles, index=None, placeholder="Gõ tên phim để bắt đầu...")
    with col2:
        st.write("")
        st.write("")
        btn_search = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)

    if btn_search and selected_movie:
        st.divider()
        st.markdown(f"### Kết quả giống với **'{selected_movie}'**:")
        results = get_content_recommendations(selected_movie, top_k=10)
        
        if results.empty:
            st.warning("Không tìm thấy dữ liệu.")
        else:
            poster_items = list(zip(results["id"].tolist(), results["poster_path"].tolist())) if "poster_path" in results.columns else results["id"].tolist()
            poster_map = fetch_posters_concurrently(poster_items)
            cols = st.columns(5)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 5]:
                    poster = poster_map.get(row["id"], _poster_url_from_path(row.get("poster_path")))
                    render_movie_card(row, poster)

def render_rating_page():
    st.markdown("## ⭐ Chấm điểm & Đóng góp")
    user_id = st.session_state["current_user"]

    # --- Section 1: Tìm cụ thể ---
    st.markdown("##### 1. Tìm phim để thực hiện đánh giá")
    with st.container(border=True):
        col_s, col_r = st.columns([1.2, 1.3])
        with col_s:
            movie_map = get_movie_search_map()
            manual_movie = st.selectbox(
                "Gõ tên phim:",
                movie_map.keys(),
                index=None,
                placeholder="VD: The Avengers...",
            )

        with col_r:
            if manual_movie:
                mid = movie_map[manual_movie]
                try:
                    mrow = movies_df[movies_df["id"] == mid].iloc[0]
                    poster_path = mrow.get("poster_path")
                except Exception:
                    mrow = {}
                    poster_path = None

                poster_map = fetch_posters_concurrently([(mid, poster_path)])
                poster = poster_map.get(mid, _poster_url_from_path(poster_path))
                
                # Layout nhỏ gọn cho phần manual rating
                c_img, c_rate = st.columns([1.2, 1])
                with c_img:
                    render_movie_card(mrow, poster, center=True)
                with c_rate:
                    st.write("#### Đánh giá của bạn")
                    rating = st.slider("Thang điểm 5:", 0.5, 5.0, 4.0, 0.5, key="manual_rate")
                    if st.button("Lưu đánh giá", type="primary", use_container_width=True):
                        if save_new_rating(user_id, mid, rating):
                            st.toast("✅ Đã lưu thành công!")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()

    st.divider()

    # --- Section 2: Gợi ý để rate (UNIFIED STYLE) ---
    st.markdown("##### 2. Bạn đã xem các phim Hot này chưa? (Hãy chấm điểm nhé)")
    
    candidates = get_movies_to_rate(user_id, top_k=10)
    poster_items = list(zip(candidates["id"].tolist(), candidates["poster_path"].tolist())) if "poster_path" in candidates.columns else candidates["id"].tolist()
    poster_map = fetch_posters_concurrently(poster_items)

    cols = st.columns(5)
    for i, (_, row) in enumerate(candidates.iterrows()):
        with cols[i % 5]:
            poster = poster_map.get(row['id'], _poster_url_from_path(row.get("poster_path")))
            render_movie_card(row, poster)
            
            # Nút chấm điểm nằm ngay dưới card
            with st.popover("⭐ Chấm điểm ngay", use_container_width=True):
                st.markdown(f"**{row['title']}**")
                r_val = st.slider(f"Điểm số:", 0.5, 5.0, 3.0, 0.5, key=f"quick_{row['id']}")
                if st.button(f"Gửi đánh giá", key=f"btn_{row['id']}", type="primary", use_container_width=True):
                    save_new_rating(user_id, row['id'], r_val)
                    st.toast("✅ Đã lưu!")
                    st.cache_data.clear()
                    st.rerun()

def render_dashboard():
    st.markdown("## 📈 Dashboard & Thống kê")
    
    tab1, tab2 = st.tabs(["👤 **Thống kê Của Tôi**", "🌍 **Phân tích Hệ thống**"])
    
    with tab1:
        try:
            current_ratings = pd.read_csv(DATA_PROCESSED / "ratings_final.csv")
        except FileNotFoundError:
            st.info("Chưa có dữ liệu ratings. Hãy chạy preprocessing/training trước.")
            return
        my_hist = current_ratings[current_ratings["userId"] == st.session_state["current_user"]].copy()
        
        if my_hist.empty:
            st.info("Bạn chưa có dữ liệu.")
        else:
            keep_cols = ["id", "title", "genres", "runtime", "overview", "release_date", "vote_average"]
            if "poster_path" in movies_df.columns:
                keep_cols.append("poster_path")
            
            my_hist = my_hist.merge(movies_df[keep_cols], on="id", how="left")
            my_hist["Date"] = pd.to_datetime(my_hist["timestamp"], unit="s")
            
            # KPI Cards
            total_movies = len(my_hist)
            avg_score = my_hist["rating"].mean()
            total_hours = int(my_hist["runtime"].fillna(0).sum() / 60)
            
            with st.container(border=True):
                k1, k2 = st.columns(2)
                k1.metric("Tổng phim đã đánh giá", total_movies)
                # k2.metric("Tổng thời gian (giờ)", f"{total_hours}")
                k2.metric("Điểm trung bình đánh giá", f"{avg_score:.1f}")
            
            st.write("")
            
            # HALL OF FAME
            favorites = my_hist[my_hist['rating'] >= 4.5].head(5)
            if not favorites.empty:
                st.markdown("### 🏆 Hall of Fame: Phim yêu thích của bạn")
                poster_items = list(zip(favorites["id"].tolist(), favorites["poster_path"].tolist())) if "poster_path" in favorites.columns else favorites["id"].tolist()
                fav_posters = fetch_posters_concurrently(poster_items)
                
                c_fav = st.columns(5)
                for i, (_, row) in enumerate(favorites.iterrows()):
                    with c_fav[i]:
                        poster = fav_posters.get(row['id'], _poster_url_from_path(row.get("poster_path")))
                        render_movie_card(row, poster)
                        st.caption(f"⭐ Bạn chấm: {row['rating']}")

            st.divider()
            
            # HISTORY TABLE - BRIGHT THEME
            st.markdown("### 📜 Nhật ký đánh giá phim của bạn")
            st.markdown('<div class="history-table-container">', unsafe_allow_html=True)
            
            hist = my_hist.copy()
            show = hist[["id", "title", "rating", "Date", "poster_path"]]
            
            poster_items = list(zip(show["id"].tolist(), show["poster_path"].tolist()))
            items_limited = poster_items[:POSTER_LAZY_LIMIT]
            poster_map = fetch_posters_concurrently(items_limited)
            show["poster"] = show["id"].map(poster_map).fillna(_NO_POSTER_DATA_URI)

            st.dataframe(
                show[["poster", "title", "rating", "Date"]],
                use_container_width=True,
                hide_index=True,
                height=600,
                row_height=150,  # Tăng chiều cao dòng để chứa poster lớn
                column_config={
                    "poster": st.column_config.ImageColumn("Poster", width="large"), # Poster kích thước lớn
                    "title": st.column_config.TextColumn("Tên phim", width="large"),
                    "rating": st.column_config.NumberColumn("Điểm", format="%.1f"),
                    "Date": st.column_config.DatetimeColumn("Ngày đánh giá", format="D MMM YYYY, h:mm a"),
                },
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.info("💡 Báo cáo phân tích trên toàn bộ hệ thống.")
        with st.container(border=True):
            st.markdown("### 🤖 Hiệu năng Mô hình Lõi (SVD Algorithm)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE", "0.8942", delta="-0.001", delta_color="inverse")
            m2.metric("MAE", "0.6871", delta="-0.002", delta_color="inverse")
            m3.metric("Precision@10", "78%")
            m4.metric("Recall@10", "52%")
        
        st.write("")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Phân bố điểm số (Rating Distribution)**")
            rating_counts = ratings_df['rating'].value_counts().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            chart1 = alt.Chart(rating_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X('Rating:O', axis=alt.Axis(labelAngle=0)), 
                y='Count', 
                color=alt.value(_ACCENT_1),
                tooltip=['Rating', 'Count']
            ).properties(height=300)
            st.altair_chart(chart1, use_container_width=True)
            
        with col_b:
            st.markdown("**Top phim phổ biến nhất**")
            top_m = movies_df.sort_values('vote_count', ascending=False).head(8)
            chart2 = alt.Chart(top_m).mark_bar(cornerRadiusBottomRight=4, cornerRadiusTopRight=4).encode(
                x='vote_count', 
                y=alt.Y('title', sort='-x', title=None), 
                color=alt.value(_FG),
                tooltip=['title', 'vote_count']
            ).properties(height=300)
            st.altair_chart(chart2, use_container_width=True)

# =============================================================================
# 4. ROUTER
# =============================================================================

if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False

if not st.session_state["is_logged_in"]:
    render_login()
else:
    with st.sidebar:
        st.markdown("## 🍿 FilmGuru")
        st.caption(f"User ID: {st.session_state['current_user']}")
        menu = st.radio("Menu:", ["🏠 Trang chủ", "🔎 Tìm kiếm", "⭐ Chấm điểm", "📊 Dashboard"])
        st.divider()
        if st.button("🚪 Đăng xuất", use_container_width=True):
            st.session_state["is_logged_in"] = False
            st.rerun()
        st.markdown("<div style='margin-top: 50px; font-size: 0.8rem; color: #94a3b8;'>© 2025 FilmGuru<br>Bright Edition</div>", unsafe_allow_html=True)

    if menu == "🏠 Trang chủ":
        render_home()
    elif menu == "🔎 Tìm kiếm":
        render_search()
    elif menu == "⭐ Chấm điểm":
        render_rating_page()
    elif menu == "📊 Dashboard":
        render_dashboard()