from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


# Minimal path helper (no external package)
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def parse_json_all(x):
    try:
        data = ast.literal_eval(x)
        if isinstance(data, list):
            return [i["name"] for i in data]
        return []
    except Exception:
        return []


def get_top_3_cast(x):
    try:
        data = ast.literal_eval(x)
        if isinstance(data, list):
            names = [i["name"] for i in data]
            return names[:3]
        return []
    except Exception:
        return []


def get_director(x):
    try:
        data = ast.literal_eval(x)
        if isinstance(data, list):
            for i in data:
                if i.get("job") == "Director":
                    return i.get("name")
        return np.nan
    except Exception:
        return np.nan


def clean_data(x):
    if isinstance(x, list):
        return [str(i).lower().replace(" ", "") for i in x]
    if isinstance(x, str):
        return x.lower().replace(" ", "")
    return ""


def create_soup(row):
    return (
        " ".join(row["keywords"])
        + " "
        + " ".join(row["cast"])
        + " "
        + str(row["director"])
        + " "
        + " ".join(row["genres"])
    )


def main():
    raw_dir = DATA_RAW
    processed_dir = DATA_PROCESSED

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 BẮT ĐẦU QUÁ TRÌNH LÀM SẠCH DỮ LIỆU...")

    # =============================================================================
    # BƯỚC 1: LOAD DATA
    # =============================================================================
    print("1️⃣  Đang đọc các file CSV...")
    try:
        meta = pd.read_csv(raw_dir / "movies_metadata.csv", low_memory=False)
        credits = pd.read_csv(raw_dir / "credits.csv")
        keywords = pd.read_csv(raw_dir / "keywords.csv")
        links = pd.read_csv(raw_dir / "links.csv")
        ratings = pd.read_csv(raw_dir / "ratings_small.csv")
    except FileNotFoundError as e:
        print(f"❌ Lỗi: Không tìm thấy file. Hãy kiểm tra lại đường dẫn: {e}")
        raise

    # =============================================================================
    # BƯỚC 2: MERGE
    # =============================================================================
    print("2️⃣  Đang xử lý ID và ghép bảng...")
    meta = meta[pd.to_numeric(meta["id"], errors="coerce").notnull()]
    meta["id"] = meta["id"].astype(int)
    keywords["id"] = keywords["id"].astype(int)
    credits["id"] = credits["id"].astype(int)

    meta = meta.merge(credits, on="id")
    meta = meta.merge(keywords, on="id")

    # =============================================================================
    # BƯỚC 3: FEATURE EXTRACTION
    # =============================================================================
    print("3️⃣  Đang trích xuất thông tin từ JSON (Genres, Cast, Keywords)...")
    meta["genres"] = meta["genres"].apply(parse_json_all)
    meta["keywords"] = meta["keywords"].apply(parse_json_all)
    meta["cast"] = meta["cast"].apply(get_top_3_cast)
    meta["director"] = meta["crew"].apply(get_director)

    # =============================================================================
    # BƯỚC 4: TEXT NORMALIZATION + SOUP
    # =============================================================================
    print("4️⃣  Đang chuẩn hóa văn bản (Lowercasing & Cleaning)...")
    for feature in ["cast", "keywords", "genres"]:
        meta[feature] = meta[feature].apply(clean_data)
    meta["director"] = meta["director"].apply(clean_data)
    meta["soup"] = meta.apply(create_soup, axis=1)

    # =============================================================================
    # BƯỚC 5: BUILD OUTPUTS + DEDUPE
    # =============================================================================
    print("5️⃣  Đang tạo file kết quả và làm sạch lần cuối...")
    cols_movies = [
        "id",
        "title",
        "genres",
        "overview",
        "cast",
        "director",
        "keywords",
        "soup",
        "vote_average",
        "vote_count",
        "release_date",
        "poster_path",
        "runtime",
        "popularity",
    ]

    meta["popularity"] = pd.to_numeric(meta["popularity"], errors="coerce")
    movies_final = meta[cols_movies].copy().dropna(subset=["title"])

    links = links[links["tmdbId"].notnull()]
    links["tmdbId"] = links["tmdbId"].astype(int)
    links["movieId"] = links["movieId"].astype(int)

    ratings_mapped = ratings.merge(links, on="movieId")
    ratings_final = ratings_mapped[["userId", "tmdbId", "rating", "timestamp"]]
    ratings_final.rename(columns={"tmdbId": "id"}, inplace=True)

    valid_ids = movies_final["id"].unique()
    ratings_final = ratings_final[ratings_final["id"].isin(valid_ids)]

    print("🧹 Đang kiểm tra và loại bỏ dữ liệu trùng lặp (Duplicate removal)...")
    movies_final.drop_duplicates(subset=["id"], inplace=True)
    ratings_final.drop_duplicates(inplace=True)

    movies_final.to_csv(processed_dir / "movies_final.csv", index=False)
    ratings_final.to_csv(processed_dir / "ratings_final.csv", index=False)

    print(f"✅ HOÀN TẤT! Đã lưu 2 file sạch tại: {processed_dir}")
    print(f"   - movies_final.csv: {movies_final.shape[0]} phim")
    print(f"   - ratings_final.csv: {ratings_final.shape[0]} lượt đánh giá")


if __name__ == "__main__":
    main()
