from __future__ import annotations

import pickle
from pathlib import Path

import joblib  # Dùng để lưu file nén siêu nhẹ
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Minimal path helper (no external package)
ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"


def main():
    processed_dir = DATA_PROCESSED
    artifacts_dir = ARTIFACTS

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 ĐANG KHỞI TẠO MODULE CONTENT-BASED FILTERING (OPTIMIZED)...")

    # 1. LOAD DỮ LIỆU
    try:
        movies = pd.read_csv(processed_dir / "movies_final.csv")
        print(f"✅ Đã tải gốc: {movies.shape[0]} dòng.")

        print("⏳ Đang sắp xếp theo độ phổ biến...")
        movies = movies.sort_values(by="vote_count", ascending=False)

        # Ở đây mình để 35,000 phim là con số "Vàng" (An toàn cho mọi máy laptop)
        movies = movies.head(35000)

        movies = movies.reset_index(drop=True)
        print(f"✅ Dữ liệu đưa vào Model: {movies.shape[0]} dòng.")

        # Lưu lại để đồng bộ index
        movies.to_csv(processed_dir / "movies_final.csv", index=False)
        print("💾 Đã cập nhật file 'movies_final.csv' để đồng bộ Index.")

    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file 'movies_final.csv'.")
        raise

    # ======================================================
    # TASK 1: VECTOR HÓA DỮ LIỆU (TF-IDF) + ÉP KIỂU FLOAT32
    # ======================================================
    print("⏳ Đang tính toán TF-IDF và ép kiểu float32 (Giảm 50% RAM)...")

    tf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, stop_words="english"
    )

    tfidf_matrix = tf.fit_transform(movies["soup"].fillna("")).astype(np.float32)
    print(f"✅ Kích thước ma trận TF-IDF: {tfidf_matrix.shape}")
    print(f"✅ Kiểu dữ liệu: {tfidf_matrix.dtype}")

    # ======================================================
    # TASK 2: TÍNH ĐỘ TƯƠNG ĐỒNG (COSINE SIMILARITY)
    # ======================================================
    print("⏳ Đang tính ma trận Cosine (Siêu tốc)...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print("✅ Đã tính xong ma trận Cosine!")

    # ======================================================
    # TASK 3: TẠO MAPPING INDEX
    # ======================================================
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    # ======================================================
    # TASK 4: LƯU MODEL
    # ======================================================
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print("⏳ Đang lưu file model...")

    pickle.dump(indices, open(artifacts_dir / "indices.pkl", "wb"))
    joblib.dump(cosine_sim, artifacts_dir / "cosine_sim.pkl", compress=0)

    print(f"\n💾 HOÀN TẤT! Model đã lưu tại '{artifacts_dir}'")
    print("⚠️ LƯU Ý: Vì dùng Joblib, hãy đảm bảo app.py dùng joblib.load() để đọc file này.")


if __name__ == "__main__":
    main()
