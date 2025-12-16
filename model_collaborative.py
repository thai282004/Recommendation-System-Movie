from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


# Minimal path helper (no external package)
ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user."""

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / k if k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    return precisions, recalls


def main():
    processed_dir = DATA_PROCESSED
    artifacts_dir = ARTIFACTS

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 ĐANG KHỞI TẠO MODULE COLLABORATIVE FILTERING (SVD)...")

    # 1. LOAD DỮ LIỆU
    try:
        ratings = pd.read_csv(processed_dir / "ratings_final.csv")
        print(f"✅ Đã tải: {ratings.shape[0]} dòng ratings.")
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file 'ratings_final.csv'.")
        raise

    # 2. CHUẨN BỊ DỮ LIỆU CHO SURPRISE
    print("⏳ Đang chuyển đổi dữ liệu...")
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[["userId", "id", "rating"]], reader)

    # 3. CHIA TẬP TRAIN/TEST
    trainset, testset = train_test_split(data, test_size=0.2)

    # 4. HUẤN LUYỆN MODEL SVD
    print("⏳ Đang huấn luyện thuật toán SVD (Matrix Factorization)...")
    svd = SVD()
    svd.fit(trainset)

    # 5. ĐÁNH GIÁ
    print("⏳ Đang tính toán các chỉ số đánh giá (RMSE, MAE, Precision@K, Recall@K)...")
    predictions = svd.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3.5)
    avg_precision = sum(precisions.values()) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls.values()) / len(recalls) if recalls else 0.0

    print("\n📊 BẢNG ĐÁNH GIÁ HIỆU NĂNG MODEL (Ghi vào báo cáo):")
    print(f"   - RMSE (Sai số bình phương): {rmse:.4f}")
    print(f"   - MAE (Sai số tuyệt đối):    {mae:.4f}")
    print(f"   - Precision@10:              {avg_precision:.4f}")
    print(f"   - Recall@10:                 {avg_recall:.4f}")
    print("   *(Precision thấp là bình thường với dataset thưa)*")

    # 6. RETRAIN FULL + SAVE
    print("\n⏳ Đang Retrain trên toàn bộ 100% dữ liệu để lưu Model...")
    full_trainset = data.build_full_trainset()
    svd.fit(full_trainset)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_file = artifacts_dir / "svd_model.pkl"
    pickle.dump(svd, open(model_file, "wb"))
    print(f"✅ HOÀN TẤT! Đã lưu model SVD vào: '{model_file}'")


if __name__ == "__main__":
    main()
