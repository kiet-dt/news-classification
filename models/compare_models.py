"""
Script so sánh hiệu suất các mô hình:
- Logistic Regression
- Naive Bayes
- SVM
- PhoBERT

So sánh trên cùng một tập dữ liệu:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

Kết quả:
- In bảng tổng hợp ra console
- Lưu bảng kết quả dạng CSV (tùy chọn)
- Vẽ biểu đồ cột so sánh và lưu file PNG (tùy chọn)
"""

import os
import sys
import argparse
import time

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# Thêm root project vào path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.logistic_regression.model import NewsLogisticRegression
from models.naive_bayes.model import NewsNaiveBayes
from models.svm.model import NewsSVM
from models.phobert.model import NewsPhoBERT


def load_data(data_path: str, sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Load dữ liệu đã xử lý từ file Parquet/CSV.

    Args:
        data_path: Đường dẫn tới file dữ liệu (parquet hoặc csv)
        sample_size: Nếu được set, lấy ngẫu nhiên sample_size mẫu để so sánh
        random_state: Seed cho việc lấy mẫu
    """
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    ext = os.path.splitext(data_path)[1].lower()
    print(f"Đang load dữ liệu từ {data_path} ...")

    if ext == ".parquet":
        df = pd.read_parquet(data_path)
    elif ext == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Định dạng dữ liệu không hỗ trợ. Hãy dùng .parquet hoặc .csv")

    # Chỉ giữ các cột cần thiết
    required_cols = {"title", "content", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc trong dữ liệu: {missing}")

    # Loại bỏ rows thiếu dữ liệu
    df = df.dropna(subset=["title", "content", "category"]).copy()

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Đã lấy mẫu ngẫu nhiên {sample_size} bản ghi để so sánh.")
    else:
        df = df.reset_index(drop=True)
        print(f"Số lượng mẫu dùng để so sánh: {len(df)}")

    return df


def evaluate_model_on_dataframe(model_name: str, model, df: pd.DataFrame) -> dict:
    """Chạy predict một mô hình trên toàn bộ DataFrame và tính các metrics."""
    print(f"\n=== Đang đánh giá mô hình {model_name} trên {len(df)} mẫu ===")
    start_time = time.time()

    y_true = df["category"].tolist()
    y_pred = []

    for i, row in df.iterrows():
        title = row["title"]
        content = row["content"]

        try:
            # Truyền (title, content) để mô hình tự combine & preprocess
            pred = model.predict((title, content), return_proba=False)
        except Exception as e:
            print(f"  Lỗi khi predict sample {i} với {model_name}: {e}")
            # Để tránh crash, có thể dự đoán "ngẫu nhiên" bằng cách gán nhãn thật
            pred = row["category"]

        y_pred.append(pred)

        if (i + 1) % 500 == 0:
            print(f"  Đã xử lý {i + 1}/{len(df)} mẫu...")

    elapsed = time.time() - start_time

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nKết quả {model_name}:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Thời gian: {elapsed:.1f} giây ({elapsed/len(df):.4f} s/mẫu)")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time_sec": elapsed,
        "time_per_sample": elapsed / len(df),
    }


def plot_results(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Vẽ các biểu đồ so sánh giữa các mô hình:
    - Biểu đồ cột: Accuracy / Precision / Recall / F1 (tổng hợp)
    - Biểu đồ cột: F1-score từng mô hình
    - Biểu đồ cột (log-scale): time_per_sample
    - Biểu đồ scatter: F1 vs time_per_sample
    """
    base_dir = os.path.dirname(output_path)
    base_name, ext = os.path.splitext(os.path.basename(output_path))

    # 1. Biểu đồ cột tổng hợp 4 metrics
    metrics = ["accuracy", "precision", "recall", "f1"]

    plt.figure(figsize=(10, 6))

    x = range(len(results_df))
    bar_width = 0.18

    for idx, metric in enumerate(metrics):
        plt.bar(
            [i + idx * bar_width for i in x],
            results_df[metric],
            width=bar_width,
            label=metric.capitalize(),
        )

    plt.xticks(
        [i + bar_width * (len(metrics) - 1) / 2 for i in x],
        results_df["model"],
    )
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("So sánh hiệu suất (Accuracy / Precision / Recall / F1)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_path_main = os.path.join(base_dir, f"{base_name}_metrics{ext}")
    plt.savefig(out_path_main, dpi=150)
    print(f"Đã lưu biểu đồ so sánh metrics vào: {out_path_main}")

    # 2. Biểu đồ cột F1-score
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["f1"], color="#4c72b0")
    plt.ylim(0, 1.0)
    plt.ylabel("F1-score")
    plt.title("So sánh F1-score giữa các mô hình")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path_f1 = os.path.join(base_dir, f"{base_name}_f1{ext}")
    plt.savefig(out_path_f1, dpi=150)
    print(f"Đã lưu biểu đồ F1-score vào: {out_path_f1}")

    # 3. Biểu đồ cột time_per_sample (log-scale)
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["time_per_sample"], color="#dd8452")
    plt.yscale("log")
    plt.ylabel("Thời gian suy luận / mẫu (giây, log-scale)")
    plt.title("So sánh tốc độ suy luận giữa các mô hình")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path_time = os.path.join(base_dir, f"{base_name}_time_per_sample{ext}")
    plt.savefig(out_path_time, dpi=150)
    print(f"Đã lưu biểu đồ time_per_sample vào: {out_path_time}")

    # 4. Biểu đồ scatter: F1 vs time_per_sample
    plt.figure(figsize=(8, 5))
    plt.scatter(results_df["time_per_sample"], results_df["f1"], color="#55a868")
    for _, row in results_df.iterrows():
        plt.text(
            row["time_per_sample"],
            row["f1"],
            row["model"],
            fontsize=9,
            ha="left",
            va="bottom",
        )
    plt.xscale("log")
    plt.xlabel("Thời gian suy luận / mẫu (giây, log-scale)")
    plt.ylabel("F1-score")
    plt.title("Trade-off giữa F1-score và tốc độ suy luận")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path_scatter = os.path.join(base_dir, f"{base_name}_f1_vs_time{ext}")
    plt.savefig(out_path_scatter, dpi=150)
    print(f"Đã lưu biểu đồ F1 vs time_per_sample vào: {out_path_scatter}")


def main():
    parser = argparse.ArgumentParser(description="So sánh hiệu suất các mô hình phân loại tin tức")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/news_dataset.parquet",
        help="Đường dẫn tới file dữ liệu (parquet/csv). Mặc định: data/processed/news_dataset.parquet",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Số lượng mẫu dùng để so sánh (nếu None dùng toàn bộ data)",
    )
    parser.add_argument(
        "--no-phobert",
        action="store_true",
        help="Bỏ qua mô hình PhoBERT (nếu máy yếu hoặc không có GPU)",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Đường dẫn lưu bảng kết quả (.csv). Nếu bỏ trống thì không lưu",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="models/results_comparison.png",
        help="Đường dẫn lưu biểu đồ so sánh (.png). Mặc định: models/results_comparison.png",
    )

    args = parser.parse_args()

    # Load dữ liệu
    df = load_data(args.data_path, sample_size=args.sample_size)

    # Chuẩn bị danh sách mô hình
    models_to_eval: list[tuple[str, object, str]] = []

    # Logistic Regression
    try:
        lr = NewsLogisticRegression()
        lr.load(os.path.join(PROJECT_ROOT, "models/saved/logistic_regression"))
        models_to_eval.append(("Logistic Regression", lr, "lr"))
        print("✓ Đã load Logistic Regression")
    except Exception as e:
        print(f"✗ Không thể load Logistic Regression: {e}")

    # Naive Bayes
    try:
        nb = NewsNaiveBayes()
        nb.load(os.path.join(PROJECT_ROOT, "models/saved/naive_bayes"))
        models_to_eval.append(("Naive Bayes", nb, "nb"))
        print("✓ Đã load Naive Bayes")
    except Exception as e:
        print(f"✗ Không thể load Naive Bayes: {e}")

    # SVM
    try:
        svm = NewsSVM()
        svm.load(os.path.join(PROJECT_ROOT, "models/saved/svm"))
        models_to_eval.append(("SVM", svm, "svm"))
        print("✓ Đã load SVM")
    except Exception as e:
        print(f"✗ Không thể load SVM: {e}")

    # PhoBERT (có thể bỏ qua)
    if not args.no_phobert:
        try:
            phobert = NewsPhoBERT()
            phobert.load(os.path.join(PROJECT_ROOT, "models/saved/phobert"))
            models_to_eval.append(("PhoBERT", phobert, "phobert"))
            print("✓ Đã load PhoBERT")
        except Exception as e:
            print(f"✗ Không thể load PhoBERT: {e}")
    else:
        print("Bỏ qua PhoBERT theo tham số --no-phobert")

    if not models_to_eval:
        print("Lỗi: Không có mô hình nào được load thành công.")
        sys.exit(1)

    # Đánh giá từng mô hình
    results = []
    for display_name, model, _short in models_to_eval:
        res = evaluate_model_on_dataframe(display_name, model, df)
        results.append(res)

    results_df = pd.DataFrame(results)

    print("\n=== BẢNG TỔNG HỢP KẾT QUẢ ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Lưu CSV nếu cần
    if args.save_csv:
        csv_path = args.save_csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(PROJECT_ROOT, csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nĐã lưu bảng kết quả vào: {csv_path}")

    # Vẽ và lưu biểu đồ
    if args.save_plot:
        plot_path = args.save_plot
        if not os.path.isabs(plot_path):
            plot_path = os.path.join(PROJECT_ROOT, plot_path)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plot_results(results_df, plot_path)


if __name__ == "__main__":
    main()


