#!/usr/bin/env python3
"""Demo: nhập thông tin học tập giả định -> mức nguy cơ + khuyến nghị hỗ trợ. Chạy: python src/risk_demo.py"""
import joblib
import numpy as np
import pandas as pd

S1 = "Curricular units 1st sem"
DISCLAIMER = ("Đây là nguyên mẫu phân tích dữ liệu, KHÔNG phải quyết định hành chính. "
              "Kết quả chỉ gợi ý nhóm cần được quan tâm; cần cố vấn học tập xem xét thêm.")
NAMES = {f"{S1} (approved)": "số môn đạt HK1", f"{S1} (grade)": "điểm TB HK1", f"{S1} (enrolled)": "số môn đăng ký HK1",
         f"{S1} (evaluations)": "số lần đánh giá HK1", f"{S1} (without evaluations)": "số môn chưa có đánh giá HK1",
         "Tuition fees up to date": "học phí đúng hạn", "Debtor": "nợ học phí", "Scholarship holder": "học bổng",
         "Age at enrollment": "tuổi nhập học", "Admission grade": "điểm đầu vào", "Course": "ngành học",
         "Application mode": "diện tuyển sinh", "Gender": "giới tính", "Previous qualification (grade)": "điểm bằng trước",
         f"{S1} (credited)": "số môn được công nhận"}


def load_bundle(path="final_model.joblib"):
    return joblib.load(path)


def predict_risk(bundle, inputs):
    """inputs: dict các trường người dùng nhập (trường thiếu lấy giá trị điển hình của tập huấn luyện)."""
    row = dict(bundle["defaults"]); row.update(inputs)
    X = pd.DataFrame([row])[bundle["raw_columns"]]
    model, t = bundle["model"], bundle["threshold"]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    pdrop = proba["Dropout"]
    level = "CAO" if pdrop >= t else ("TRUNG BÌNH" if pdrop >= t / 2 else "THẤP")
    # giải thích: thay từng trường đã nhập bằng giá trị điển hình, xem P(Dropout) thay đổi bao nhiêu
    contrib = []
    for k in inputs:
        alt = dict(row); alt[k] = bundle["defaults"][k]
        pa = model.predict_proba(pd.DataFrame([alt])[bundle["raw_columns"]])[0][list(model.classes_).index("Dropout")]
        contrib.append((k, pdrop - pa))
    contrib = sorted(contrib, key=lambda z: -abs(z[1]))[:3]
    top2 = sorted(proba.values(), reverse=True)[:2]
    uncertain = level == "TRUNG BÌNH" or (top2[0] - top2[1]) < 0.15
    rec = {"CAO": "Mời sinh viên trao đổi với cố vấn học tập sớm; rà soát tải môn học, tài chính, nhu cầu hỗ trợ học thuật.",
           "TRUNG BÌNH": "Theo dõi thêm kết quả đầu HK2; gửi thông tin hỗ trợ (phụ đạo, tư vấn) và hỏi nhu cầu của sinh viên.",
           "THẤP": "Duy trì theo dõi thông thường; thông tin các kênh hỗ trợ khi cần."}[level]
    return {"proba": proba, "p_dropout": pdrop, "level": level, "threshold": t, "recommendation": rec,
            "drivers": [(NAMES.get(k, k), float(v)) for k, v in contrib], "uncertain": uncertain}


def show(title, desc, res):
    print(f"\n=== {title} ===\n{desc}")
    print("  Xác suất:", ", ".join(f"{k}={v:.2f}" for k, v in res["proba"].items()))
    print(f"  Mức nguy cơ: {res['level']} (cảnh báo CAO khi P(Dropout) >= {res['threshold']:.2f})")
    print("  Yếu tố ảnh hưởng nhiều nhất (so với SV điển hình):",
          "; ".join(f"{n} ({d:+.2f})" for n, d in res["drivers"]))
    print("  Khuyến nghị hỗ trợ:", res["recommendation"])
    if res["uncertain"]:
        print("  ⚠ Không nên kết luận chắc chắn: dự đoán nằm gần ranh giới / các lớp xác suất sát nhau.")


SCENARIOS = [
    ("Tình huống 1 - Ổn định", "Học đủ 6 môn, đạt 6/6, điểm 14.5, học phí đúng hạn, có học bổng.",
     {"Course": 9500, "Application mode": 1, "Admission grade": 150, "Age at enrollment": 18, "Scholarship holder": 1,
      "Tuition fees up to date": 1, "Debtor": 0, f"{S1} (enrolled)": 6, f"{S1} (evaluations)": 6,
      f"{S1} (approved)": 6, f"{S1} (grade)": 14.5, f"{S1} (without evaluations)": 0}),
    ("Tình huống 2 - Nằm ở ranh giới", "Đăng ký 6 môn, đạt 3, điểm 11, học phí đúng hạn, đi làm thêm (tuổi 24).",
     {"Course": 9119, "Application mode": 17, "Admission grade": 125, "Age at enrollment": 24, "Scholarship holder": 0,
      "Tuition fees up to date": 1, "Debtor": 0, f"{S1} (enrolled)": 6, f"{S1} (evaluations)": 7,
      f"{S1} (approved)": 3, f"{S1} (grade)": 11, f"{S1} (without evaluations)": 1}),
    ("Tình huống 3 - Nguy cơ cao", "Đăng ký 6 môn, không đạt môn nào, 3 môn chưa đánh giá, nợ học phí, tuổi 30.",
     {"Course": 9991, "Application mode": 39, "Admission grade": 110, "Age at enrollment": 30, "Scholarship holder": 0,
      "Tuition fees up to date": 0, "Debtor": 1, f"{S1} (enrolled)": 6, f"{S1} (evaluations)": 3,
      f"{S1} (approved)": 0, f"{S1} (grade)": 0, f"{S1} (without evaluations)": 3}),
]

if __name__ == "__main__":
    b = load_bundle()
    print(DISCLAIMER)
    for t, d, i in SCENARIOS:
        show(t, d, predict_risk(b, i))
