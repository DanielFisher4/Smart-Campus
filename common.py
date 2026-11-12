"""Mã dùng chung: cấu hình đề AI01, nạp dữ liệu, quy tắc chống rò rỉ thông tin, đặc trưng, mô hình, chỉ số."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
CLASSES = ["Dropout", "Enrolled", "Graduate"]
TARGET = "Target"

# ---------------------------------------------------------------- tên cột chuẩn (UCI id=697)
S1 = "Curricular units 1st sem"
S2 = "Curricular units 2nd sem"
CANON = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
    "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Admission grade", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International",
] + [f"{s} ({k})" for s in (S1, S2) for k in
     ("credited", "enrolled", "evaluations", "approved", "grade", "without evaluations")] + [
    "Unemployment rate", "Inflation rate", "GDP", TARGET]
_LOWER = {c.lower(): c for c in CANON}

# Biến định danh (mã số không có thứ tự) -> one-hot
NOMINAL = ["Marital status", "Application mode", "Course", "Previous qualification", "Nacionality",
           "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
MACRO = ["Unemployment rate", "Inflation rate", "GDP"]  # chỉ số vĩ mô: không phải tuyển sinh/HK1 -> loại cho an toàn

# ---------------------------------------------------------------- BASELINE (9 đặc trưng)
# !!! THAY bằng danh sách 9 đặc trưng + 2 biến danh mục mà BTC công bố nếu khác. !!!
BASE_NUM = ["Admission grade", "Previous qualification (grade)", "Age at enrollment",
            "Tuition fees up to date", "Scholarship holder", f"{S1} (enrolled)",
            f"{S1} (approved)"]
BASE_CAT = ["Application mode", "Course"]

FE_COLS = ["s1_approval_rate", "s1_eval_rate", "s1_pass_per_eval", "s1_no_eval_rate",
           "s1_credited_rate", "fees_or_debt_flag"]


def load_data(csv_path=None):
    """Đọc CSV (dấu ; hoặc ,) hoặc tải từ UCI qua ucimlrepo."""
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=697)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = [_LOWER.get(re.sub(r"\s+", " ", c).strip().lower(), re.sub(r"\s+", " ", c).strip())
                  for c in df.columns]
    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df


def allowed_features(df):
    """Chỉ thông tin tuyển sinh + kết quả học kỳ 1. Cấm mọi cột học kỳ 2 (và vĩ mô)."""
    return [c for c in df.columns if c != TARGET and "2nd sem" not in c and c not in MACRO]


def add_features(df):
    """Đặc trưng kỹ thuật từ kết quả HK1 (chỉ dùng dữ liệu HK1/tuyển sinh)."""
    d = df.copy()
    enr = d[f"{S1} (enrolled)"].astype(float).replace(0, np.nan)
    ev = d[f"{S1} (evaluations)"].astype(float).replace(0, np.nan)
    ap = d[f"{S1} (approved)"].astype(float)
    d["s1_approval_rate"] = (ap / enr).fillna(0).clip(0, 2)
    d["s1_eval_rate"] = (d[f"{S1} (evaluations)"] / enr).fillna(0).clip(0, 5)
    d["s1_pass_per_eval"] = (ap / ev).fillna(0).clip(0, 3)
    d["s1_no_eval_rate"] = (d[f"{S1} (without evaluations)"] / enr).fillna(0).clip(0, 2)
    d["s1_credited_rate"] = (d[f"{S1} (credited)"] / enr).fillna(0).clip(0, 5)
    d["fees_or_debt_flag"] = ((d["Tuition fees up to date"] == 0) | (d["Debtor"] == 1)).astype(int)
    return d


def _prep(num_cols, cat_cols):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=15, sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")


def make_model(kind, num_cols, cat_cols):
    """kind: lr (chưa cân bằng) | lr_bal (class_weight=balanced) | hgb (gradient boosting, balanced)."""
    if kind == "lr":
        clf = LogisticRegression(max_iter=3000)
    elif kind == "lr_bal":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=4,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("fe", FunctionTransformer(add_features)),
                     ("prep", _prep(num_cols, cat_cols)), ("clf", clf)])


def split_cols(raw_cols, with_fe):
    cat = [c for c in raw_cols if c in NOMINAL]
    num = [c for c in raw_cols if c not in NOMINAL] + (FE_COLS if with_fe else [])
    return num, cat


# ---------------------------------------------------------------- chỉ số
def predict_with_threshold(proba, t=None):
    """argmax; nếu t được đặt: P(Dropout) >= t -> Dropout, ngược lại chọn lớp tốt nhất trong hai lớp còn lại."""
    p = proba.copy()
    if t is not None:
        p[:, 0] = np.where(p[:, 0] >= t, 2.0, -1.0)
    return np.array(CLASSES)[p.argmax(1)]


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    tp, fn = cm[0, 0], cm[0, 1:].sum()
    fp, tn = cm[1:, 0].sum(), cm[1:, 1:].sum()
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro")),
        "recall_dropout": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "precision_dropout": float(tp / (tp + fp)) if tp + fp else 0.0,
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),   # FPR: SV không bỏ học bị cảnh báo
        "false_discovery_rate": float(fp / (tp + fp)) if tp + fp else 0.0,         # trong số được cảnh báo, bao nhiêu sai
        "accuracy": float(np.trace(cm) / cm.sum()),
        "confusion_matrix": cm.tolist(),
    }
