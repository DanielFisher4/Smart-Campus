#!/usr/bin/env python3
"""AI01 - Phân tích, baseline, cải tiến, kiểm thử. Chạy: python src/pipeline.py --data data/data.csv"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import (BASE_CAT, BASE_NUM, CLASSES, SEED, TARGET, allowed_features, load_data,
                    make_model, metrics, predict_with_threshold, split_cols, S1)

warnings.filterwarnings("ignore")
COLORS = {"Dropout": "#C0392B", "Enrolled": "#E6A23C", "Graduate": "#2E86C1"}


def cv_oof(model, X, y, seed=SEED, k=5):
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    oof, f1s = np.zeros((len(y), 3)), []
    for tr, va in skf.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])
        oof[va] = p
        f1s.append(f1_score(y.iloc[va], predict_with_threshold(p), average="macro"))
    return oof, f1s


def pick_threshold(oof, y, target_recall):
    best = None
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        m = metrics(y, predict_with_threshold(oof, t))
        if m["recall_dropout"] >= target_recall:
            best = (float(t), m)          # t lớn nhất vẫn đạt recall mục tiêu -> ít cảnh báo sai nhất
    if best is None:
        ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
        ms = [metrics(y, predict_with_threshold(oof, t)) for t in ts]
        i = int(np.argmax([m["recall_dropout"] for m in ms]))
        best = (float(ts[i]), ms[i])
    return best


def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--target-recall", type=float, default=0.80)
    a = ap.parse_args()
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df = load_data(a.data)
    feats = allowed_features(df)
    banned = [c for c in df.columns if c not in feats and c != TARGET]
    print(f"[Dữ liệu] {df.shape[0]} bản ghi | {len(feats)} đặc trưng được phép | loại: {len(banned)} cột (HK2/vĩ mô)")
    assert not any("2nd sem" in c for c in feats), "Rò rỉ HK2!"

    # ------------------------------------------------ 1. Phân tích dữ liệu
    vc = df[TARGET].value_counts().reindex(CLASSES)
    analysis = {"n": int(len(df)), "class_counts": vc.to_dict(),
                "class_pct": (vc / len(df) * 100).round(2).to_dict(),
                "n_features_allowed": len(feats), "banned_columns": banned,
                "missing_total": int(df[feats].isna().sum().sum())}
    s1a, s1g = f"{S1} (approved)", f"{S1} (grade)"
    analysis["by_class_means"] = df.groupby(TARGET)[[s1a, s1g, "Admission grade", "Age at enrollment",
                                                     "Tuition fees up to date", "Scholarship holder", "Debtor"]].mean().round(3).T.to_dict()
    num_df = df[feats].select_dtypes("number").assign(is_dropout=(df[TARGET] == "Dropout").astype(int))
    corr = num_df.corr()["is_dropout"].drop("is_dropout").sort_values(key=abs, ascending=False).head(8)
    analysis["top_corr_dropout"] = corr.round(3).to_dict()

    plt.figure(figsize=(5.5, 3.8))
    plt.bar(CLASSES, vc.values, color=[COLORS[c] for c in CLASSES])
    for i, v in enumerate(vc.values):
        plt.text(i, v + 20, f"{v} ({v / len(df):.1%})", ha="center", fontsize=9)
    plt.ylabel("Số sinh viên"); plt.title("Phân bố ba lớp"); savefig(out / "figures/class_dist.png")

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for a_, col, ttl in zip(ax, [s1a, s1g], ["Số môn đạt HK1", "Điểm TB HK1"]):
        a_.boxplot([df.loc[df[TARGET] == c, col] for c in CLASSES], patch_artist=True,
                   boxprops=dict(facecolor="#D6EAF8"))
        a_.set_xticks([1, 2, 3], CLASSES)
        a_.set_title(ttl)
    savefig(out / "figures/sem1_by_class.png")

    # ------------------------------------------------ 2. Chia dữ liệu: train (CV chọn mô hình) / test (chỉ dùng 1 lần)
    X, y = df[feats], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"[Chia] train={len(Xtr)} test={len(Xte)} (stratified, seed={SEED})")

    all_num, all_cat = split_cols(feats, with_fe=False)
    fe_num, fe_cat = split_cols(feats, with_fe=True)
    specs = {
        "M0 Baseline (LR, 9 đặc trưng)":           make_model("lr", BASE_NUM, BASE_CAT),
        "M1 +class_weight balanced":               make_model("lr_bal", BASE_NUM, BASE_CAT),
        "M2 +toàn bộ đặc trưng cho phép":          make_model("lr_bal", all_num, all_cat),
        "M3 +đặc trưng kỹ thuật HK1 (LR)":         make_model("lr_bal", fe_num, fe_cat),
        "M4 Gradient Boosting + đặc trưng HK1":    make_model("hgb", fe_num, fe_cat),
    }
    names = list(specs)

    # ------------------------------------------------ 3. CV trên train (chọn mô hình, KHÔNG dùng test)
    cv = {}
    for n in names:
        oof, f1s = cv_oof(specs[n], Xtr, ytr)
        m = metrics(ytr, predict_with_threshold(oof))
        cv[n] = {"cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s)),
                 "oof_recall_dropout": m["recall_dropout"], "oof_false_alarm_rate": m["false_alarm_rate"],
                 "_oof": oof}
        print(f"[CV] {n:42s} macro-F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} recall_D={m['recall_dropout']:.3f}")
    final_name = max(names[1:], key=lambda n: cv[n]["cv_macro_f1_mean"])
    print(f"[Chọn] Mô hình cuối (theo CV macro-F1 trên train): {final_name}")

    thr, thr_m = pick_threshold(cv[final_name]["_oof"], ytr, a.target_recall)
    print(f"[Ngưỡng cảnh báo] P(Dropout) >= {thr:.2f} (OOF recall={thr_m['recall_dropout']:.3f}, FPR={thr_m['false_alarm_rate']:.3f})")

    # ------------------------------------------------ 4. Kiểm thử độc lập trên test (một lần)
    fitted, test = {}, {}
    for n in names:
        fitted[n] = clone(specs[n]).fit(Xtr, ytr)
        test[n] = metrics(yte, predict_with_threshold(fitted[n].predict_proba(Xte)))
    base_n = names[0]
    pte_final = fitted[final_name].predict_proba(Xte)
    test_policy = metrics(yte, predict_with_threshold(pte_final, thr))
    test_base_policy = None
    # Baseline cũng được đặt cùng mục tiêu recall để so sánh công bằng ở mức "cảnh báo"
    tb, _ = pick_threshold(cv[base_n]["_oof"], ytr, a.target_recall)
    test_base_policy = metrics(yte, predict_with_threshold(fitted[base_n].predict_proba(Xte), tb))
    print(f"[Test] baseline macro-F1={test[base_n]['macro_f1']:.4f} | cuối={test[final_name]['macro_f1']:.4f}")

    # bootstrap cặp: chênh lệch macro-F1 (cuối - baseline)
    rng = np.random.default_rng(SEED)
    pb = predict_with_threshold(fitted[base_n].predict_proba(Xte))
    pf = predict_with_threshold(pte_final)
    yv = yte.to_numpy(); diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(yv), len(yv))
        diffs.append(f1_score(yv[i], pf[i], average="macro") - f1_score(yv[i], pb[i], average="macro"))
    boot = {"mean_diff": float(np.mean(diffs)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_diff_gt_0": float(np.mean(np.array(diffs) > 0))}

    # độ vững: 5 lần chia ngẫu nhiên khác (seed khác)
    rob = {"baseline": [], "final": [], "baseline_recall_D": [], "final_recall_D": []}
    for s in range(1, 6):
        a1, a2, b1, b2 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=100 + s)
        for key, n in (("baseline", base_n), ("final", final_name)):
            m = clone(specs[n]).fit(a1, b1)
            mm = metrics(b2, predict_with_threshold(m.predict_proba(a2)))
            rob[key].append(mm["macro_f1"]); rob[key + "_recall_D"].append(mm["recall_dropout"])
    robust = {k: [float(np.mean(v)), float(np.std(v))] for k, v in rob.items()}

    # ------------------------------------------------ 5. Giải thích mô hình & phân tích lỗi
    pi = permutation_importance(fitted[final_name], Xte, yte, scoring="f1_macro", n_repeats=5, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False).head(12)
    plt.figure(figsize=(6.5, 4.2)); imp[::-1].plot.barh(color="#1F4E79")
    plt.xlabel("Giảm macro-F1 khi xáo trộn đặc trưng"); plt.title("Mức quan trọng (permutation, test)")
    savefig(out / "figures/importance.png")

    pol_pred = predict_with_threshold(pte_final, thr)
    groups = {"Giới tính": "Gender", "Học bổng": "Scholarship holder", "Học phí đúng hạn": "Tuition fees up to date",
              "Nợ học phí": "Debtor"}
    sub = []
    te = Xte.assign(y=yv, pred=pol_pred)
    te["Nhóm tuổi (>=25)"] = (te["Age at enrollment"] >= 25).astype(int)
    for gname, col in list(groups.items()) + [("Nhóm tuổi", "Nhóm tuổi (>=25)")]:
        for val, g in te.groupby(col):
            d = g[g.y == "Dropout"]; nd = g[g.y != "Dropout"]
            sub.append({"group": gname, "value": int(val), "n": int(len(g)), "dropout_rate": float(len(d) / len(g)),
                        "recall_dropout": float((d.pred == "Dropout").mean()) if len(d) >= 5 else None,
                        "false_alarm_rate": float((nd.pred == "Dropout").mean()) if len(nd) >= 5 else None})

    # ------------------------------------------------ 6. Hình so sánh / ma trận nhầm lẫn / đánh đổi
    plt.figure(figsize=(8, 3.8)); short = [n.split(" ")[0] for n in names]
    plt.bar(np.arange(5) - .2, [cv[n]["cv_macro_f1_mean"] for n in names], .4,
            yerr=[cv[n]["cv_macro_f1_std"] for n in names], label="CV (train)", color="#85C1E9")
    plt.bar(np.arange(5) + .2, [test[n]["macro_f1"] for n in names], .4, label="Test độc lập", color="#1F4E79")
    plt.xticks(range(5), short); plt.ylabel("Macro-F1"); plt.legend(); plt.title("So sánh mô hình")
    plt.ylim(0.5, max(0.9, plt.ylim()[1])); savefig(out / "figures/model_comparison.png")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.9))
    for a_, (ttl, m) in zip(ax, [("Baseline", test[base_n]), (f"Cải tiến ({final_name.split(' ')[0]}, ngưỡng cảnh báo)", test_policy)]):
        cm = np.array(m["confusion_matrix"]); a_.imshow(cm, cmap="Blues")
        a_.set_xticks(range(3), CLASSES); a_.set_yticks(range(3), CLASSES)
        a_.set_xlabel("Dự đoán"); a_.set_ylabel("Thực tế"); a_.set_title(ttl, fontsize=9)
        for i in range(3):
            for j in range(3):
                a_.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    savefig(out / "figures/confusion.png")

    ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = [metrics(ytr, predict_with_threshold(cv[final_name]["_oof"], t)) for t in ts]
    plt.figure(figsize=(5.8, 4)); plt.plot([c["false_alarm_rate"] for c in curve], [c["recall_dropout"] for c in curve], color="#1F4E79")
    plt.scatter([thr_m["false_alarm_rate"]], [thr_m["recall_dropout"]], color="#C0392B", zorder=3, label=f"Ngưỡng chọn = {thr:.2f}")
    plt.xlabel("Tỷ lệ cảnh báo sai (FPR)"); plt.ylabel("Recall lớp Dropout"); plt.legend()
    plt.title("Đánh đổi recall – cảnh báo sai (OOF, train)"); savefig(out / "figures/tradeoff.png")

    # ------------------------------------------------ 7. Lưu kết quả & mô hình cho demo
    defaults = {c: (float(Xtr[c].median()) if c not in NOMINAL_SET else Xtr[c].mode().iloc[0]) for c in feats}
    options = {c: sorted(Xtr[c].dropna().unique().tolist()) for c in ("Course", "Application mode", "Marital status")}
    joblib.dump({"model": fitted[final_name], "threshold": thr, "raw_columns": feats, "defaults": defaults,
                 "options": options, "final_name": final_name}, "models/final_model.joblib")

    res = {"analysis": analysis, "split": {"train": len(Xtr), "test": len(Xte), "seed": SEED},
           "cv": {n: {k: v for k, v in cv[n].items() if k != "_oof"} for n in names},
           "test": test, "final_name": final_name, "baseline_name": base_n,
           "threshold": thr, "target_recall": a.target_recall, "oof_at_threshold": thr_m,
           "test_final_policy": test_policy, "test_baseline_policy": test_base_policy, "baseline_threshold": tb,
           "bootstrap": boot, "robustness": robust, "subgroups": sub,
           "top_importance": imp.round(4).to_dict()}
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{"model": n, "cv_macro_f1": cv[n]["cv_macro_f1_mean"], "test_macro_f1": test[n]["macro_f1"],
                   "test_recall_dropout": test[n]["recall_dropout"], "test_false_alarm_rate": test[n]["false_alarm_rate"]}
                  for n in names]).to_csv(out / "results_table.csv", index=False)
    print("[Xong] outputs/metrics.json, outputs/results_table.csv, outputs/figures/*, models/final_model.joblib")


from common import NOMINAL as NOMINAL_SET  # noqa: E402

if __name__ == "__main__":
    main()
