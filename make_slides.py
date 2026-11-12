#!/usr/bin/env python3
"""Tạo slide (<=15 trang nội dung) từ outputs/metrics.json + outputs/figures. Chạy: python src/make_slides.py"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

# >>> SỬA: tên thành viên + vai trò
TEAM = [("Thành viên A", "Phân tích dữ liệu, EDA, quy tắc chống rò rỉ HK2"),
        ("Thành viên B", "Baseline, cải tiến, kiểm thử độc lập"),
        ("Thành viên C", "Demo, slide, thuyết trình, phản biện")]
BLUE = RGBColor(0x1F, 0x4E, 0x79)
M = json.loads(Path("outputs/metrics.json").read_text(encoding="utf-8"))
F = Path("outputs/figures")
A, T, CV = M["analysis"], M["test"], M["cv"]
base, fin = M["baseline_name"], M["final_name"]
pct = lambda x: f"{x * 100:.1f}%"
f3 = lambda x: f"{x:.3f}"

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def slide(title, sub=None):
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(.6), Inches(.3), Inches(12.1), Inches(.9)).text_frame
    tb.text = title; r = tb.paragraphs[0].runs[0]; r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = BLUE
    if sub:
        p = tb.add_paragraph(); p.text = sub; p.runs[0].font.size = Pt(14); p.runs[0].font.italic = True
    return s


def bullets(s, items, x=.6, y=1.5, w=12.1, h=5.5, size=20):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame; tf.word_wrap = True
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + t; p.runs[0].font.size = Pt(size); p.space_after = Pt(8)


def table(s, rows, x, y, w, colw=None, size=14, rowh=.42):
    sh = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(rowh * len(rows)))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = sh.table.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size); run.font.bold = i == 0
                    if i == 0: run.font.color.rgb = RGBColor(255, 255, 255)
            if i == 0: c.fill.solid(); c.fill.fore_color.rgb = BLUE
    if colw:
        for j, cw in enumerate(colw): sh.table.columns[j].width = Inches(cw)


def pic(s, name, x, y, w=None, h=None):
    s.shapes.add_picture(str(F / name), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


# Tiêu đề
s = prs.slides.add_slide(blank)
tf = s.shapes.add_textbox(Inches(.8), Inches(2.2), Inches(11.7), Inches(3)).text_frame; tf.word_wrap = True
tf.text = "AI01 – AI cảnh báo sớm nguy cơ sinh viên bỏ học"; tf.paragraphs[0].runs[0].font.size = Pt(38)
tf.paragraphs[0].runs[0].font.bold = True; tf.paragraphs[0].runs[0].font.color.rgb = BLUE
for t in ["Smart Campus AI – Trí tuệ nhân tạo vì Trường Đại học thông minh", "Nhóm: " + ", ".join(n for n, _ in TEAM)]:
    p = tf.add_paragraph(); p.text = t; p.runs[0].font.size = Pt(20)

s = slide("1. Bài toán và ranh giới sử dụng")
bullets(s, ["Mục tiêu: nhận diện sớm nhóm sinh viên cần được quan tâm, hỗ trợ học tập.",
            "Phân loại 3 lớp: Dropout / Enrolled / Graduate – chỉ dùng thông tin tuyển sinh + kết quả học kỳ 1.",
            "Nguyên mẫu phân tích dữ liệu: KHÔNG tạo quyết định hành chính tự động.",
            "Chỉ số chính: Macro-F1, Recall(Dropout); bổ sung: tỷ lệ cảnh báo sai.",
            "Dữ liệu giả định, không dùng tên/MSSV thật; dữ liệu nghiên cứu không đại diện cho sinh viên UNETI."])

s = slide("2. Dữ liệu và thông tin được phép biết")
bullets(s, [f"UCI Predict Students' Dropout and Academic Success (id 697): {A['n']} bản ghi, 3 lớp.",
            "Nguồn: archive.ics.uci.edu/dataset/697 (giấy phép CC BY 4.0).",
            f"Giữ {A['n_features_allowed']} đặc trưng: thông tin tuyển sinh, nhân khẩu/tài chính lúc nhập học, kết quả HK1.",
            f"Loại {len(A['banned_columns'])} cột: toàn bộ đặc trưng học kỳ 2 (rò rỉ thông tin tương lai) và 3 chỉ số vĩ mô.",
            "Kiểm tra tự động trong mã: assert không còn cột chứa '2nd sem'.",
            f"Giá trị thiếu trong đặc trưng được dùng: {A['missing_total']} (vẫn có bước impute trung vị/phổ biến nhất)."], size=19)

s = slide("3. Phân tích: tỷ lệ ba lớp")
pic(s, "class_dist.png", .6, 1.4, h=4.6)
bullets(s, [f"{k}: {A['class_counts'][k]} ({A['class_pct'][k]:.1f}%)" for k in A["class_counts"]] +
        ["Lớp Dropout là thiểu số vừa phải → accuracy dễ gây hiểu lầm; dùng Macro-F1 và Recall(Dropout).",
         "Cần chia tập stratified để giữ tỷ lệ ba lớp."], x=7.2, y=1.6, w=5.6, size=17)

s = slide("4. Phân tích: kết quả HK1 liên hệ với nhãn")
pic(s, "sem1_by_class.png", .5, 1.5, w=7.4)
corr = list(A["top_corr_dropout"].items())[:4]
bullets(s, ["Tương quan mạnh nhất với Dropout (|r|):"] + [f"{k}: {v:+.2f}" for k, v in corr] +
        ["Tương quan ≠ nhân quả; chỉ dùng để gợi ý hỗ trợ."], x=8.1, y=1.6, w=4.8, size=15)

s = slide("5. Baseline BTC và thiết kế thực nghiệm")
bullets(s, ["Baseline: Logistic Regression nhiều lớp, chuẩn hóa số, xử lý thiếu, mã hóa one-hot 2 biến danh mục, 9 đặc trưng.",
            f"Chia train/test = {M['split']['train']}/{M['split']['test']} (80/20, stratified, seed {M['split']['seed']}).",
            "Test chỉ dùng MỘT lần ở cuối; mọi lựa chọn mô hình/ngưỡng dựa trên CV 5-fold của train.",
            "Báo cáo: Macro-F1, Recall(Dropout), tỷ lệ cảnh báo sai (FPR), ma trận nhầm lẫn."], size=19)

b = T[base]
s = slide("6. Kết quả baseline")
table(s, [["Chỉ số (test)", "Giá trị"], ["Macro-F1", f3(b["macro_f1"])], ["Recall (Dropout)", f3(b["recall_dropout"])],
          ["Cảnh báo sai (FPR)", f3(b["false_alarm_rate"])], ["Precision (Dropout)", f3(b["precision_dropout"])],
          ["Accuracy", f3(b["accuracy"])]], .6, 1.6, 5.2, size=15)
pic(s, "confusion.png", 6.2, 1.5, w=6.8)

s = slide("7. Các phương án cải tiến")
bullets(s, ["M1: giữ nguyên baseline nhưng class_weight='balanced' (tăng độ nhạy lớp thiểu số).",
            "M2: dùng toàn bộ đặc trưng được phép (không HK2) với one-hot cho biến danh mục.",
            "M3: thêm đặc trưng HK1: tỷ lệ môn đạt, tỷ lệ đánh giá, đạt/lần đánh giá, môn chưa đánh giá, tỷ lệ công nhận, cờ học phí/nợ.",
            "M4: Gradient Boosting (HistGradientBoosting) trên cùng đặc trưng.",
            f"Chọn mô hình theo CV macro-F1 trên train → {fin}.",
            f"Chính sách cảnh báo: ngưỡng P(Dropout) ≥ {M['threshold']:.2f} chọn trên OOF để đạt Recall mục tiêu {M['target_recall']:.0%}."], size=18)

s = slide("8. So sánh có minh chứng (ablation)")
rows = [["Mô hình", "CV macro-F1", "Test macro-F1", "Test Recall(D)", "Test FPR"]]
for n in CV:
    rows.append([n, f"{CV[n]['cv_macro_f1_mean']:.3f}±{CV[n]['cv_macro_f1_std']:.3f}", f3(T[n]["macro_f1"]),
                 f3(T[n]["recall_dropout"]), f3(T[n]["false_alarm_rate"])])
table(s, rows, .5, 1.6, 12.3, colw=[5.1, 2.0, 1.8, 1.8, 1.6], size=14)
pic(s, "model_comparison.png", 3.3, 4.2, h=3.0)

bt, rb = M["bootstrap"], M["robustness"]
s = slide("9. Kiểm thử độc lập")
bullets(s, [f"Hold-out chưa từng dùng khi chọn mô hình: {M['split']['test']} sinh viên.",
            f"Bootstrap 2000 lần: Δ macro-F1 (cải tiến − baseline) = {bt['mean_diff']:+.3f}, KTC95% [{bt['ci95'][0]:+.3f}, {bt['ci95'][1]:+.3f}], P(Δ>0) = {bt['p_diff_gt_0']:.2f}.",
            f"5 lần chia ngẫu nhiên khác: baseline {rb['baseline'][0]:.3f}±{rb['baseline'][1]:.3f} vs cải tiến {rb['final'][0]:.3f}±{rb['final'][1]:.3f} (macro-F1).",
            "Nếu KTC chứa 0, cải tiến chưa có bằng chứng thống kê rõ ràng – nói thẳng trong phản biện.",
            "Không so sánh F1 trực tiếp với đề khác; bộ kiểm thử riêng của BTC có thể cho số khác."], size=18)

p, o = M["test_final_policy"], M["test_baseline_policy"]
s = slide("10. Cảnh báo sai và đánh đổi")
pic(s, "tradeoff.png", .5, 1.4, h=4.3)
table(s, [["Test", "Recall(D)", "FPR", "FDR"],
          ["Baseline (cùng mục tiêu recall)", f3(o["recall_dropout"]), f3(o["false_alarm_rate"]), f3(o["false_discovery_rate"])],
          ["Cải tiến + ngưỡng", f3(p["recall_dropout"]), f3(p["false_alarm_rate"]), f3(p["false_discovery_rate"])]],
      6.6, 1.7, 6.3, colw=[3.3, 1.2, 0.9, 0.9], size=13)
bullets(s, ["FPR: tỷ lệ SV không bỏ học bị cảnh báo. FDR: trong số được cảnh báo, tỷ lệ cảnh báo sai.",
            "Tăng recall → tăng cảnh báo sai; chọn ngưỡng theo năng lực tư vấn của nhà trường."], x=6.6, y=3.6, w=6.3, size=15)

s = slide("11. Demo nhập thông tin giả định")
bullets(s, ["Chạy: python src/risk_demo.py (CLI) hoặc streamlit run src/app.py (giao diện).",
            "Đầu vào: ngành, diện tuyển sinh, điểm đầu vào, tuổi, học bổng/học phí/nợ, số môn đăng ký/đạt/chưa đánh giá, điểm HK1.",
            "Đầu ra: xác suất 3 lớp, mức nguy cơ (Thấp/Trung bình/Cao), 3 yếu tố ảnh hưởng nhất, khuyến nghị hỗ trợ.",
            "Mức nguy cơ: Cao nếu P(Dropout) ≥ ngưỡng; Trung bình nếu ≥ ½ ngưỡng; còn lại Thấp.",
            "Luôn kèm tuyên bố: không phải quyết định hành chính."], size=19)

s = slide("12. Ba tình huống giả định và cách giải thích")
bullets(s, ["TH1 – Ổn định (đạt 6/6 môn, học bổng): nguy cơ Thấp → theo dõi thông thường.",
            "TH2 – Ranh giới (đạt 3/6, tuổi 24): xác suất Dropout và Enrolled sát nhau → KHÔNG kết luận chắc chắn; chỉ theo dõi và hỏi nhu cầu.",
            "TH3 – Nguy cơ cao (0 môn đạt, nợ học phí): nguy cơ Cao → mời trao đổi với cố vấn.",
            "Không nên kết luận khi: dữ liệu bất thường/thiếu, ca nằm sát ngưỡng, hoặc hoàn cảnh ngoài dữ liệu (ốm đau, làm thêm, chuyển ngành).",
            "(Điền số xác suất thật từ lần chạy demo vào lời thuyết trình.)"], size=18)

s = slide("13. Phản biện: hạn chế và rủi ro")
sg = [r for r in M["subgroups"] if r["recall_dropout"] is not None][:6]
rows = [["Nhóm", "Giá trị", "n", "Recall(D)", "FPR"]] + [[r["group"], r["value"], r["n"], f3(r["recall_dropout"]),
                                                          f3(r["false_alarm_rate"]) if r["false_alarm_rate"] is not None else "-"] for r in sg]
table(s, rows, .5, 1.5, 6.3, size=12, rowh=.38)
bullets(s, ["Dữ liệu từ một học viện ở Bồ Đào Nha, không đại diện UNETI → cần hiệu chỉnh lại trước khi dùng.",
            "Hiệu năng khác nhau giữa các nhóm → nguy cơ thiên lệch, cần giám sát công bằng.",
            "Nhãn Enrolled chưa kết thúc; tương quan ≠ nhân quả.",
            "Cảnh báo sai gây gắn nhãn; phải có con người xem xét và bảo mật dữ liệu.",
            "Chưa kiểm định trên dữ liệu ngoài phân phối/nhiều khóa."], x=7.1, y=1.5, w=5.8, size=15)

s = slide("14. Phân chia vai trò và hướng dẫn chạy")
table(s, [["Thành viên", "Nhiệm vụ"]] + [list(t) for t in TEAM], .6, 1.5, 12.1, colw=[3, 9.1], size=15)
bullets(s, ["pip install -r requirements.txt", "python src/pipeline.py --data data/data.csv", "python src/risk_demo.py  |  streamlit run src/app.py",
            "python src/make_slides.py"], y=3.9, size=17)

s = slide("15. Tài liệu tham khảo")
bullets(s, ["Realinho, V. et al. (2021). Predict Students' Dropout and Academic Success. UCI ML Repository, doi:10.24432/C5MC89.",
            "Realinho, V. et al. (2022). Predicting Student Dropout and Academic Success. Data 7(11), 146.",
            "scikit-learn (Logistic Regression, HistGradientBoosting), pandas, matplotlib, Streamlit, python-pptx.",
            "Mọi thư viện mã nguồn mở; không dùng API trả phí/GPU; không sao chép mã từ nguồn khác."], size=18)

prs.save("outputs/AI01_slides.pptx"); print("Đã tạo outputs/AI01_slides.pptx")
