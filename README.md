# AI01 – AI cảnh báo sớm nguy cơ sinh viên bỏ học (Smart Campus AI)

Nguyên mẫu phân tích dữ liệu; KHÔNG tạo quyết định hành chính tự động. Dữ liệu: UCI "Predict Students' Dropout and Academic Success" (id 697, 4.424 bản ghi, 3 lớp). Chỉ dùng thông tin tuyển sinh + kết quả HK1; mã tự loại mọi cột HK2 (có `assert`) và 3 chỉ số vĩ mô.

## Chạy (không cần GPU, không cần API trả phí)
```bash
pip install -r requirements.txt
# Tải data.csv từ https://archive.ics.uci.edu/dataset/697 vào data/data.csv (nếu không có, mã tự tải bằng ucimlrepo)
python src/pipeline.py --data data/data.csv      # phân tích + baseline + cải tiến + kiểm thử -> outputs/, models/
python src/risk_demo.py                          # demo CLI với 3 tình huống giả định
streamlit run src/app.py                         # demo giao diện (tùy chọn)
python src/make_slides.py                        # tạo outputs/AI01_slides.pptx bằng số liệu thật
```

## PHẢI KIỂM TRA TRƯỚC KHI NỘP
1. `src/common.py`: `BASE_NUM`, `BASE_CAT` là 9 đặc trưng + 2 biến danh mục của baseline. Hãy thay bằng danh sách BTC công bố nếu khác.
2. `src/make_slides.py`: sửa `TEAM` (tên + vai trò thật).
3. Sau khi chạy, đọc `outputs/metrics.json`; điền xác suất thật của 3 tình huống vào lời thuyết trình.
4. Số liệu trong slide do script tự lấy từ lần chạy của bạn; không có số nào gõ tay.

## Thiết kế thực nghiệm
- Chia 80/20 stratified (seed 42). Chọn mô hình + ngưỡng chỉ bằng CV 5-fold trên train; test dùng một lần.
- Ablation: M0 baseline → M1 class_weight → M2 toàn bộ đặc trưng cho phép → M3 đặc trưng HK1 → M4 Gradient Boosting.
- Ngưỡng cảnh báo P(Dropout) chọn trên OOF để đạt Recall(Dropout) mục tiêu (mặc định 0.80, `--target-recall`).
- Kiểm thử độc lập: bootstrap 2000 lần (Δ macro-F1 + KTC95%), 5 lần chia ngẫu nhiên khác, phân tích theo nhóm.
- Chỉ số: Macro-F1, Recall(Dropout), cảnh báo sai (FPR = SV không bỏ học bị cảnh báo; FDR = trong số được cảnh báo, tỷ lệ sai).

## Lưu ý đạo đức
Không dùng tên/MSSV thật; dữ liệu không đại diện sinh viên UNETI; cần con người xem xét mọi cảnh báo.
