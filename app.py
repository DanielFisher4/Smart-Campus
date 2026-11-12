"""Giao diện Streamlit (Đã tối ưu hóa tốc độ): streamlit run src/app.py"""
import pandas as pd
import streamlit as st
from risk_demo import DISCLAIMER, S1, load_bundle, predict_risk

st.set_page_config(
    page_title="Smart Campus AI – Cảnh báo sớm nguy cơ bỏ học", layout="wide"
)
st.title("AI cảnh báo sớm nguy cơ sinh viên bỏ học")
st.caption(
    "Dữ liệu giả định – không nhập tên/MSSV thật. "
    + DISCLAIMER
)


# Sử dụng cache để không phải load lại mô hình liên tục, giúp tăng tốc độ
@st.cache_resource
def cached_load_bundle():
  return load_bundle()


b = cached_load_bundle()
o = b["options"]

# Tạo thanh điều hướng chọn chế độ sử dụng
tab1, tab2 = st.tabs(
    ["🔍 Đánh giá từng sinh viên", "📁 Xử lý file danh sách lớn (CSV/Excel)"]
)

with tab1:
  st.subheader("Đánh giá nguy cơ cho một trường hợp cụ thể")
  c1, c2 = st.columns(2)
  with c1:
    course = st.selectbox(
        "Ngành (mã)",
        o["Course"],
        index=(
            o["Course"].index(b["defaults"]["Course"])
            if b["defaults"]["Course"] in o["Course"]
            else 0
        ),
    )
    mode = st.selectbox("Diện tuyển sinh (mã)", o["Application mode"])
    adm = st.slider("Điểm đầu vào", 90.0, 200.0, 125.0)
    age = st.slider("Tuổi nhập học", 17, 60, 19)
    sch = st.checkbox("Có học bổng")
    fee = st.checkbox("Học phí đúng hạn", True)
    debt = st.checkbox("Nợ học phí")
  with c2:
    enr = st.slider("Số môn đăng ký HK1", 0, 12, 6)
    ev = st.slider("Số lần đánh giá HK1", 0, 20, 6)
    ap = st.slider("Số môn đạt HK1", 0, 12, 4)
    gr = st.slider("Điểm TB HK1 (0–20)", 0.0, 20.0, 12.0)
    noev = st.slider("Số môn chưa có đánh giá HK1", 0, 12, 0)

  if st.button("Đánh giá nguy cơ"):
    r = predict_risk(
        b,
        {
            "Course": course,
            "Application mode": mode,
            "Admission grade": adm,
            "Age at enrollment": age,
            "Scholarship holder": int(sch),
            "Tuition fees up to date": int(fee),
            "Debtor": int(debt),
            f"{S1} (enrolled)": enr,
            f"{S1} (evaluations)": ev,
            f"{S1} (approved)": ap,
            f"{S1} (grade)": gr,
            f"{S1} (without evaluations)": noev,
        },
    )
    st.subheader(f"Mức nguy cơ: {r['level']}")
    st.write({k: round(v, 3) for k, v in r["proba"].items()})
    st.write(
        "**Yếu tố ảnh hưởng nhất:** "
        + "; ".join(f"{n} ({d:+.2f})" for n, d in r["drivers"])
    )
    st.info("**Khuyến nghị hỗ trợ:** " + r["recommendation"])
    if r["uncertain"]:
      st.warning("Không nên kết luận chắc chắn: dự đoán gần ranh giới.")

with tab2:
  st.subheader(
      "Tải lên file danh sách sinh viên (Định dạng .csv, .xlsx hoặc .xls)"
  )
  uploaded_file = st.file_uploader(
      "Chọn file dữ liệu của bạn", type=["csv", "xlsx", "xls"]
  )

  if uploaded_file is not None:
    try:
      # Đọc file
      if uploaded_file.name.endswith(".csv"):
        df_input = pd.read_csv(
            uploaded_file, sep=None, engine="python", encoding="utf-8-sig"
        )
      else:
        df_input = pd.read_excel(uploaded_file)

      # Làm sạch tên cột
      df_input.columns = [c.strip() for c in df_input.columns]
      st.success(
          f"Đã tải và đọc thành công {len(df_input)} bản ghi từ file!"
      )

      # --- TỐI ƯU HÓA: Dự đoán hàng loạt bằng Vector hóa (Nhanh gấp nhiều lần) ---
      with st.spinner("Đang xử lý và dự đoán hàng loạt..."):
        # Điền giá trị thiếu (nếu có) bằng giá trị mặc định của model
        row_defaults = b["defaults"]
        X_df = df_input.copy()
        for col in b["raw_columns"]:
          if col not in X_df.columns:
            X_df[col] = row_defaults.get(col, 0)

        # Trích xuất đúng các cột mà mô hình yêu cầu
        X_input = X_df[b["raw_columns"]]

        # Gọi mô hình dự đoán xác suất toàn bộ ma trận cùng lúc
        model = b["model"]
        threshold = b["threshold"]
        probas = model.predict_proba(X_input)
        classes = model.classes_
        dropout_idx = list(classes).index("Dropout")

        results = []
        for idx, prob in enumerate(probas):
          proba_dict = dict(zip(classes, prob))
          pdrop = proba_dict.get("Dropout", 0)
          level = (
              "CAO"
              if pdrop >= threshold
              else ("TRUNG BÌNH" if pdrop >= threshold / 2 else "THẤP")
          )

          rec = {
              "CAO": (
                  "Mời sinh viên trao đổi với cố vấn học tập sớm; rà soát tải"
                  " môn học, tài chính."
              ),
              "TRUNG BÌNH": (
                  "Theo dõi thêm kết quả đầu HK2; gửi thông tin hỗ trợ."
              ),
              "THẤP": "Duy trì theo dõi thông thường.",
          }[level]

          results.append({
              "STT": idx + 1,
              "Mức nguy cơ": level,
              "Xác suất Dropout": round(pdrop, 3),
              "Xác suất Enrolled": round(proba_dict.get("Enrolled", 0), 3),
              "Xác suất Graduate": round(proba_dict.get("Graduate", 0), 3),
              "Khuyến nghị": rec,
          })

        # SỬA LỖI: không set_index("STT") (STT bắt đầu từ 1, còn df_input bắt đầu từ 0
        # -> concat axis=1 sẽ lệch 1 dòng). Ghép theo vị trí với cùng index 0..n-1.
        df_result = pd.DataFrame(results)
        df_full = pd.concat(
            [df_input.reset_index(drop=True), df_result.reset_index(drop=True)],
            axis=1,
        )

      # Hiển thị thống kê tổng quan (Bảng số liệu & Biểu đồ)
      st.markdown("---")
      st.markdown("### 📊 Thống kê tổng quan mức độ nguy cơ")
      col_stat1, col_stat2 = st.columns(2)

      with col_stat1:
        if "Mức nguy cơ" in df_full.columns:
          level_counts = df_full["Mức nguy cơ"].value_counts()
          st.write("Bảng số lượng sinh viên theo mức nguy cơ:")
          st.dataframe(level_counts)

      with col_stat2:
        if "Mức nguy cơ" in df_full.columns:
          st.write("Biểu đồ phân bố mức nguy cơ:")
          st.bar_chart(level_counts)

      # Lọc danh sách sinh viên nguy cơ cao
      st.markdown("---")
      st.markdown("### 🚨 Danh sách sinh viên thuộc diện Nguy cơ Cao")
      if "Mức nguy cơ" in df_full.columns:
        df_high_risk = df_full[df_full["Mức nguy cơ"] == "CAO"]

        if not df_high_risk.empty:
          st.warning(
              f"Tìm thấy {len(df_high_risk)} sinh viên cần đặc biệt quan"
              " tâm!"
          )
          st.dataframe(df_high_risk)

          # Nút tải xuống file danh sách nguy cơ cao
          csv_high_risk = df_high_risk.to_csv(index=False).encode("utf-8-sig")
          st.download_button(
              label="📥 Tải xuống danh sách nguy cơ CAO (.csv)",
              data=csv_high_risk,
              file_name="danh_sach_sinh_vien_nguy_co_cao.csv",
              mime="text/csv",
          )
        else:
          st.success("Không có sinh viên nào nằm trong mức nguy cơ CAO.")

      # Hiển thị toàn bộ bảng kết quả dự đoán chi tiết
      st.markdown("---")
      st.markdown("### 📋 Toàn bộ bảng kết quả dự đoán kèm theo")
      st.dataframe(df_full)

      # Nút tải xuống toàn bộ kết quả
      csv_all = df_full.to_csv(index=False).encode("utf-8-sig")
      st.download_button(
          label="📥 Tải xuống toàn bộ bảng kết quả (.csv)",
          data=csv_all,
          file_name="ket_qua_du_doan_toan_bo.csv",
          mime="text/csv",
      )

    except Exception as e:
      st.error(
          f"Lỗi khi đọc file hoặc xử lý dữ liệu: {e}. Vui lòng kiểm tra lại cấu"
          " trúc cột của file!"
      )