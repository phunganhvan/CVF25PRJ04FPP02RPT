# Bai2 - Xử lý ảnh với OpenCV + Streamlit

Ứng dụng web minh họa hai chức năng chính trong thư mục `Bai2`:

- Căn chỉnh một ảnh theo ảnh mẫu (sử dụng ORB feature matching + Homography)
- Scan tài liệu (biến đổi phối cảnh 4 điểm) để làm phẳng tờ giấy

Ứng dụng web được xây dựng bằng **Streamlit**, có thể chạy cục bộ hoặc deploy lên các nền tảng hỗ trợ Streamlit (ví dụ: Streamlit Community Cloud).

---

## 1. Cấu trúc thư mục chính

```text
project/
  Bai2/
    bai2.py          # Script gốc scan tài liệu (dùng cv2.imshow)
    bai2_2.py        # Script gốc căn chỉnh ảnh theo ảnh mẫu (dùng cv2.imshow)
    app.py           # Ứng dụng web Streamlit (dùng cho deploy)
  requirements.txt   # Các thư viện Python cần cài
  README.md          # Tài liệu hướng dẫn (file này)
```

`app.py` đã được viết lại để không dùng `cv2.imshow` mà hiển thị kết quả trực tiếp trên web thông qua Streamlit.

---

## 2. Cài đặt môi trường (chạy cục bộ)

Yêu cầu: đã cài **Python 3.9+** trên máy.

### Bước 1: Tạo và kích hoạt virtual environment (khuyến nghị)

```bash
# Windows PowerShell
cd path/to/project
python -m venv venv
venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

File `requirements.txt` bao gồm các thư viện chính:

- `streamlit`
- `opencv-python-headless`
- `numpy`

---

## 3. Chạy ứng dụng Streamlit

Từ thư mục gốc `project`:

```bash
streamlit run Bai2/app.py
```

Sau khi chạy lệnh trên, trình duyệt sẽ tự mở (hoặc bạn truy cập đường dẫn được in ra, thường là `http://localhost:8501`).

---

## 4. Hướng dẫn sử dụng giao diện web

Trong app Streamlit (`Bai2/app.py`) có 2 chế độ, chọn ở **sidebar**:

1. **Căn chỉnh ảnh theo ảnh mẫu**
   - Tải lên 2 ảnh:
     - Ảnh cần căn chỉnh.
     - Ảnh mẫu (chuẩn) làm tham chiếu.
   - Bấm nút **"Thực hiện căn chỉnh"**.
   - Ứng dụng sẽ hiển thị:
     - Ảnh gốc.
     - Ảnh mẫu.
     - Ảnh sau khi căn chỉnh.
     - (Tuỳ chọn) Hình minh hoạ kết quả **feature matching**.

2. **Scan tài liệu (4 điểm)**
   - Tải lên một ảnh chụp tài liệu / tờ giấy.
   - Điều chỉnh 2 thanh trượt **Canny T1** và **Canny T2** nếu cần để bắt biên tốt hơn.
   - Bấm nút **"Thực hiện scan"**.
   - Ứng dụng sẽ hiển thị:
     - Ảnh gốc.
     - Ảnh biên sau Canny.
     - Ảnh contours / outline (khi tìm được tứ giác tài liệu).
     - Ảnh tài liệu sau khi đã được "làm phẳng" bằng biến đổi phối cảnh.

Nếu app báo **không tìm thấy contour dạng tài liệu**, hãy thử:

- Chọn ảnh tài liệu có nền tương phản rõ với tờ giấy.
- Ảnh không bị quá tối / quá sáng.
- Điều chỉnh lại ngưỡng Canny T1, T2.

---

## 5. Gợi ý deploy lên Streamlit Community Cloud

1. Đưa toàn bộ thư mục `project` (bao gồm `Bai2`, `requirements.txt`, `README.md`) lên một repository GitHub.
2. Vào trang [https://share.streamlit.io](https://share.streamlit.io) (Streamlit Community Cloud).
3. Kết nối với GitHub, chọn repo của bạn.
4. Ở phần **Main file path**, điền:

   ```
   Bai2/app.py
   ```

5. Deploy và chờ hệ thống build xong. Sau đó bạn sẽ có một đường link web để truy cập ứng dụng.

---

## 6. Ghi chú

- Các file `bai2.py` và `bai2_2.py` giữ nguyên logic gốc sử dụng `cv2.imshow`, phù hợp cho chạy thử trực tiếp bằng Python (không qua web).
- File `Bai2/app.py` là phiên bản thân thiện với web, dùng Streamlit để hiển thị toàn bộ kết quả xử lý ảnh.

Nếu bạn muốn chỉnh thêm giao diện hoặc thêm chức năng, có thể sửa trực tiếp trong `Bai2/app.py`.
