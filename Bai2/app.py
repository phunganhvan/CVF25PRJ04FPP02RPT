import cv2
import numpy as np
import streamlit as st


# ====== Các hàm xử lý ảnh (phiên bản dùng cho web, KHÔNG dùng cv2.imshow) ======

def align_images_core(image, template, max_features=5000, keep_percent=0.2):
    """Căn chỉnh ảnh 'image' theo ảnh mẫu 'template'.
    Trả về: (aligned, matched_vis) hoặc (None, None) nếu lỗi.
    """
    if image is None or template is None:
        return None, None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kpsA, descsA = orb.detectAndCompute(gray_image, None)
    kpsB, descsB = orb.detectAndCompute(gray_template, None)

    if descsA is None or descsB is None:
        return None, None

    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    if len(matches) == 0:
        return None, None

    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keep_percent)
    matches = matches[:max(4, keep)]

    if len(matches) < 4:
        return None, None

    ptsA = np.zeros((len(matches), 2), dtype="float32")
    ptsB = np.zeros((len(matches), 2), dtype="float32")

    for i, m in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    H, mask = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    if H is None:
        return None, None

    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    matched_vis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
    return aligned, matched_vis


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def scan_document_core(image, canny_t1=70, canny_t2=200):
    """Giả lập chức năng scan tài liệu.
    Trả về: (warped, edged, outline, error_message)
    """
    if image is None:
        return None, None, None, "Không đọc được ảnh."

    # Làm việc trên phiên bản đã được resize để ổn định hơn
    resized = image.copy()
    if resized.shape[0] != 500:
        ratio = resized.shape[0] / 500.0
        resized = cv2.resize(resized, (int(resized.shape[1] / ratio), 500))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.4)

    edged = cv2.Canny(gray, canny_t1, canny_t2)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, edged, None, "Không tìm thấy đường biên nào đủ lớn."

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    outline = resized.copy()
    cv2.drawContours(outline, cnts, -1, (0, 0, 255), 2)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return None, edged, outline, "Không tìm thấy contour dạng tài liệu (4 đỉnh). Hãy thử ảnh rõ hơn, nền tương phản hơn."

    cv2.drawContours(outline, [screenCnt], -1, (0, 255, 0), 2)

    # Biến đổi phối cảnh trực tiếp trên ảnh đã resize để tránh sai lệch tỉ lệ
    warped = four_point_transform(resized, screenCnt.reshape(4, 2))
    return warped, edged, outline, None


# ====== Hàm tiện ích cho Streamlit ======


def load_image_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def bgr_to_rgb(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ====== Giao diện Streamlit ======

st.set_page_config(page_title="Bài 2: Ứng dụng vào xử lý ảnh", layout="wide")

# --- Sidebar ---
st.sidebar.title("Bài 2")
st.sidebar.markdown(
    """<div style='font-size: 14px;'>
    <b>Ứng dụng vào xử lý ảnh</b><br>
    Chọn một chức năng bên dưới để thử nghiệm.
    </div>""",
    unsafe_allow_html=True,
)

mode = st.sidebar.radio(
    "Chọn chức năng",
    ["Căn chỉnh ảnh theo ảnh mẫu", "Scan tài liệu (4 điểm)"]
)

# Hướng dẫn sử dụng bên sidebar (gọn, không bị tràn)
with st.sidebar.expander("Hướng dẫn sử dụng", expanded=True):
    if mode == "Căn chỉnh ảnh theo ảnh mẫu":
        st.markdown(
            """- Tải lên **ảnh cần căn chỉnh**.
- Tải lên **ảnh mẫu (chuẩn)**.
- Nhấn **Thực hiện căn chỉnh**.
- Xem ảnh gốc, ảnh mẫu và ảnh đã căn chỉnh ở phần chính."""
        )
    elif mode == "Scan tài liệu (4 điểm)":
        st.markdown(
            """- Tải lên ảnh **tài liệu / tờ giấy**.
- Điều chỉnh **Canny T1** và **Canny T2** nếu cần.
- Nhấn **Thực hiện scan**.
- Xem ảnh gốc, biên Canny, contours và ảnh tài liệu đã làm phẳng."""
        )

# --- Tiêu đề chính ---
st.markdown(
    """<div style='background: linear-gradient(90deg, #1f77b4, #2ca02c); padding: 18px 24px; border-radius: 10px; color: white; margin-bottom: 16px;'>
    <h2 style='margin: 0;'>Bài 2: Ứng dụng vào xử lý ảnh</h2>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div style='padding: 14px 18px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #fafafa; margin-bottom: 20px;'>
    Ứng dụng minh họa hai chức năng chính với OpenCV:
    <ul style='margin-top: 6px; margin-bottom: 0;'>
        <li>Căn chỉnh ảnh theo ảnh mẫu bằng ORB + Homography.</li>
        <li>Scan tài liệu bằng biến đổi phối cảnh 4 điểm.</li>
    </ul>
    </div>""",
    unsafe_allow_html=True,
)


if mode == "Căn chỉnh ảnh theo ảnh mẫu":
    st.subheader("1. Căn chỉnh ảnh theo ảnh mẫu")
    col1, col2 = st.columns(2)

    with col1:
        img_file = st.file_uploader("Ảnh cần căn chỉnh", type=["jpg", "jpeg", "png"], key="img")
    with col2:
        template_file = st.file_uploader("Ảnh mẫu (chuẩn)", type=["jpg", "jpeg", "png"], key="template")

    if st.button("Thực hiện căn chỉnh"):
        if img_file is None or template_file is None:
            st.error("Vui lòng tải lên đủ 2 ảnh.")
        else:
            image = load_image_from_upload(img_file)
            template = load_image_from_upload(template_file)

            aligned, matched_vis = align_images_core(image, template)

            if aligned is None:
                st.error("Không thể căn chỉnh ảnh. Hãy thử ảnh khác hoặc ảnh rõ nét hơn.")
            else:
                st.subheader("Kết quả")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption("Ảnh gốc")
                    st.image(bgr_to_rgb(image))
                with c2:
                    st.caption("Ảnh mẫu")
                    st.image(bgr_to_rgb(template))
                with c3:
                    st.caption("Ảnh sau khi căn chỉnh")
                    st.image(bgr_to_rgb(aligned))

                if matched_vis is not None:
                    st.caption("Minh họa feature matching")
                    st.image(bgr_to_rgb(matched_vis))


elif mode == "Scan tài liệu (4 điểm)":
    st.subheader("2. Scan tài liệu (biến đổi phối cảnh 4 điểm)")

    doc_file = st.file_uploader("Ảnh tài liệu cần scan", type=["jpg", "jpeg", "png"], key="doc")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t1 = st.slider("Ngưỡng Canny T1", 0, 500, 70, 1)
    with col_t2:
        t2 = st.slider("Ngưỡng Canny T2", 0, 500, 200, 1)

    if st.button("Thực hiện scan"):
        if doc_file is None:
            st.error("Vui lòng tải lên ảnh tài liệu.")
        else:
            image = load_image_from_upload(doc_file)
            warped, edged, outline, error_message = scan_document_core(image, t1, t2)

            if error_message is not None:
                st.error(error_message)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.caption("Ảnh gốc")
                st.image(bgr_to_rgb(image))
            with col_b:
                if edged is not None:
                    st.caption("Biên Canny")
                    st.image(edged)
            with col_c:
                if outline is not None:
                    st.caption("Contours / Outline")
                    st.image(bgr_to_rgb(outline))

            if warped is not None:
                st.subheader("Ảnh sau khi 'scan'")
                st.image(bgr_to_rgb(warped))
