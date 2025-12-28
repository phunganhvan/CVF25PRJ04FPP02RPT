import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np

def order_points(pts):
    # Khởi tạo danh sách 4 điểm tọa độ: [TL, TR, BR, BL]
    rect = np.zeros((4, 2), dtype="float32")

    # Điểm Trên-Trái (Top-Left) sẽ có tổng (x + y) nhỏ nhất
    # Điểm Dưới-Phải (Bottom-Right) sẽ có tổng (x + y) lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR

    # Điểm Trên-Phải (Top-Right) sẽ có hiệu (y - x) nhỏ nhất
    # Điểm Dưới-Trái (Bottom-Left) sẽ có hiệu (y - x) lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL

    # Trả về 4 điểm đã được sắp xếp đúng thứ tự
    return rect

def four_point_transform(image, pts):
    # 1. Lấy danh sách điểm đã sắp xếp
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 2. Tính toán chiều rộng (width) và chiều cao (height) của ảnh mới
    # Chiều rộng = khoảng cách tối đa giữa (BR và BL) hoặc (TR và TL)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Chiều cao = khoảng cách tối đa giữa (TR và BR) hoặc (TL và BL)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 3. Tạo tập điểm đích (Destination Points)

    dst = np.array([
        [0, 0],                 # Top-Left
        [maxWidth - 1, 0],      # Top-Right
        [maxWidth - 1, maxHeight - 1], # Bottom-Right
        [0, maxHeight - 1]],    # Bottom-Left
        dtype="float32")

    # 4. Tính ma trận biến đổi và áp dụng
    # getPerspectiveTransform cần 4 điểm nguồn (rect) và 4 điểm đích (dst)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) 

    return warped

def nothing(x):
    pass


def scan_document(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("Không tìm thấy ảnh!")
        return
    
    # Resize ảnh nhỏ lại để xử lý nhanh hơn và phát hiện cạnh tốt hơn 
    org = image.copy()
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Tiền xử lý
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.4)

    cv2.namedWindow('Canny')

    cv2.createTrackbar('T1', 'Canny', 0, 500, nothing)
    cv2.createTrackbar('T2', 'Canny', 0, 500, nothing)
    edged = None # Phát hiện biên

    while True:
        t1 = cv2.getTrackbarPos('T1', 'Canny')
        t2 = cv2.getTrackbarPos('T2', 'Canny')

        edged = cv2.Canny(gray, t1, t2)
        cv2.imshow('Canny', edged)

        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
            break
    # edged = cv2.Canny(gray, 70, 200)
 
    # Tìm Contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sắp xếp contours theo diện tích giảm dần (để tìm vật thể lớn nhất - tờ giấy)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    # VẼ CÁC CONTOURS TÌM ĐƯỢC ĐỂ KIỂM TRA
    imcopy = image.copy()
    cv2.drawContours(imcopy, cnts, -1, (0, 0, 255), 2)
    cv2.imshow("Contours", imcopy)
    
    # Lặp qua các contours để tìm hình tứ giác (4 đỉnh)
    for c in cnts:
        # Tính chu vi
        peri = cv2.arcLength(c, True)
        # Xấp xỉ đa giác (Approximate Polygon)
        # 0.02 * peri là độ sai số cho phép. Càng lớn thì đa giác càng đơn giản (ít đỉnh)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        print(f"Số đỉnh tìm được: {len(approx)}")
        # Nếu đa giác xấp xỉ có đúng 4 đỉnh -> Khả năng cao là tờ giấy
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print("Không tìm thấy contour dạng tài liệu. Hãy thử chụp trên nền tương phản hơn.")
        return

    # Hiển thị contour tìm được trên ảnh gốc 
    cv2.drawContours(imcopy, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", imcopy)
    print(screenCnt.reshape(4, 2))
        # ta cần nhân lại tỷ lệ (ratio) để áp dụng lên ảnh gốc chất lượng cao
    warped = four_point_transform(org, screenCnt.reshape(4, 2) * ratio)

    
    cv2.imshow("Scanned", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

scan_document('thesvR.jpg')