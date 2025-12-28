import cv2
import numpy as np

def align_images(image_path, template_path, max_features=5000, keep_percent=0.2):
    # 1. Đọc ảnh và chuyển sang Grayscale
    image = cv2.imread(image_path)      # Ảnh chụp (bị nghiêng)
    template = cv2.imread(template_path) # Ảnh mẫu (chuẩn)

    #resize để hiển thị dễ hơn
    ratio_image = image.shape[0] / 500.0
    ratio_template = template.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio_image), 500))
    template = cv2.resize(template, (int(template.shape[1] / ratio_template), 500))

    cv2.imshow("Input Image", image)
    cv2.imshow("Template Image", template)

    if image is None or template is None:
        print("Lỗi: Không tìm thấy ảnh.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 2. Phát hiện đặc trưng (Feature Detection) bằng ORB
    # ORB nhanh và hiệu quả cho các biến đổi hình học cơ bản
    orb = cv2.ORB_create(max_features)
    
    # Tìm keypoints và descriptors
    (kpsA, descsA) = orb.detectAndCompute(gray_image, None)
    (kpsB, descsB) = orb.detectAndCompute(gray_template, None)

    # 3. So khớp đặc trưng (Feature Matching)
    # Dùng Hamming distance cho ORB (nhanh hơn Euclidean)
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # 4. Lọc các cặp điểm tốt nhất
    # Sắp xếp theo khoảng cách (distance càng nhỏ càng giống nhau)
    matches = sorted(matches, key=lambda x: x.distance)

    # Chỉ giữ lại top % điểm tốt nhất (loại bỏ nhiễu)
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    # Kiểm tra xem có đủ điểm để tính toán không (cần tối thiểu 4 điểm)
    if len(matches) < 4:
        print("Không đủ điểm đặc trưng để khớp ảnh!")
        return

    # Vẽ các đường nối điểm giống nhau để kiểm tra (Debug)
    matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
    cv2.imshow("Feature Matching", matchedVis)

    # 5. Tính ma trận Homography
    # Trích xuất tọa độ điểm từ các matches
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt # Tọa độ trên ảnh chụp
        ptsB[i] = kpsB[m.trainIdx].pt # Tọa độ trên ảnh mẫu

    # Dùng RANSAC để loại bỏ các điểm ngoại lai (outliers) sai lệch lớn
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # 6. Biến đổi ảnh (Warp Perspective)
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # Hiển thị kết quả
    cv2.imshow("Result",aligned)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return aligned

align_images('thesvP.jpg','thesv.jpg')