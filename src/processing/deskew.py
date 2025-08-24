import cv2
import numpy as np 
import math
# ==============================================================================
#                      *** Xoay, Chỉnh nghiêng trang ***
# ==============================================================================
def rotate_image_safe(image: np.ndarray, angle: float,
    border_val: tuple[int, int, int] | int | None = None,
    border_mode: int = cv2.BORDER_REPLICATE
) -> np.ndarray:
    if image is None or image.size == 0 or abs(angle) < 0.01:
        return image
    try:
        h, w = image.shape[:2]; center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        if border_val is None: border_val = (255,255,255) if image.ndim==3 else 255
        rotated_image = cv2.warpAffine(image,rotation_matrix,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=border_mode,borderValue=border_val)
        return rotated_image if rotated_image is not None and rotated_image.size > 0 else image
    except Exception as e_rotate:
        print(f"Cảnh báo: Lỗi trong quá trình xoay ảnh: {e_rotate}")
        return image

def deskew_page_basic(image_bgr: np.ndarray, angle_thresh_deg: float = 1.0,
    bg_color: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0: return image_bgr
    original_image_copy = image_bgr.copy(); h_orig, w_orig = image_bgr.shape[:2]
    try:
        gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try: thresh_inv_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,10)
        except cv2.error: _, thresh_inv_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        kernel_width = max(15,int(w_orig*0.04)); kernel_height = max(3,int(h_orig*0.005))
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_width,kernel_height))
        dilated_img = cv2.dilate(thresh_inv_img,morph_kernel,iterations=2)
        edges_img = cv2.Canny(dilated_img,50,150,apertureSize=3)
        if cv2.countNonZero(edges_img)==0: return original_image_copy
        hough_line_thresh=max(50,int(w_orig*0.1)); min_line_length_hough=max(40,int(w_orig*0.08)); max_line_gap_hough=max(10,int(w_orig*0.02))
        lines_detected = cv2.HoughLinesP(edges_img,1,np.pi/180,hough_line_thresh,minLineLength=min_line_length_hough,maxLineGap=max_line_gap_hough)
        if lines_detected is None or len(lines_detected)==0: return original_image_copy
        detected_angles_deg = [math.degrees(math.atan2(line[0][3]-line[0][1],line[0][2]-line[0][0]))
                               for line in lines_detected
                               if line[0][2]!=line[0][0] and abs(math.degrees(math.atan2(line[0][3]-line[0][1],line[0][2]-line[0][0])))<30.0]
        if not detected_angles_deg: return original_image_copy
        median_detected_angle = float(np.median(detected_angles_deg))
        if not np.isfinite(median_detected_angle): return original_image_copy
        correction_angle_val = -median_detected_angle
        if abs(correction_angle_val) >= angle_thresh_deg:
            deskewed_img_result = rotate_image_safe(original_image_copy, correction_angle_val, border_val=bg_color)
            return deskewed_img_result
        return original_image_copy
    except Exception as e_deskew_page:
        print(f"Cảnh báo: Lỗi trong quá trình chỉnh nghiêng trang: {e_deskew_page}")
        return original_image_copy
