import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from configs import settings

# Font mặc định
try:
    DEFAULT_FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
except:
    DEFAULT_FONT = ImageFont.load_default()

def resize_if_needed(image_np):
    """Resize ảnh nếu quá to"""
    h, w = image_np.shape[:2]
    if w > settings.MAX_IMAGE_WIDTH:
        scale = settings.MAX_IMAGE_WIDTH / w
        new_h = int(h * scale)
        return cv2.resize(image_np, (settings.MAX_IMAGE_WIDTH, new_h))
    return image_np

def crop_bbox(image_np, bbox):
    """Crop ảnh theo bbox (x,y,w,h)"""
    x, y, w, h = bbox
    return image_np[y:y+h, x:x+w]

def display_image_with_bboxes(title, image_np, cv_bbox_abs=None, tt_bbox_abs=None):
    """Vẽ và hiển thị ảnh annotate (nếu bật SHOW_IMAGES)"""
    if not settings.SHOW_IMAGES:
        return
    display_img_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(display_img_pil)
    if cv_bbox_abs:
        x, y, w, h = cv_bbox_abs
        draw.rectangle([x, y, x+w, y+h], outline="blue", width=3)
        draw.text((x+2, y+2), "CV", fill="blue", font=DEFAULT_FONT)
    if tt_bbox_abs:
        x, y, w, h = tt_bbox_abs
        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
        draw.text((x+2, y+2), "TT", fill="red", font=DEFAULT_FONT)
    annotated = cv2.cvtColor(np.array(display_img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow(title, annotated)
    cv2.waitKey(1)
