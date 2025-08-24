from models import yolo_wrapper, ocr_easyocr
from processing import image_utils

def extract_tables(page_img, yolo_model, ocr_reader):
    detections = yolo_wrapper.detect_tables(yolo_model, page_img)
    results = []
    for det in detections:
        crop_img = image_utils.crop_bbox(page_img, det["bbox"])
        ocr_text = ocr_easyocr.run_ocr(ocr_reader, crop_img)
        results.append({"bbox": det["bbox"], "text": ocr_text})
    return results
